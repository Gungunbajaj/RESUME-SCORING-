"""
Resume Scoring & Job Filtering Platform — Flask API
=====================================================
Endpoints:
  POST /predict        — Score a single resume vs job
  POST /rank           — Rank multiple resumes for one job
  POST /shortlist      — Shortlist candidates (3-tier)
  GET  /health         — API health check
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Load model files ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model     = pickle.load(open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb'))
scaler    = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
tfidf_main = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_main.pkl'), 'rb'))
tfidf_cos  = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_cos.pkl'), 'rb'))

# ── Thresholds ───────────────────────────────────────────────
THRESHOLD_STRONG = 0.75
THRESHOLD_MAYBE  = 0.60

# ── Helper functions ─────────────────────────────────────────
def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def education_score(text):
    text = str(text).lower()
    if any(kw in text for kw in ['phd', 'ph.d', 'doctorate']):       return 4
    if any(kw in text for kw in ['master', 'm.sc', 'mba', 'msc', 'm.tech']): return 3
    if any(kw in text for kw in ['bachelor', 'b.sc', 'bsc', 'b.tech', 'btech', 'honours']): return 2
    if any(kw in text for kw in ['diploma', 'associate']):            return 1
    return 0

def get_shortlist_label(score):
    if score >= THRESHOLD_STRONG: return 'Strong Match'
    if score >= THRESHOLD_MAYBE:  return 'Maybe'
    return 'Not Shortlisted'

def extract_features(resume_text, job_text, skills=None, skills_required=None):
    """Extract all features for a resume-job pair"""
    resume_clean = clean_text(resume_text)
    job_clean    = clean_text(job_text)
    combined     = resume_clean + ' ' + job_clean

    # 1. TF-IDF Cosine Similarity
    r_vec = tfidf_cos.transform([resume_clean])
    j_vec = tfidf_cos.transform([job_clean])
    cos_sim = cosine_similarity(r_vec, j_vec)[0][0]

    # 2. Skill Features
    s = set(x.lower().strip() for x in (skills or []))
    r = set(x.lower().strip() for x in (skills_required or []))
    ov = s & r
    skill_overlap_count  = len(ov)
    skill_overlap_ratio  = len(ov) / len(r) if r else 0
    candidate_skill_count = len(s)
    required_skill_count  = len(r)
    missing_skills = len(r - s)
    extra_skills   = len(s - r)

    # 3. Text Stats
    r_words = set(resume_clean.split())
    j_words = set(job_clean.split())
    resume_word_count = len(resume_clean.split())
    common_word_ratio = len(r_words & j_words) / max(len(j_words), 1)

    # 4. Job Title Match
    job_title_words = set(job_clean.split()[:5])
    job_title_match = sum(1 for w in job_title_words if w in resume_clean and len(w) > 2) / max(5, 1)

    # 5. Education
    resume_edu = education_score(resume_text)
    job_edu    = education_score(job_text)
    edu_match  = int(resume_edu >= job_edu)

    # 6. Combine all
    semantic = np.array([[
        cos_sim, skill_overlap_count, skill_overlap_ratio,
        candidate_skill_count, required_skill_count,
        missing_skills, extra_skills, resume_word_count,
        common_word_ratio, job_title_match,
        resume_edu, job_edu, edu_match
    ]])
    tfidf_vec = tfidf_main.transform([combined]).toarray()
    features = np.hstack([semantic, tfidf_vec])
    return scaler.transform(features)

def predict_score(resume_text, job_text, skills=None, skills_required=None):
    features = extract_features(resume_text, job_text, skills, skills_required)
    score = float(np.clip(model.predict(features)[0], 0, 1))
    return score


# ══════════════════════════════════════════════════════════════
# ENDPOINT 1 — /predict
# Score a single resume against a job
# ══════════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    """
    Request body:
    {
        "resume_text": "...",
        "job_text": "...",
        "skills": ["Python", "SQL"],           # optional
        "skills_required": ["Python", "AWS"]   # optional
    }
    """
    data = request.get_json()

    if not data or 'resume_text' not in data or 'job_text' not in data:
        return jsonify({'error': 'resume_text and job_text are required'}), 400

    score = predict_score(
        data['resume_text'],
        data['job_text'],
        data.get('skills', []),
        data.get('skills_required', [])
    )

    label = get_shortlist_label(score)

    # Skill gap analysis
    s = set(x.lower().strip() for x in data.get('skills', []))
    r = set(x.lower().strip() for x in data.get('skills_required', []))
    missing = list(r - s)
    matched = list(s & r)

    return jsonify({
        'score': round(score, 4),
        'percentage': f"{score*100:.1f}%",
        'shortlist_label': label,
        'skill_analysis': {
            'matched_skills': matched,
            'missing_skills': missing,
            'match_ratio': round(len(s&r)/len(r), 3) if r else None
        }
    })


# ══════════════════════════════════════════════════════════════
# ENDPOINT 2 — /rank
# Rank multiple resumes for one job (recruiter view)
# ══════════════════════════════════════════════════════════════
@app.route('/rank', methods=['POST'])
def rank():
    """
    Request body:
    {
        "job_text": "...",
        "skills_required": ["Python", "SQL"],
        "resumes": [
            {"id": "r1", "resume_text": "...", "skills": ["Python"]},
            {"id": "r2", "resume_text": "...", "skills": ["Java"]},
            ...
        ]
    }
    """
    data = request.get_json()

    if not data or 'job_text' not in data or 'resumes' not in data:
        return jsonify({'error': 'job_text and resumes are required'}), 400

    job_text        = data['job_text']
    skills_required = data.get('skills_required', [])
    resumes         = data['resumes']

    if len(resumes) == 0:
        return jsonify({'error': 'At least 1 resume required'}), 400

    results = []
    for resume in resumes:
        score = predict_score(
            resume.get('resume_text', ''),
            job_text,
            resume.get('skills', []),
            skills_required
        )
        results.append({
            'id': resume.get('id', 'unknown'),
            'score': round(score, 4),
            'percentage': f"{score*100:.1f}%",
            'shortlist_label': get_shortlist_label(score)
        })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results, 1):
        r['rank'] = i

    summary = {
        'total_resumes': len(results),
        'strong_match': sum(1 for r in results if r['shortlist_label'] == 'Strong Match'),
        'maybe': sum(1 for r in results if r['shortlist_label'] == 'Maybe'),
        'not_shortlisted': sum(1 for r in results if r['shortlist_label'] == 'Not Shortlisted'),
        'top_candidate_id': results[0]['id'] if results else None
    }

    return jsonify({
        'summary': summary,
        'ranked_resumes': results
    })


# ══════════════════════════════════════════════════════════════
# ENDPOINT 3 — /shortlist
# Just return shortlisted candidates above threshold
# ══════════════════════════════════════════════════════════════
@app.route('/shortlist', methods=['POST'])
def shortlist():
    """
    Same as /rank but returns only shortlisted candidates
    Optional: custom threshold
    {
        "job_text": "...",
        "skills_required": [...],
        "threshold": 0.65,         # optional, default 0.60
        "resumes": [...]
    }
    """
    data = request.get_json()

    if not data or 'job_text' not in data or 'resumes' not in data:
        return jsonify({'error': 'job_text and resumes are required'}), 400

    threshold = float(data.get('threshold', THRESHOLD_MAYBE))
    job_text  = data['job_text']
    skills_required = data.get('skills_required', [])

    results = []
    for resume in data['resumes']:
        score = predict_score(
            resume.get('resume_text', ''),
            job_text,
            resume.get('skills', []),
            skills_required
        )
        if score >= threshold:
            results.append({
                'id': resume.get('id', 'unknown'),
                'score': round(score, 4),
                'percentage': f"{score*100:.1f}%",
                'shortlist_label': get_shortlist_label(score)
            })

    results.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({
        'threshold_used': threshold,
        'total_submitted': len(data['resumes']),
        'total_shortlisted': len(results),
        'shortlisted_candidates': results
    })


# ══════════════════════════════════════════════════════════════
# ENDPOINT 4 — /health
# ══════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'GradientBoostingRegressor',
        'features': 513,
        'thresholds': {
            'strong_match': THRESHOLD_STRONG,
            'maybe': THRESHOLD_MAYBE
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
