"""
Resume Scoring & Job Filtering Platform — Flask API 
======================================================
Improved score differentiation using penalty logic
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model      = pickle.load(open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb'))
scaler     = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
tfidf_main = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_main.pkl'), 'rb'))
tfidf_cos  = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_cos.pkl'), 'rb'))

THRESHOLD_STRONG = 0.75
THRESHOLD_MAYBE  = 0.60

def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def education_score(text):
    text = str(text).lower()
    if any(kw in text for kw in ['phd', 'ph.d', 'doctorate']):              return 4
    if any(kw in text for kw in ['master', 'm.sc', 'mba', 'msc', 'm.tech']): return 3
    if any(kw in text for kw in ['bachelor', 'b.sc', 'bsc', 'b.tech', 'btech', 'honours']): return 2
    if any(kw in text for kw in ['diploma', 'associate']):                   return 1
    return 0

def apply_penalty(base_score, skills, skills_required):
    """
    Penalty logic — differentiates scores better:
    - No skill match at all → heavy penalty
    - Missing critical skills → medium penalty
    - Good skill match → bonus
    """
    if not skills_required:
        return base_score

    s = set(x.lower().strip() for x in skills)
    r = set(x.lower().strip() for x in skills_required)
    overlap = s & r
    overlap_ratio = len(overlap) / len(r) if r else 0
    missing_ratio = len(r - s) / len(r) if r else 0

    score = base_score

    # Heavy penalty — zero skill match
    if overlap_ratio == 0:
        score = score * 0.55

    # Medium penalty — less than 30% skills match
    elif overlap_ratio < 0.30:
        score = score * (0.65 + overlap_ratio * 0.35)

    # Slight penalty — 30-60% match
    elif overlap_ratio < 0.60:
        score = score * (0.80 + overlap_ratio * 0.20)

    # Bonus — more than 80% skills match
    elif overlap_ratio >= 0.80:
        score = min(score * 1.10, 1.0)

    return float(np.clip(score, 0, 1))

def get_label(score):
    if score >= THRESHOLD_STRONG: return 'Strong Match'
    if score >= THRESHOLD_MAYBE:  return 'Maybe'
    return 'Not Shortlisted'

def predict_score(resume_text, job_text, skills=None, skills_required=None):
    rc = clean_text(resume_text)
    jc = clean_text(job_text)
    combined = rc + ' ' + jc

    cos_sim = cosine_similarity(tfidf_cos.transform([rc]), tfidf_cos.transform([jc]))[0][0]

    s = set(x.lower().strip() for x in (skills or []))
    r = set(x.lower().strip() for x in (skills_required or []))
    ov = s & r

    r_words = set(rc.split())
    j_words = set(jc.split())
    jtw = set(jc.split()[:5])
    jtm = sum(1 for w in jtw if w in rc and len(w) > 2) / max(5, 1)
    re_edu = education_score(resume_text)
    je_edu = education_score(job_text)

    sem = np.array([[
        cos_sim, len(ov), len(ov)/len(r) if r else 0,
        len(s), len(r), len(r-s), len(s-r),
        len(rc.split()),
        len(r_words & j_words) / max(len(j_words), 1),
        jtm, re_edu, je_edu, int(re_edu >= je_edu)
    ]])
    tfidf_vec = tfidf_main.transform([combined]).toarray()
    features  = np.hstack([sem, tfidf_vec])
    base_score = float(np.clip(model.predict(scaler.transform(features))[0], 0, 1))

    # Apply penalty for better differentiation
    final_score = apply_penalty(base_score, skills or [], skills_required or [])
    return final_score


# ══════════════════════════════════════════════════
# ENDPOINT 1 — /predict
# ══════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'resume_text' not in data or 'job_text' not in data:
        return jsonify({'error': 'resume_text and job_text required'}), 400

    score = predict_score(
        data['resume_text'], data['job_text'],
        data.get('skills', []), data.get('skills_required', [])
    )

    s = set(x.lower().strip() for x in data.get('skills', []))
    r = set(x.lower().strip() for x in data.get('skills_required', []))

    return jsonify({
        'score': round(score, 4),
        'percentage': f"{score*100:.1f}%",
        'shortlist_label': get_label(score),
        'skill_analysis': {
            'matched_skills': list(s & r),
            'missing_skills': list(r - s),
            'match_ratio': round(len(s&r)/len(r), 3) if r else None
        }
    })


# ══════════════════════════════════════════════════
# ENDPOINT 2 — /rank
# ══════════════════════════════════════════════════
@app.route('/rank', methods=['POST'])
def rank():
    data = request.get_json()
    if not data or 'job_text' not in data or 'resumes' not in data:
        return jsonify({'error': 'job_text and resumes required'}), 400

    job_text        = data['job_text']
    skills_required = data.get('skills_required', [])
    results = []

    for resume in data['resumes']:
        score = predict_score(
            resume.get('resume_text', ''), job_text,
            resume.get('skills', []), skills_required
        )
        results.append({
            'id': resume.get('id', 'unknown'),
            'score': round(score, 4),
            'percentage': f"{score*100:.1f}%",
            'shortlist_label': get_label(score)
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results, 1):
        r['rank'] = i

    return jsonify({
        'summary': {
            'total_resumes': len(results),
            'strong_match':     sum(1 for r in results if r['shortlist_label'] == 'Strong Match'),
            'maybe':            sum(1 for r in results if r['shortlist_label'] == 'Maybe'),
            'not_shortlisted':  sum(1 for r in results if r['shortlist_label'] == 'Not Shortlisted'),
            'top_candidate_id': results[0]['id'] if results else None
        },
        'ranked_resumes': results
    })


# ══════════════════════════════════════════════════
# ENDPOINT 3 — /shortlist
# ══════════════════════════════════════════════════
@app.route('/shortlist', methods=['POST'])
def shortlist():
    data = request.get_json()
    if not data or 'job_text' not in data or 'resumes' not in data:
        return jsonify({'error': 'job_text and resumes required'}), 400

    threshold       = float(data.get('threshold', THRESHOLD_MAYBE))
    job_text        = data['job_text']
    skills_required = data.get('skills_required', [])
    results = []

    for resume in data['resumes']:
        score = predict_score(
            resume.get('resume_text', ''), job_text,
            resume.get('skills', []), skills_required
        )
        if score >= threshold:
            results.append({
                'id': resume.get('id', 'unknown'),
                'score': round(score, 4),
                'percentage': f"{score*100:.1f}%",
                'shortlist_label': get_label(score)
            })

    results.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({
        'threshold_used':        threshold,
        'total_submitted':       len(data['resumes']),
        'total_shortlisted':     len(results),
        'shortlisted_candidates': results
    })


# ══════════════════════════════════════════════════
# ENDPOINT 4 — /health
# ══════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'GradientBoostingRegressor',
        'version': 'v2 - with penalty logic',
        'features': 513,
        'thresholds': {
            'strong_match': THRESHOLD_STRONG,
            'maybe': THRESHOLD_MAYBE
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
