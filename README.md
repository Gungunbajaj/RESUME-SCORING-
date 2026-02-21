# RESUME-SCORING-
AI-powered Resume Scoring &amp; Job Filtering Platform. ML backend ranks resumes, scores match (0-1), and shortlists candidates automatically. Built using Gradient Boosting &amp; TF-IDF on 9,544 resume-job pairs. REST API with Flask integrates with MERN Stack. R²: 0.59 | NDCG: 0.99 | F1: 0.86
# Resume Scoring & Job Filtering Platform — API

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Server starts at: `http://localhost:5000`

---

## Folder Structure

```
resume_api/
├── app.py              ← Flask API
├── requirements.txt
└── models/
    ├── model.pkl       ← Trained GradientBoosting model
    ├── scaler.pkl      ← StandardScaler
    ├── tfidf_main.pkl  ← TF-IDF for combined text
    └── tfidf_cos.pkl   ← TF-IDF for cosine similarity
```

---

## Endpoints

### 1. POST /predict
Score a single resume vs job.

**Request:**
```json
{
  "resume_text": "Data Engineer with 4 years experience...",
  "job_text": "Data Engineer Azure\nETL Tools\nSQL...",
  "skills": ["Python", "SQL", "Azure"],
  "skills_required": ["SQL", "ETL", "Power BI"]
}
```
**Response:**
```json
{
  "score": 0.788,
  "percentage": "78.8%",
  "shortlist_label": "Strong Match",
  "skill_analysis": {
    "matched_skills": ["sql"],
    "missing_skills": ["etl", "power bi"],
    "match_ratio": 0.333
  }
}
```

---

### 2. POST /rank
Rank multiple resumes for one job (recruiter view).

**Request:**
```json
{
  "job_text": "AI Engineer...",
  "skills_required": ["Python", "TensorFlow"],
  "resumes": [
    {"id": "r1", "resume_text": "...", "skills": ["Python", "TensorFlow"]},
    {"id": "r2", "resume_text": "...", "skills": ["JavaScript"]}
  ]
}
```
**Response:**
```json
{
  "summary": {
    "total_resumes": 2,
    "strong_match": 1,
    "maybe": 1,
    "not_shortlisted": 0,
    "top_candidate_id": "r1"
  },
  "ranked_resumes": [
    {"rank": 1, "id": "r1", "score": 0.803, "shortlist_label": "Strong Match"},
    {"rank": 2, "id": "r2", "score": 0.674, "shortlist_label": "Maybe"}
  ]
}
```

---

### 3. POST /shortlist
Return only shortlisted candidates above a threshold.

**Request:**
```json
{
  "job_text": "...",
  "skills_required": [...],
  "threshold": 0.65,
  "resumes": [...]
}
```

---

### 4. GET /health
```json
{
  "status": "ok",
  "model": "GradientBoostingRegressor",
  "features": 513,
  "thresholds": {"strong_match": 0.75, "maybe": 0.60}
}
```

---

## Shortlist Labels

| Label | Score Range | Meaning |
|-------|-------------|---------|
| 🟢 Strong Match | >= 0.75 | Definitely interview |
| 🟡 Maybe | 0.60 - 0.75 | Recruiter review |
| 🔴 Not Shortlisted | < 0.60 | Not suitable |

---

## Production Deployment

```bash
# Use gunicorn instead of Flask dev server
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Model Performance
- R² Score: 0.59
- RMSE: 0.107
- Ranking NDCG: 0.991
- Shortlist F1: 0.863

