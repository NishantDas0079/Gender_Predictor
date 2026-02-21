# 🎭 GenderSpark – AI-Powered Gender Predictor

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://gender-predictor.onrender.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-lightgrey)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GenderSpark** is a machine learning web application that predicts the likely gender associated with a given first name. It combines a character‑n‑gram model with an optional interactive questionnaire to refine predictions for ambiguous names. Built with Flask and deployed on Render.

🌐 **Live Demo**: [https://gender-predictor.onrender.com](https://gender-predictor.onrender.com)

*(Note: Free tier may spin down after inactivity – first request may take a few seconds.)*

---

## ✨ Features

- 🔮 **Name‑based prediction** – uses a trained Random Forest model with character n‑grams (2‑5 characters) for high accuracy.
- 📋 **Interactive refinement** – 5 optional questions that adjust the prediction for ambiguous names (weighted 60% name model, 40% questionnaire).
- 🌍 **Global dataset** – trained on combined Indian and US SSA names for cross‑cultural performance.
- 🧠 **Smart combination** – when confidence is low (<0.7) or the name is flagged as ambiguous, users can refine via a simple Likert‑scale questionnaire.
- 🎨 **Vibrant, responsive UI** – gradient backgrounds, smooth animations, and a modern tech aesthetic.
- 📡 **REST API endpoint** – programmatic predictions with JSON responses.

---

## 🛠️ Tech Stack

| Component       | Technology                                                                 |
|-----------------|----------------------------------------------------------------------------|
| **Backend**     | Python, Flask, Gunicorn                                                    |
| **ML / NLP**    | scikit‑learn, pandas, numpy, joblib                                        |
| **Frontend**    | HTML5, CSS3 (inline styles), Jinja2                                        |
| **Deployment**  | Render (free tier)                                                         |
| **Version Control** | Git, GitHub                                                             |

---

## 🧠 How It Works

1. **Name Model**: A `TfidfVectorizer` (character n‑grams, 2‑5) + `LogisticRegression` pipeline predicts gender probabilities.
2. **Questionnaire**: 5 questions with directions (male/female) are answered on a 1‑5 Likert scale. A score is computed (0 = strongly female, 1 = strongly male).
3. **Combination**: Final prediction = `0.6 * name_male_prob + 0.4 * questionnaire_score`. A threshold of 0.5 determines the final class.
4. **Ambiguity Handling**: If initial confidence < 0.7 or name exists in the ambiguous list, users are prompted to refine.

---

## 📈 Model Performance

- **Training data**: ~110,000 name‑gender pairs (Indian + US SSA)
- **Test accuracy**: ~87% (baseline Logistic Regression)
- **Feature importance**: Character endings (e.g., "a" → female, "n" → male) are most influential.

| Class   | Precision | Recall | F1‑Score |
|---------|-----------|--------|----------|
| Female  | 0.86      | 0.88   | 0.87     |
| Male    | 0.88      | 0.86   | 0.87     |

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/NishantDas0079/Gender_Predictor.git
cd Gender_Predictor
```

# 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

# 3. Installing Depedencies
```bash
pip install -r requirements.txt
```

# 4. Run the Flask app
```bash
python app.py
```

# 📡 API Usage
You can use the prediction endpoint programmatically.

Endpoint: `https://gender-predictor.onrender.com/predict`

Method: `POST`

Content-Type: `application/json`

# 🤝 Contributing
Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

# 👨‍💻 Author
Nishant Das
GitHub · LinkedIn

