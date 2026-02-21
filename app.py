# app.py
from flask import Flask, request, render_template, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change to a random secret for production

# Load the model and classes
model = joblib.load('models/gender_model.pkl')
classes = joblib.load('models/classes.pkl')  # ['female', 'male'] or whatever order

# Optional: load ambiguous names list to display in UI
try:
    ambiguous_df = pd.read_csv('data/processed/ambiguous_names.csv')
    ambiguous_names = set(ambiguous_df['name'].unique())
except:
    ambiguous_names = set()

# Questionnaire questions (5 questions, each with a direction)
# direction: 'male' means higher agreement indicates male, 'female' means higher agreement indicates female
questions = [
    {
        'id': 1,
        'text': 'I enjoy working on cars or fixing mechanical things.',
        'direction': 'male'
    },
    {
        'id': 2,
        'text': 'I often use words like "adorable", "lovely", or "cute" to describe things.',
        'direction': 'female'
    },
    {
        'id': 3,
        'text': 'I prefer logical puzzles over emotional stories.',
        'direction': 'male'
    },
    {
        'id': 4,
        'text': 'I am usually the one who plans social gatherings.',
        'direction': 'female'
    },
    {
        'id': 5,
        'text': 'I tend to be more assertive than accommodating.',
        'direction': 'male'
    }
]

# Likert scale mapping (1=Strongly Disagree, 5=Strongly Agree)
def likert_to_score(answer):
    """Convert 1-5 answer to a score between 0 and 1."""
    return (answer - 1) / 4.0

def compute_questionnaire_score(responses):
    """
    responses: list of integers (1-5) for each question, in order.
    Returns a score between 0 and 1 where 1 = strongly male, 0 = strongly female.
    """
    scores = []
    for i, answer in enumerate(responses):
        score = likert_to_score(answer)
        if questions[i]['direction'] == 'male':
            scores.append(score)
        else:
            scores.append(1 - score)  # reverse for female-oriented questions
    return np.mean(scores)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    name = request.form['name'].strip().lower()
    if not name:
        return redirect(url_for('home'))
    
    # Predict using the model
    proba = model.predict_proba([name])[0]
    pred_class = model.predict([name])[0]
    confidence = max(proba)
    
    # Store in session for later refinement
    session['name'] = name
    session['name_proba'] = proba.tolist()  # store as list
    session['pred_class'] = pred_class
    session['confidence'] = confidence
    
    # Determine if the name is ambiguous (optional)
    is_ambiguous = name in ambiguous_names
    
    return render_template('result.html',
                           name=name,
                           pred_class=pred_class,
                           confidence=confidence,
                           proba=proba,
                           classes=classes,
                           is_ambiguous=is_ambiguous)

@app.route('/questionnaire')
def questionnaire():
    # Ensure we have a name stored
    if 'name' not in session:
        return redirect(url_for('home'))
    return render_template('questionnaire.html', questions=questions)

@app.route('/refine', methods=['POST'])
def refine():
    # Get answers from form
    responses = []
    for i in range(1, len(questions)+1):
        ans = request.form.get(f'q{i}')
        if ans is None:
            # If any answer missing, redirect back
            return redirect(url_for('questionnaire'))
        responses.append(int(ans))
    
    # Compute questionnaire score (0 to 1, 1=male)
    q_score = compute_questionnaire_score(responses)
    
    # Retrieve name model probability for male (assuming classes[1] is male)
    # Adjust based on your class order. Let's find index of male.
    male_idx = classes.index('male') if 'male' in classes else 1  # fallback
    name_male_prob = session['name_proba'][male_idx]
    
    # Combine: weighted average (0.6 name, 0.4 questionnaire)
    combined_male_prob = 0.6 * name_male_prob + 0.4 * q_score
    combined_female_prob = 1 - combined_male_prob
    
    # Determine final class
    final_pred = 'male' if combined_male_prob >= 0.5 else 'female'
    final_confidence = max(combined_male_prob, combined_female_prob)
    
    return render_template('refine.html',
                           name=session['name'],
                           initial_pred=session['pred_class'],
                           initial_conf=session['confidence'],
                           final_pred=final_pred,
                           final_conf=final_confidence,
                           combined_probs=[combined_female_prob, combined_male_prob])

if __name__ == '__main__':
    app.run(debug=True)