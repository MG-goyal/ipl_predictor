# =========================
# IMPORT LIBRARIES
# =========================

import pickle
import pandas as pd

# =========================
# LOAD SAVED FILES
# =========================

# Load trained model
model = pickle.load(open('models/winner_model.pkl', 'rb'))

# Load label encoder
encoder = pickle.load(open('models/winner_encoder.pkl', 'rb'))

# Load training columns
columns = pickle.load(open('models/winner_columns.pkl', 'rb'))

# =========================
# PREDICTION FUNCTION
# =========================

def predict_match_winner(input_data):
    """
    Predict match winner between team1 and team2
    """

    # Step 1: Convert input dictionary → DataFrame
    df = pd.DataFrame([input_data])

    # Step 2: Apply One-Hot Encoding
    df = pd.get_dummies(df)

    # Step 3: Align columns with training data
    df = df.reindex(columns=columns, fill_value=0)

    # Step 4: Get probabilities for all teams
    proba = model.predict_proba(df)

    # Step 5: Get team1 and team2
    team1 = input_data['team1']
    team2 = input_data['team2']

    # Step 6: Get their encoded indexes
    team1_idx = encoder.transform([team1])[0]
    team2_idx = encoder.transform([team2])[0]

    # Step 7: Compare probabilities
    if proba[0][team1_idx] > proba[0][team2_idx]:
        return team1
    else:
        return team2