# =========================
# IMPORT LIBRARIES
# =========================
import pickle
import pandas as pd
import numpy as np

# =========================
# LOAD SAVED FILES
# =========================

# Load trained model
score_model = pickle.load(open('models/score_model.pkl', 'rb'))

# Load training columns
score_columns = pickle.load(open('models/score_columns.pkl', 'rb'))


# =========================
# PREDICTION FUNCTION
# =========================

def predict_score(input_data):
    """
    Predict the first-innings target score.

    Required keys in input_data:
        - team1, team2, toss_winner, toss_decision, venue
        - batting_team_avg (float)
        - venue_avg_score  (float)
        - recent_form      (float, optional)

    Returns:
        dict: {
            predicted_score (float),
            batting_team    (str),
            bowling_team    (str)
        }
    """
    # Step 1: Convert input dictionary → DataFrame
    df = pd.DataFrame([input_data])

    # Step 2: Derive batting team from toss decision
    def get_batting_team(row):
        if row['toss_decision'] == 'bat':
            return row['toss_winner']
        return row['team1'] if row['toss_winner'] == row['team2'] else row['team2']

    df['batting_team'] = df.apply(get_batting_team, axis=1)

    # Step 3: Derive bowling team
    df['bowling_team'] = df.apply(
        lambda row: row['team1'] if row['batting_team'] == row['team2'] else row['team2'],
        axis=1
    )

    # Step 4: Create toss feature
    df['is_chasing'] = df['toss_decision'].apply(lambda x: 1 if x == 'field' else 0)

    # Step 5: Create team-venue interaction
    df['team_venue'] = df['batting_team'] + "_" + df['venue']

    # Step 6: Fill recent_form if not provided
    if 'recent_form' not in df.columns or df['recent_form'].isnull().any():
        df['recent_form'] = df.get('batting_team_avg', np.nan)

    # Step 7: Align columns with training data
    df = df.reindex(columns=score_columns, fill_value=np.nan)

    # Step 8: Predict score
    score = float(score_model.predict(df)[0])

    return {
        "predicted_score": round(score, 2),
        "batting_team":    df["batting_team"].iloc[0],
        "bowling_team":    df["bowling_team"].iloc[0],
    }