import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------
# LOAD MODEL & COLUMNS
# ---------------------------
MODEL_PATH = Path("score_model.pkl")
COLUMNS_PATH = Path("score_columns.pkl")

_model = None
_columns = None


def _load_artifacts():
    global _model, _columns
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if _columns is None:
        with open(COLUMNS_PATH, "rb") as f:
            _columns = pickle.load(f)


# ---------------------------
# FEATURE ENGINEERING (mirrors training logic)
# ---------------------------
def _engineer_features(input_dict: dict) -> pd.DataFrame:
    """
    Accepts raw match input and returns a feature-engineered DataFrame
    ready for prediction.

    Required keys in input_dict:
        - team1 (str)
        - team2 (str)
        - toss_winner (str)
        - toss_decision (str): 'bat' or 'field'
        - venue (str)
        - batting_team_avg (float): historical average score of batting team
        - venue_avg_score (float): historical average score at the venue
        - recent_form (float): rolling average of last 5 innings (optional, fallback to batting_team_avg)
    """
    df = pd.DataFrame([input_dict])

    # Derive batting/bowling team
    def get_batting_team(row):
        if row['toss_decision'] == 'bat':
            return row['toss_winner']
        return row['team1'] if row['toss_winner'] == row['team2'] else row['team2']

    df['batting_team'] = df.apply(get_batting_team, axis=1)
    df['bowling_team'] = df.apply(
        lambda row: row['team1'] if row['batting_team'] == row['team2'] else row['team2'],
        axis=1
    )

    # Toss feature
    df['is_chasing'] = df['toss_decision'].apply(lambda x: 1 if x == 'field' else 0)

    # Team-venue interaction
    df['team_venue'] = df['batting_team'] + "_" + df['venue']

    # Fill recent_form fallback
    if 'recent_form' not in df.columns or df['recent_form'].isnull().any():
        df['recent_form'] = df.get('batting_team_avg', np.nan)

    return df


# ---------------------------
# CORE PREDICTION FUNCTION
# ---------------------------
def predict_score(input_dict: dict) -> dict:
    """
    Predict the first-innings target score for a match.

    Parameters
    ----------
    input_dict : dict
        Raw match features. See _engineer_features() for required keys.

    Returns
    -------
    dict with keys:
        - predicted_score (float): rounded predicted runs
        - batting_team (str)
        - bowling_team (str)
    """
    _load_artifacts()

    df = _engineer_features(input_dict)

    # Align columns to training schema (fills missing with NaN)
    df = df.reindex(columns=_columns, fill_value=np.nan)

    score = float(_model.predict(df)[0])

    return {
        "predicted_score": round(score, 2),
        "batting_team": input_dict.get("batting_team") or df["batting_team"].iloc[0],
        "bowling_team": input_dict.get("bowling_team") or df["bowling_team"].iloc[0],
    }


# ---------------------------
# BATCH PREDICTION
# ---------------------------
def predict_score_batch(input_list: list[dict]) -> list[dict]:
    """
    Run predict_score over a list of match dicts.
    Returns a list of result dicts in the same order.
    """
    return [predict_score(inp) for inp in input_list]