# main.py
# Run:
# uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel

from predictors.score_predict import predict_score
from predictors.defend_predict import predict_defendability
from predictors.winner_predict import predict_match_winner

app = FastAPI(title="IPL Prediction API")


# ==================================================
# INPUT SCHEMA
# ==================================================
class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

    batting_team_avg: float
    venue_avg_score: float
    recent_form: float


# ==================================================
# HOME
# ==================================================
@app.get("/")
def home():
    return {"message": "IPL API Running"}


# ==================================================
# FULL PIPELINE
# Score -> Defend -> Winner
# ==================================================
@app.post("/full-prediction")
def full_prediction(data: MatchInput):

    # ---------------------------------
    # COMMON INPUT
    # ---------------------------------
    match = {
        "team1": data.team1,
        "team2": data.team2,
        "venue": data.venue,
        "toss_winner": data.toss_winner,
        "toss_decision": data.toss_decision,
        "batting_team_avg": data.batting_team_avg,
        "venue_avg_score": data.venue_avg_score,
        "recent_form": data.recent_form
    }

    # ---------------------------------
    # STEP 1 : SCORE MODEL
    # ---------------------------------
    score_result = predict_score(match)

    predicted_score = score_result["predicted_score"]
    batting_team = score_result["batting_team"]
    bowling_team = score_result["bowling_team"]

    # ---------------------------------
    # STEP 2 : DEFEND MODEL
    # ---------------------------------
    defend_prob, chase_prob = predict_defendability(
        match,
        predicted_score
    )

    defendable = True if defend_prob > 0.65 else False

    # ---------------------------------
    # STEP 3 : WINNER MODEL
    # ---------------------------------
    winner_input = {
        "team1": data.team1,
        "team2": data.team2,
        "venue": data.venue,
        "toss_winner": data.toss_winner,
        "toss_decision": data.toss_decision,
        "predicted_score": predicted_score,
        "defendable": defendable
    }

    predicted_winner = predict_match_winner(winner_input)

    # ---------------------------------
    # OUTPUT
    # ---------------------------------
    return {
    "predicted_score": float(predicted_score),
    "batting_team": batting_team,
    "bowling_team": bowling_team,

    "defend_probability": float(defend_prob),
    "chase_probability": float(chase_prob),
    "defendable": bool(defendable),

    "winner_prediction": batting_team if defendable else bowling_team
    }