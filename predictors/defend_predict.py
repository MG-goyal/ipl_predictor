# =========================
# 🔹 LOAD MODEL
# =========================
import pickle
import pandas as pd

model = pickle.load(open("models/defend_model.pkl", "rb"))
model_columns = pickle.load(open("models/defend_columns.pkl", "rb"))

# =========================
# 🔹 FUNCTION
# =========================
def predict_defendability(match_input, score):
    
    # create dataframe
    df = pd.DataFrame([{
        'team 1': match_input['team1'],
        'team 2': match_input['team2'],
        'venue': match_input['venue'],
        'toss_winner': match_input['toss_winner'],
        'toss_decision': match_input['toss_decision'],
        'first_innings_score': score
    }])
    
    # encode
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # predict
    prob = model.predict_proba(df)[0]
    
    defend_prob = prob[1]
    chase_prob = prob[0]
    
    return defend_prob, chase_prob