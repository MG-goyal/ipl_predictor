# app.py
# Run:
# streamlit run app.py

import streamlit as st
import requests

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# =====================================
# TITLE
# =====================================
st.title("🏏 IPL Match Predictor")
st.caption("Predict Score • Defendability • Winner")

st.divider()

# =====================================
# INPUTS
# =====================================
teams = [
    "Royal Challengers Bengaluru",
    "Mumbai Indians",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Delhi Capitals",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

venues = [
    "Chinnaswamy Stadium",
    "Wankhede Stadium",
    "Chepauk",
    "Eden Gardens",
    "Narendra Modi Stadium",
    "Arun Jaitley Stadium"
]

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss Winner", teams)
    batting_team_avg = st.number_input("Batting Team Avg Score", 100.0, 250.0, 180.0)

with col2:
    team2 = st.selectbox("Team 2", teams, index=1)
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    venue_avg_score = st.number_input("Venue Avg Score", 100.0, 250.0, 175.0)
    recent_form = st.number_input("Recent Form Score", 100.0, 250.0, 185.0)

st.divider()

# =====================================
# BUTTON
# =====================================
if st.button("🔥 Predict Match", use_container_width=True):

    payload = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "batting_team_avg": batting_team_avg,
        "venue_avg_score": venue_avg_score,
        "recent_form": recent_form
    }

    try:
        with st.spinner("Analyzing Match..."):

            response = requests.post(
                "https://ipl-predictor-vjyd.onrender.com/full-prediction",
                json=payload,
                timeout=10
            )

        # ---------------------------------
        # SUCCESS
        # ---------------------------------
        if response.status_code == 200:

            result = response.json()

            # If backend returns clean error
            if "error" in result:
                st.error("Prediction failed. Please try again.")
                st.stop()

            st.success("Prediction Complete ✅")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Predicted Score", round(result["predicted_score"]))

            with c2:
                st.metric(
                    "Defend Probability",
                    f'{round(result["defend_probability"]*100)}%'
                )

            with c3:
                st.metric(
                    "Chase Probability",
                    f'{round(result["chase_probability"]*100)}%'
                )

            st.divider()

            st.subheader("🏆 Winner Prediction")
            st.info(result["winner_prediction"])

            st.subheader("📌 Verdict")

            if result["defendable"]:
                st.success("Likely Defendable Score")
            else:
                st.warning("Likely Chaseable Score")

        # ---------------------------------
        # API FAILED
        # ---------------------------------
        # else:
        #     st.error("Server error. Please try again later.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    # ---------------------------------
    # NO RAW LOGS SHOWN
    # ---------------------------------
    except requests.exceptions.ConnectionError:
        st.error("Backend server is not running.")

    except requests.exceptions.Timeout:
        st.error("Server timeout. Try again.")

    # except Exception:
    #     st.error("Something went wrong. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")