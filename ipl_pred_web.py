import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("matches.csv")


df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'season', 'winner']].dropna()
df.replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad'
}, inplace=True)


teams = [
    'Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings',
    'Rajasthan Royals', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Capitals', 'Sunrisers Hyderabad'
]
df = df[
    (df['team1'].isin(teams)) &
    (df['team2'].isin(teams)) &
    (df['winner'].isin(teams))
]


encoders = {}
for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


target_encoder = LabelEncoder()
df['winner'] = target_encoder.fit_transform(df['winner'])


X = df.drop('winner', axis=1)
y = df['winner']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)





st.title("🏏 IPL Match Winner Predictor")


team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
toss_winner = st.selectbox("Who won the toss?", [team1, team2])
toss_decision = st.selectbox("Toss decision", ["bat", "field"])
venues = encoders['venue'].classes_
selected_venue_name = st.selectbox("Match Venue", venues)
season = st.number_input("Season", min_value=2008, max_value=2025, value=2019)

if st.button("Predict Winner"):
    try:
        input_data = pd.DataFrame([{
            'team1': encoders['team1'].transform([team1])[0],
            'team2': encoders['team2'].transform([team2])[0],
            'toss_winner': encoders['toss_winner'].transform([toss_winner])[0],
            'toss_decision': encoders['toss_decision'].transform([toss_decision])[0],
            'venue': encoders['venue'].transform([selected_venue_name])[0],
            'season': season
        }])
        prediction = model.predict(input_data)
        predicted_winner = target_encoder.inverse_transform(prediction)[0]
        st.success(f"🏆 Predicted Winner: {predicted_winner}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
