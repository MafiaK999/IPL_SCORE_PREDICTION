import math
import numpy as np
import pickle
import streamlit as st

# SET PAGE WIDE
st.set_page_config(page_title='IPL_Score_Predictor', layout="centered")

# Get the ML model
filename = 'ipl_score_prediction_old.pkl'
model = pickle.load(open(filename, 'rb'))

# Title of the page with CSS
st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor </h1>", unsafe_allow_html=True)

# Add background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add description
with st.expander("Description"):
    st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match. To make sure the model results accurate score and some reliability the minimum no. of current overs considered is greater than 5 overs.""")

# SELECT THE BATTING TEAM
batting_team = st.selectbox('Select the Batting Team ', ('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'))

# Initialize the prediction array
prediction_array = []
# Batting Team
if batting_team == 'Chennai Super Kings':
    prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
elif batting_team == 'Delhi Daredevils':
    prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
elif batting_team == 'Kings XI Punjab':
    prediction_array += [0, 0, 1, 0, 0, 0, 0, 0]
elif batting_team == 'Kolkata Knight Riders':
    prediction_array += [0, 0, 0, 1, 0, 0, 0, 0]
elif batting_team == 'Mumbai Indians':
    prediction_array += [0, 0, 0, 0, 1, 0, 0, 0]
elif batting_team == 'Rajasthan Royals':
    prediction_array += [0, 0, 0, 0, 0, 1, 0, 0]
elif batting_team == 'Royal Challengers Bangalore':
    prediction_array += [0, 0, 0, 0, 0, 0, 1, 0]
elif batting_team == 'Sunrisers Hyderabad':
    prediction_array += [0, 0, 0, 0, 0, 0, 0, 1]

# SELECT BOWLING TEAM
bowling_team = st.selectbox('Select the Bowling Team ', ('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'))
if bowling_team == batting_team:
    st.error('Bowling and Batting teams should be different')

# Bowling Team
if bowling_team == 'Chennai Super Kings':
    prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
elif bowling_team == 'Delhi Daredevils':
    prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
elif bowling_team == 'Kings XI Punjab':
    prediction_array += [0, 0, 1, 0, 0, 0, 0, 0]
elif bowling_team == 'Kolkata Knight Riders':
    prediction_array += [0, 0, 0, 1, 0, 0, 0, 0]
elif bowling_team == 'Mumbai Indians':
    prediction_array += [0, 0, 0, 0, 1, 0, 0, 0]
elif bowling_team == 'Rajasthan Royals':
    prediction_array += [0, 0, 0, 0, 0, 1, 0, 0]
elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array += [0, 0, 0, 0, 0, 0, 1, 0]
elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array += [0, 0, 0, 0, 0, 0, 0, 1]

col1, col2 = st.columns(2)

# Enter the Current Ongoing Over
with col1:
    overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs - math.floor(overs) > 0.5:
        st.error('Please enter valid over input as one over only contains 6 balls')
with col2:
    # Enter Current Run
    runs = st.number_input('Enter Current runs', min_value=0, max_value=354, step=1, format='%i')

# Wickets Taken till now
wickets = st.slider('Enter Wickets fallen till now', 0, 9)
wickets = int(wickets)

col3, col4 = st.columns(2)

with col3:
    # Runs in last 5 overs
    runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1, format='%i')

with col4:
    # Wickets in last 5 overs
    wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1, format='%i')

# Get all the data for predicting
prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
prediction_array = np.array([prediction_array])
predict = model.predict(prediction_array)

if st.button('Predict Score'):
    # Calculate current run rate
    current_run_rate = runs / math.floor(overs) if math.floor(overs) > 0 else 0
    
    # Projected score based on current run rate for 20 overs
    projected_score = int(current_run_rate * 20)

    # Adjust the prediction based on the number of wickets and overs
    if overs < 10:
        if wickets == 0:
            decrease_percentage = 0.02
            projected_score = int(projected_score * (1 - decrease_percentage)); # 3% reduction for 1-2 wickets
        elif wickets == 1:
            decrease_percentage = 0.035
            projected_score = int(projected_score * (1 - decrease_percentage))
        elif wickets == 2:
            decrease_percentage = 0.055
            projected_score = int(projected_score * (1 - decrease_percentage))  # 5% reduction for 3-4 wickets
        elif wickets == 3:
            decrease_percentage = 0.085
            projected_score = int(projected_score * (1 - decrease_percentage))  # 10% reduction for 5-6 wickets
        elif wickets == 4:
            decrease_percentage = 0.1
            projected_score = int(projected_score * (1 - decrease_percentage))  # 15% reduction for 7-8 wickets
        elif wickets == 5:
            decrease_percentage = 0.
            projected_score = int(projected_score * (1 - decrease_percentage))  # 50% reduction for 9 wickets, indicating limited scoring potential
        elif wickets == 6:
            decrease_percentage = 0.45
            projected_score = int(projected_score * (1 - decrease_percentage))       
        elif wickets == 7:
            decrease_percentage = 0.95
            projected_score = int(projected_score * (1 - decrease_percentage))
        elif wickets == 8:
            decrease_percentage = 1.3
            projected_score = int(projected_score * (1 - decrease_percentage))
        elif wickets == 9:
            decrease_percentage = 1.6
            projected_score = int(projected_score * (1 - decrease_percentage))
    else:
        # Standard reduction based on wickets if more than 10 overs
        if wickets < 4:
            decrease_percentage = 0.04
            projected_score = int(projected_score * (1 - decrease_percentage)) * wickets  # 1% to 4% reduction for less than 4 wickets
        elif 4 <= wickets <= 7:
            decrease_percentage = 0.1
            projected_score = int(projected_score * (1 - decrease_percentage))  # 10% reduction for 4-7 wickets
        else:
            decrease_percentage = 0.15
            projected_score = int(projected_score * (1 - decrease_percentage))  # 15% reduction for more than 7 wickets

    # Apply the reduction to the projected score
   # projected_score = int(projected_score * (1 - decrease_percentage))

    # Ensure the predicted score is at least one run more than the current runs
    if projected_score <= runs:
        projected_score = runs + 1  # Ensure it's at least one run more than the input runs

    # Limit the scoring potential when there is only one wicket left
    if wickets == 9:
        runs_left = 10 * (20 - math.floor(overs))  # Assuming the last batsman can hit a few runs
        projected_score = min(projected_score, runs + runs_left)  # Cap it based on current runs plus remaining potential
        projected_score = max(projected_score, runs + 1)  # Ensure at least 1 run more than current runs

    # Define the range based on the projected score (±5 runs)
    lower_bound = max(0, projected_score - 5)
    upper_bound = projected_score + 5

    # Display the predicted score range
    st.success(f'**PREDICTED MATCH SCORE:** {lower_bound} to {upper_bound}')



