# Importing Necessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

# Importing dataset
ipl_df = pd.read_csv('ipl_data.csv')
print(f"Dataset successfully imported with shape: {ipl_df.shape}")

# First 5 Rows of Data
print(ipl_df.head())

# Describing the ipl_df
print(ipl_df.describe())

# Information about Each Column
print(ipl_df.info())

# Number of Unique Values in each column
print(ipl_df.nunique())

# Data types of all Columns
print(ipl_df.dtypes)

'''
#Wickets Distribution
sns.displot(ipl_df['wickets'],kde=False,bins=10)
plt.title("Wickets Distribution")

plt.show()

#Runs Distribution
sns.displot(ipl_df['total'],kde=False,bins=10)
plt.title("Runs Distribution")

plt.show()
'''

# Remove irrelevant columns
irrelevant = ['mid', 'date', 'bowler', 'striker', 'non-striker','batsman',]
print(f'Before Removing Irrelevant Columns: {ipl_df.shape}')
ipl_df = ipl_df.drop(irrelevant, axis=1)  # Drop Irrelevant Columns
print(f'After Removing Irrelevant Columns: {ipl_df.shape}')
print(ipl_df.head())


# Consistent Teams List (removed duplicate)
const_teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Delhi Capitals', 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
]
print(f'Before Removing Inconsistent Teams: {ipl_df.shape}')
ipl_df = ipl_df[(ipl_df['batting_team'].isin(const_teams)) & (ipl_df['bowling_team'].isin(const_teams))]
print(f'After Removing Inconsistent Teams: {ipl_df.shape}')
print(f"Consistent Teams: \n{ipl_df['batting_team'].unique()}")
ipl_df.head()

# Remove First 5 Overs of every match
print(f'Before Removing Overs: {ipl_df.shape}')
ipl_df = ipl_df[ipl_df['overs'] >= 5.0]
print(f'After Removing Overs: {ipl_df.shape}')
ipl_df.head()

'''
#Runs Distribution
sns.displot(ipl_df['total'],kde=False,bins=10)
plt.title("Runs Distribution")

plt.show()
'''

# Create a LabelEncoder instance
le = LabelEncoder()

# List of columns to encode
columns_to_encode = ['batting_team', 'bowling_team', 'venue']

# Loop through the specified columns and apply Label Encoding
for col in columns_to_encode:
    ipl_df[col] = le.fit_transform(ipl_df[col])

# One-Hot Encoding
columns_to_encode = ['batting_team', 'bowling_team', 'venue']
columnTransformer = ColumnTransformer(
    [('encoder', OneHotEncoder(), columns_to_encode)],  # Encode specified columns
    remainder='passthrough'  # Keep other columns unchanged
)

# Transform the dataframe
ipl_df_transformed = columnTransformer.fit_transform(ipl_df)

ohe_feature_names = columnTransformer.named_transformers_['encoder'].get_feature_names_out(columns_to_encode)

# Save the Numpy Array in a new DataFrame with transformed columns
cols = [
    'batting_team_Chennai Super Kings','batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab' ,'batting_team_Kolkata Knight Riders',
    'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals', 'batting_team_Royal Challengers Bangalore',
    'batting_team_Sunrisers Hyderabad','bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'batting_team_Kings XI Punjab', 'bowling_team_Kolkata Knight Riders',
    'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals', 'bowling_team_Royal Challengers Bangalore',
    'bowling_team_Sunrisers Hyderabad','runs', 'wickets', 'venue','overs', 'runs_last_5', 'wickets_last_5'
]



# Ensure df is defined here
df = pd.DataFrame(ipl_df, columns=cols)

# Replace NaN values with 0.0
df.fillna(0.0, inplace=True)

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)

print("Encoded Data:")
print(df.head())

# Split features and labels
features = df.drop(['runs'], axis=1)
labels = df['runs']

# Train-test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set: {train_features.shape}\nTesting Set: {test_features.shape}")

features = df.drop(['runs'], axis=1)
labels = df['runs']

# Train-test split
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

model = RandomForestRegressor()
model.fit(train_features, train_labels)

# Evaluate Model
train_score_model = str(model.score(train_features, train_labels) * 100)
test_score_model = str(model.score(test_features, test_labels) * 100)
print(f'Train Score: {train_score_model[:5]}%\nTest Score: {test_score_model[:5]}%')



print("---- Random Forest Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, model.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, model.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, model.predict(test_features)))))

def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model):
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

    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kings XI Punjab':
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

    prediction_array += [runs, wickets, overs, runs_last_5, wickets_last_5]

    return model.predict(np.array(prediction_array).reshape(1, -1))[0]

import pickle
with open('ipl_score_prediction_old.pkl', 'wb') as file:
    pickle.dump(model, file)
