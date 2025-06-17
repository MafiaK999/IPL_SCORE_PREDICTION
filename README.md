# IPL_SCORE_PREDICTION
- Training the Model:
• Run the train_model.py script once to create/update the model and encoder pickles.
• Ensure your CSV file (ipl_data.csv) is available and that the relevant columns exist (including batsman and bowler).
- Launching the App:
• Run the app.py script using the command streamlit run app.py.
• The app will load the saved model and encoder; update the player lists and teams if needed to match your training dataset.
- Further Enhancements:
• You might want to also add a selection for venue or any other feature you consider important.
• The order and names in the one-hot encoding must match those from your training transformer. The example here uses a temporary DataFrame to leverage the saved encoder.
