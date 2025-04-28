# Phillies_Playoff_Predictor.py

# 1. Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 2. Load your dataset
df = pd.read_csv('Sample_MLB_Team_Data.csv')

# 2. Define your features and label
features = ['Wins', 'Run_Diff', 'Team_WAR', 'Bullpen_ERA', 'RISP_AVG']
label = 'Made_Playoffs'

X = df[features] # Features - everything the model will use to predict
y = df[label] # Label - the outcome we want to predict

# 3. Split the data into training and testing sets
# Training = seasons 2019 - 2022, Testing = 2023 (future prediction)
train_data = df[df['Year'] < 2023]
test_data = df[df['Year'] == 2023]

X_train = train_data[features]
y_train = train_data[label]
X_test = test_data[features]
y_test = test_data[label]

# 4. Scale the features
scaler = StandardScaler()

# Fit the scaler on the training data, then transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled are ready for use in a machine learning model
# 5. Train a machine learning model (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make predictions on the test set
predictions = model.predict(X_test_scaled)
predictions_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of making playoffs
print('Probability of making the playoffs:',predictions_proba)  # Print the probabilities of making playoffs
