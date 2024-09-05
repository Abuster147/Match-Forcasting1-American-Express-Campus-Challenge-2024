# Match Forecasting | American Express Campus Challenge 2024

## Overview

This project focuses on predicting team performance and match outcomes using statistical models and machine learning techniques. The model generates team ratings using ridge regression and mixed-effects models and then utilizes these ratings to predict offensive/defensive efficiency and game outcomes. Finally, the results are used to simulate tournament win probabilities.

### Key Features:
- **Ridge Regression** to generate team ratings based on point difference, offensive/defensive efficiency, and pace of play.
- **Mixed Effects Models** to adjust predictions for opponent strength, home court, and other external factors.
- **XGBoost** and **LightGBM** for team-level predictions (offensive efficiency and pace) and game-level predictions (score difference).
- **Simulations** of 100k possible matchups to predict win probabilities.
- Ranked **16th out of 4000+ participants** by leveraging machine learning techniques such as XGBoost, LightGBM, CatBoost, DBSCAN, and the Adam optimizer.

## Workflow

1. **Team Ratings Generation**:
    - Ridge regression and mixed-effects models predict point differences and offensive/defensive efficiency, adjusting for home court, opponent strength, etc.
    - A matchup adjustment model predicts whether a team will play up or down to its competition.

2. **Team-Level Efficiency and Pace Prediction**:
    - Using XGBoost to predict offensive efficiency (points/possession) and pace (possessions/game) for each team.

3. **Game-Level Score Difference Prediction**:
    - Game-level score differences are predicted using XGBoost based on team ratings and efficiency predictions.

4. **Win Probability Simulations**:
    - A simple GLM is used to convert predicted score differences into win probabilities.
    - 100k simulations are run to predict tournament outcomes.

## Model Performance

- **16th out of 4000+ participants**: Achieved top ranking in a data science competition by applying advanced machine learning models and optimization techniques.

## How to Run the Project on Google Colab

### Prerequisites

Ensure that you have access to Google Colab and the following libraries installed:

```bash
!pip install numpy pandas scikit-learn xgboost lightgbm catboost lme4 glmnet
!git clone https://github.com/your-repo/match-forecasting.git
from google.colab import files
uploaded = files.upload()  # Upload the match_data.csv file
from google.colab import drive
drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/match_data.csv'

import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_csv('match_data.csv')
X = data[['teamA_stats', 'teamB_stats']]  # Features
y = data['point_difference']  # Target
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("Team ratings generated.")

import xgboost as xgb

X_train = data[['team_ratings']]
y_train = data['offensive_efficiency']

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

print("Team-level offensive efficiency predicted.")

import numpy as np
num_simulations = 100000
win_probabilities = []

for i in range(num_simulations):
    simulated_outcomes = model_diff.predict(X_train)  # Example prediction
    win_prob = np.mean(simulated_outcomes > 0)  # Predicting team A's win probability
    win_probabilities.append(win_prob)

print("Tournament win probabilities generated.")



This `README.md` file contains all the relevant sections like project overview, workflow, setup instructions, code snippets, and how to run the project on Google Colab. It should be sufficient for someone to understand the project and execute it on their system.
