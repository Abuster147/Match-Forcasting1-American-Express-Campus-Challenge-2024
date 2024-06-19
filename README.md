# Match-Forcasting1-American-Express-Campus-Challenge-2024
Match Forcasting | American Express Campus Challenge 2024
1) generate team ratings via ridge (glmnet)/mixed effects (lme4) regressions predicting point difference, offense/defense efficiency, offense/defense pace, etc. (i.e. what is team A's impact on offense efficiency, score difference, etc. after controlling for opponent, home court, etc.?). I also fit a 'matchup adjustment' mixed effects model, which tries to predict if a team will play up or down to the competition.

2) use team ratings from 1) to predict team level offensive efficiency (points/possession) and pace (possessions) for each game (via XGBoost)

3) use team ratings from 1) and team level predictions from 2) to predict game level score difference (via XGBoost)

The most important features to predict the team level efficiencies and the game level score differences are the various team ratings from 1). Each of these models are trained after each day of each season. As such, it takes multiple days for the entirety of this modeling framework to run for the first time. A simple GLM converts the predicted game level score difference to predicted win probability. I predicted each possible matchup's win probability (similar to the submission format of previous competitions) and then conducted 100k simulations of each tournament.
