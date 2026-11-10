# Cricket Match Outcome & Score Prediction

Random Forest models that predict **(1) the first-innings score** and **(2) the probability that the team batting first wins**, from historical match data.

**Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Run
```bash
pip install -r requirements.txt
python main.py                      # uses the synthetic dataset
python main.py --data matches.csv   # uses your own data
```
Required CSV columns: `date, team_a (bats first), team_b, venue, toss_winner, toss_decision, day_night, first_innings_score, second_innings_score, winner`
(Real ball-by-ball data from [Cricsheet](https://cricsheet.org) can be aggregated into this format.)

## Pipeline
1. **Preprocessing** - dedupe, drop invalid rows, sort chronologically.
2. **Leakage-safe feature engineering** - every feature for a match uses only matches played *before* it: last-5 runs scored/conceded, last-10 win rate, venue average (last 10), head-to-head win rate, toss, day/night, year.
3. **EDA** (Matplotlib/Seaborn) - score distribution, venue boxplots, yearly trend, toss impact, correlation heatmap, team win rates.
4. **Models** - `RandomForestRegressor` (score) and `RandomForestClassifier` (win probability), one-hot encoding in a `Pipeline`, tuned with `RandomizedSearchCV`.
5. **Evaluation** - chronological train/test split (train on past, test on latest 20%), compared with naive baselines. Metrics: MAE, RMSE, R², accuracy, ROC-AUC, Brier score.
6. **Inference** - `predict_match(...)` returns predicted score and win probabilities for a new fixture.

## Results (synthetic data, test = latest 20%)
| Task | Metric | Model | Baseline |
|---|---|---|---|
| Score | MAE | 15.0 runs | 18.5 |
| Score | R² | 0.28 | 0 |
| Win | Accuracy | 61.3% | 53.3% |
| Win | ROC-AUC | 0.655 | 0.5 |

Cricket is high-variance, so modest, honest gains over baseline are expected. Real data with player-level and in-match features (overs, wickets, run rate) would improve this.

## Ideas to extend
- Live prediction with ball-by-ball state (overs, wickets, current run rate)
- Player-level features (batter strike rates, bowler economy)
- Compare with Gradient Boosting / XGBoost; add probability calibration
- Streamlit dashboard
