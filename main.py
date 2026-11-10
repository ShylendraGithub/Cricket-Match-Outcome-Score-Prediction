"""Cricket Match Outcome & Score Prediction.

Pipeline:
    1. Load data (synthetic by default, or your own CSV via --data)
    2. Preprocess + leakage-safe feature engineering (rolling form, venue, H2H)
    3. EDA plots
    4. Random Forest Regressor  -> first-innings score
       Random Forest Classifier -> probability that the team batting first wins
    5. Time-based evaluation, plots, feature importances
    6. predict_match() for new fixtures

Usage:
    python main.py
    python main.py --data my_matches.csv
"""
import argparse
import os
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_generator import generate

OUT = "outputs"
PLOTS = os.path.join(OUT, "plots")
sns.set_theme(style="whitegrid")

CATEGORICAL = ["team_a", "team_b", "venue", "toss_decision", "day_night"]
NUMERIC = ["toss_won_by_a", "a_avg_runs_l5", "b_avg_conceded_l5",
           "a_win_rate_l10", "b_win_rate_l10", "venue_avg_l10",
           "h2h_a_win_rate", "year"]
FEATURES = CATEGORICAL + NUMERIC


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def load_data(path=None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]) if path else generate()
    df = df.drop_duplicates().dropna(subset=["team_a", "team_b", "winner"])
    df = df[df["team_a"] != df["team_b"]]
    df = df.sort_values("date").reset_index(drop=True)
    df["first_innings_score"] = df["first_innings_score"].clip(lower=0)
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering (only uses information available BEFORE each match)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame):
    runs_scored = defaultdict(lambda: deque(maxlen=5))
    runs_conceded = defaultdict(lambda: deque(maxlen=5))
    results = defaultdict(lambda: deque(maxlen=10))
    venue_scores = defaultdict(lambda: deque(maxlen=10))
    h2h = defaultdict(lambda: [0, 0])  # (a_wins, matches) keyed by sorted pair

    g_score = df["first_innings_score"].mean()
    mean = lambda q, default: float(np.mean(q)) if len(q) else default

    feats = []
    for r in df.itertuples():
        pair = tuple(sorted([r.team_a, r.team_b]))
        wins, n = h2h[pair]
        a_h2h_wins = wins if pair[0] == r.team_a else n - wins
        feats.append(dict(
            toss_won_by_a=int(r.toss_winner == r.team_a),
            a_avg_runs_l5=mean(runs_scored[r.team_a], g_score),
            b_avg_conceded_l5=mean(runs_conceded[r.team_b], g_score),
            a_win_rate_l10=mean(results[r.team_a], 0.5),
            b_win_rate_l10=mean(results[r.team_b], 0.5),
            venue_avg_l10=mean(venue_scores[r.venue], g_score),
            h2h_a_win_rate=(a_h2h_wins / n) if n else 0.5,
            year=r.date.year,
        ))
        # --- update state AFTER recording features (prevents leakage) ---
        a_won = int(r.winner == r.team_a)
        runs_scored[r.team_a].append(r.first_innings_score)
        runs_scored[r.team_b].append(r.second_innings_score)
        runs_conceded[r.team_b].append(r.first_innings_score)
        runs_conceded[r.team_a].append(r.second_innings_score)
        results[r.team_a].append(a_won)
        results[r.team_b].append(1 - a_won)
        venue_scores[r.venue].append(r.first_innings_score)
        h2h[pair][1] += 1
        if (pair[0] == r.team_a and a_won) or (pair[0] == r.team_b and not a_won):
            h2h[pair][0] += 1

    X = pd.concat([df[CATEGORICAL].reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    y_score = df["first_innings_score"].values
    y_win = (df["winner"] == df["team_a"]).astype(int).values
    return X, y_score, y_win


# --------------------------------------------------------------------------- #
# 3. EDA
# --------------------------------------------------------------------------- #
def eda_plots(df, X, y_score, y_win):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["first_innings_score"], bins=30, kde=True, ax=ax[0], color="#2a6f97")
    ax[0].set(title="First-innings score distribution", xlabel="Runs")
    order = df.groupby("venue")["first_innings_score"].mean().sort_values().index
    sns.boxplot(data=df, y="venue", x="first_innings_score", order=order, ax=ax[1],
                color="#a9d6e5")
    ax[1].set(title="Scores by venue", xlabel="Runs", ylabel="")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/01_score_distribution_venue.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    yearly = df.groupby(df["date"].dt.year)["first_innings_score"].mean()
    ax[0].plot(yearly.index, yearly.values, marker="o", color="#bc4749")
    ax[0].set(title="Average first-innings score by year", xlabel="Year", ylabel="Runs")
    toss = df.assign(bat_first_won=y_win, decision=df["toss_decision"]) \
             .groupby("decision")["bat_first_won"].mean()
    sns.barplot(x=toss.index, y=toss.values, ax=ax[1], palette="crest", hue=toss.index, legend=False)
    ax[1].set(title="Win rate of team batting first, by toss decision",
              ylabel="Win rate", xlabel="Toss decision", ylim=(0, 1))
    plt.tight_layout(); plt.savefig(f"{PLOTS}/02_trend_toss.png", dpi=130); plt.close()

    corr = X[NUMERIC].assign(score=y_score, bat_first_win=y_win).corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature correlation heatmap")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/03_correlation_heatmap.png", dpi=130); plt.close()

    team_wr = pd.concat([
        df.assign(team=df["team_a"], won=(df["winner"] == df["team_a"])),
        df.assign(team=df["team_b"], won=(df["winner"] == df["team_b"])),
    ]).groupby("team")["won"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=team_wr.values, y=team_wr.index, color="#6a994e")
    plt.axvline(0.5, ls="--", c="grey"); plt.title("Overall team win rate"); plt.xlabel("Win rate")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/04_team_win_rate.png", dpi=130); plt.close()


# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #
def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])


def tuned_rf(model, grid, seed=42):
    return RandomizedSearchCV(
        Pipeline([("prep", preprocessor()), ("rf", model)]),
        param_distributions=grid, n_iter=8, cv=3, random_state=seed, n_jobs=-1,
        scoring="neg_mean_absolute_error" if isinstance(model, RandomForestRegressor)
        else "roc_auc")


GRID = {"rf__n_estimators": [200, 300, 500], "rf__max_depth": [6, 10, 14, None],
        "rf__min_samples_leaf": [2, 5, 10], "rf__max_features": ["sqrt", 0.5, 1.0]}


def feature_names(pipe):
    return pipe.named_steps["prep"].get_feature_names_out()


def plot_importance(pipe, title, fname, top=15):
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_,
                    index=[n.split("__")[1] for n in feature_names(pipe)])
    imp = imp.sort_values().tail(top)
    plt.figure(figsize=(8, 5.5))
    imp.plot.barh(color="#386641")
    plt.title(title); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/{fname}", dpi=130); plt.close()


def main(data_path=None, test_frac=0.2):
    os.makedirs(PLOTS, exist_ok=True)
    df = load_data(data_path)
    print(f"Loaded {len(df)} matches ({df['date'].min().date()} -> {df['date'].max().date()})")

    X, y_score, y_win = engineer_features(df)
    eda_plots(df, X, y_score, y_win)

    # time-based split: train on the past, test on the most recent matches
    cut = int(len(df) * (1 - test_frac))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ys_tr, ys_te = y_score[:cut], y_score[cut:]
    yw_tr, yw_te = y_win[:cut], y_win[cut:]

    # ---- score regression ----
    reg = tuned_rf(RandomForestRegressor(random_state=42), GRID).fit(Xtr, ys_tr)
    pred = reg.predict(Xte)
    baseline = np.full_like(ys_te, ys_tr.mean(), dtype=float)
    print("\n=== First-innings score (Random Forest Regressor) ===")
    print(f"Best params : {reg.best_params_}")
    print(f"MAE  : {mean_absolute_error(ys_te, pred):.2f} runs "
          f"(baseline mean-predictor: {mean_absolute_error(ys_te, baseline):.2f})")
    print(f"RMSE : {np.sqrt(mean_squared_error(ys_te, pred)):.2f}")
    print(f"R^2  : {r2_score(ys_te, pred):.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(ys_te, pred, alpha=0.5, color="#2a6f97")
    lim = [min(ys_te.min(), pred.min()) - 5, max(ys_te.max(), pred.max()) + 5]
    plt.plot(lim, lim, "r--"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Actual score"); plt.ylabel("Predicted score"); plt.title("Actual vs predicted score")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/05_actual_vs_predicted.png", dpi=130); plt.close()
    plot_importance(reg.best_estimator_, "Feature importance - score model", "06_importance_score.png")

    # ---- win probability classification ----
    clf = tuned_rf(RandomForestClassifier(random_state=42, class_weight="balanced"), GRID).fit(Xtr, yw_tr)
    proba = clf.predict_proba(Xte)[:, 1]
    label = (proba >= 0.5).astype(int)
    print("\n=== Win prediction (Random Forest Classifier) ===")
    print(f"Best params : {clf.best_params_}")
    print(f"Accuracy : {accuracy_score(yw_te, label):.3f} "
          f"(majority baseline: {max(yw_te.mean(), 1 - yw_te.mean()):.3f})")
    print(f"ROC-AUC  : {roc_auc_score(yw_te, proba):.3f}")
    print(f"Brier    : {brier_score_loss(yw_te, proba):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        yw_te, label, display_labels=["Bat-2nd wins", "Bat-1st wins"], ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title("Confusion matrix")
    fpr, tpr, _ = roc_curve(yw_te, proba)
    ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(yw_te, proba):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--"); ax[1].legend()
    ax[1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/07_classifier_eval.png", dpi=130); plt.close()
    plot_importance(clf.best_estimator_, "Feature importance - win model", "08_importance_win.png")

    # ---- save models + state needed for new predictions ----
    joblib.dump({"reg": reg.best_estimator_, "clf": clf.best_estimator_, "df": df},
                os.path.join(OUT, "models.joblib"))
    print(f"\nPlots saved in {PLOTS}/ ; models saved in {OUT}/models.joblib")

    # ---- demo ----
    demo = predict_match(reg.best_estimator_, clf.best_estimator_, df,
                         team_a="India", team_b="Australia", venue="Mumbai",
                         toss_winner="India", toss_decision="bat", day_night="night")
    print("\nDemo prediction:", demo)


# --------------------------------------------------------------------------- #
# 5. Predict a new fixture
# --------------------------------------------------------------------------- #
def predict_match(reg, clf, history, team_a, team_b, venue, toss_winner,
                  toss_decision, day_night, date=None):
    """team_a bats first. `history` is the historical dataframe (for form features)."""
    date = pd.Timestamp(date) if date else history["date"].max() + pd.Timedelta(days=1)
    fixture = pd.DataFrame([dict(
        date=date, team_a=team_a, team_b=team_b, venue=venue, toss_winner=toss_winner,
        toss_decision=toss_decision, day_night=day_night,
        first_innings_score=0, second_innings_score=0, winner=team_a)])
    # build features from history + this fixture; the fixture is last so its
    # features only reflect past matches
    X, _, _ = engineer_features(pd.concat([history, fixture], ignore_index=True))
    row = X.iloc[[-1]]
    score = float(reg.predict(row)[0])
    p_a = float(clf.predict_proba(row)[0, 1])
    return {"predicted_score_team_a": round(score),
            f"P({team_a} wins)": round(p_a, 3),
            f"P({team_b} wins)": round(1 - p_a, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with the columns described in data_generator.py")
    main(ap.parse_args().data)
