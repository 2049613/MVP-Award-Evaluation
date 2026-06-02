from pathlib import Path

import collections
import collections.abc
import sys

import pandas as pd
import six
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Compatibility patch for older skrules releases.
sys.modules["sklearn.externals.six"] = six
collections.Iterable = collections.abc.Iterable
from skrules import SkopeRules

BASE_DIR = Path(__file__).resolve().parent
LABEL_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "outputs" / "skoperules"

FEATURE_COLS = [
    "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "AST%", "STL%", "BLK%",
    "TOV%", "USG%", "OWS", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P",
    "2PA", "2P%", "eFG%", "FT", "FTA", "TOV", "PF", "PTS", "PER", "TS%",
    "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "DWS", "WS", "WS/48", "OBPM",
    "DBPM", "BPM", "VORP", "win",
]

SCENARIOS = [
    (
        "historical_mvp",
        "MVP_ture.xlsx",
        "Non_MVP_ture.xlsx",
        "full_data_rules.xlsx",
        "MVP_predictions_full.xlsx",
        0.8,
        0.5,
    ),
    (
        "data_driven_top3",
        "MVPtop3_rk.xlsx",
        "NonMVPtop3_rk.xlsx",
        "full_data_MVPlikerules.xlsx",
        "MVPlike_predictions_full.xlsx",
        0.8,
        0.63,
    ),
    (
        "data_driven_mvp",
        "MVP_rk.xlsx",
        "NonMVP_rk.xlsx",
        "full_data_MVPprerules.xlsx",
        "MVPpre_predictions_full.xlsx",
        0.8,
        0.52,
    ),
]


def resolve_label_path(filename):
    path = LABEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required label file not found: {path}")
    return path


def load_and_preprocess_data(mvp_path, non_mvp_path):
    mvp_df = pd.read_excel(mvp_path, engine="openpyxl")
    non_mvp_df = pd.read_excel(non_mvp_path, engine="openpyxl")

    mvp_df["is_MVP"] = 1
    non_mvp_df["is_MVP"] = 0
    df = pd.concat([mvp_df, non_mvp_df], ignore_index=True)

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[FEATURE_COLS]
    y = df["is_MVP"]

    mask = ~X.isnull().any(axis=1)
    if not mask.all():
        print(f"Detected {(~mask).sum()} rows with missing values; removing them.")

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    df_clean = df.loc[mask].reset_index(drop=True)

    return X, y, df_clean


def train_predict_extract_rules_full(X, y, precision_min=0.5, recall_min=0.1):
    model = SkopeRules(
        feature_names=FEATURE_COLS,
        n_estimators=300,
        precision_min=precision_min,
        recall_min=recall_min,
        random_state=100,
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    rules_df = pd.DataFrame(
        [
            {
                "rule": rule,
                "precision": precision,
                "recall": recall,
                "support": support,
            }
            for rule, (precision, recall, support) in model.rules_
        ]
    )

    print(
        "[Training on full data] "
        f"accuracy={accuracy_score(y, y_pred):.3f}, "
        f"precision={precision_score(y, y_pred, zero_division=0):.3f}, "
        f"recall={recall_score(y, y_pred, zero_division=0):.3f}"
    )

    return y_pred, rules_df


def run_scenario(name, mvp_file, non_mvp_file, rules_file, predictions_file, precision_min, recall_min):
    print(f"\n===== Skope-Rules scenario: {name} =====")
    X, y, raw_df = load_and_preprocess_data(
        resolve_label_path(mvp_file),
        resolve_label_path(non_mvp_file),
    )
    print(f"Data loaded. Samples: {len(y)}, features: {len(FEATURE_COLS)}")

    preds_full, rules_full_df = train_predict_extract_rules_full(
        X, y, precision_min=precision_min, recall_min=recall_min
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_path = OUTPUT_DIR / rules_file
    predictions_path = OUTPUT_DIR / predictions_file

    rules_full_df.to_excel(rules_path, index=False)

    raw_df_full = raw_df.copy()
    raw_df_full["Predicted_is_MVP_Full"] = preds_full.astype(int)
    raw_df_full.to_excel(predictions_path, index=False)

    print(f"Rules saved to: {rules_path}")
    print(f"Predictions saved to: {predictions_path}")


def main():
    for scenario in SCENARIOS:
        run_scenario(*scenario)


if __name__ == "__main__":
    main()
