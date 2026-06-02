from pathlib import Path

import pandas as pd
from rulefit import RuleFit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
LABEL_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "outputs" / "rulefit"

FEATURE_COLS = [
    "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "AST%", "STL%", "BLK%",
    "TOV%", "USG%", "OWS", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P",
    "2PA", "2P%", "eFG%", "FT", "FTA", "TOV", "PF", "PTS", "PER", "TS%",
    "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "DWS", "WS", "WS/48", "OBPM",
    "DBPM", "BPM", "VORP", "win",
]

SCENARIOS = [
    ("historical_mvp", "MVP_ture.xlsx", "Non_MVP_ture.xlsx", "MVP_rules.xlsx"),
    ("data_driven_top3", "MVPtop3_rk.xlsx", "NonMVPtop3_rk.xlsx", "MVPtop3_rules.xlsx"),
    ("data_driven_mvp", "MVP_rk.xlsx", "NonMVP_rk.xlsx", "MVPrk_rules.xlsx"),
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

    if X.isnull().any().any():
        missing_rows = X.isnull().any(axis=1)
        print(f"Detected {missing_rows.sum()} rows with missing values; removing them.")
        X = X.loc[~missing_rows]
        y = y.loc[X.index]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def train_and_extract_rules(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tree_generator = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
    )

    model = RuleFit(
        tree_generator=tree_generator,
        max_rules=30,
        random_state=42,
    )

    model.fit(X_train.values, y_train, feature_names=FEATURE_COLS)
    rules = model.get_rules()
    rules = rules[(rules["coef"] != 0) & (rules["support"] > 0)]
    rules = rules.sort_values("importance", ascending=False).copy()

    if "condition" not in rules.columns:
        rules["condition"] = rules["rule"].astype(str) if "rule" in rules.columns else "(No rule text available)"

    rules["prediction_direction"] = rules["coef"].apply(lambda c: "MVP" if c > 0 else "Non-MVP")
    return model, rules, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test.values)
    y_pred = (y_pred_proba > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def run_scenario(name, mvp_file, non_mvp_file, output_file):
    print(f"\n===== RuleFit scenario: {name} =====")
    X, y = load_and_preprocess_data(
        resolve_label_path(mvp_file),
        resolve_label_path(non_mvp_file),
    )
    print(f"Data loaded. Samples: {len(y)}, features: {len(FEATURE_COLS)}")

    model, rules, X_test, y_test = train_and_extract_rules(X, y)
    metrics = evaluate_model(model, X_test, y_test)
    print(
        "Test metrics: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_file
    rules.to_excel(output_path, index=False)
    print(f"Rules saved to: {output_path}")


def main():
    for scenario in SCENARIOS:
        run_scenario(*scenario)


if __name__ == "__main__":
    main()
