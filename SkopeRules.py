import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import six, sys
sys.modules['sklearn.externals.six'] = six
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
from skrules import SkopeRules


# ============================== Data Loading & Preprocessing ==============================
def load_and_preprocess_data(mvp_path, non_mvp_path):
    mvp_df = pd.read_excel(mvp_path, engine="openpyxl")
    non_mvp_df = pd.read_excel(non_mvp_path, engine="openpyxl")

    mvp_df["is_MVP"] = 1
    non_mvp_df["is_MVP"] = 0
    df = pd.concat([mvp_df, non_mvp_df], ignore_index=True)

    feature_cols = [
        'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'AST%', 'STL%', 'BLK%', 
        'TOV%', 'USG%', 'OWS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', 
        '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'TOV', 'PF', 'PTS', 'PER', 'TS%', 
        '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'DWS', 'WS', 'WS/48', 'OBPM', 
        'DBPM', 'BPM', 'VORP', 'win'
    ]

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[feature_cols]
    y = df["is_MVP"]

    mask = ~X.isnull().any(axis=1)
    if not mask.all():
        print(f"⚠ Detected { (~mask).sum() } rows with missing values, removed.")

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    df_clean = df.loc[mask].reset_index(drop=True)

    return X, y, df_clean, feature_cols


# ============================== Training & Rule Extraction ==============================
def train_predict_extract_rules_full(X, y, feature_names, precision_min=0.5, recall_min=0.1):

    skope = SkopeRules(
        feature_names=feature_names,
        n_estimators=300,
        precision_min=precision_min,
        recall_min=recall_min,
        random_state=100
    )

    skope.fit(X, y)
    y_pred = skope.predict(X)

    rules_data = []
    for rule, (prec_rule, rec_rule, supp_rule) in skope.rules_:
        rules_data.append({
            "rule": rule,
            "precision": prec_rule,
            "recall": rec_rule,
            "support": supp_rule
        })
    rules_df = pd.DataFrame(rules_data)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    print(f"\n[Training on Full Data] Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

    return y_pred, rules_df



##### ===================== True MVP Rule Extraction =====================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVP_ture.xlsx"
    NON_MVP_DATA_PATH = "Non_MVP_ture.xlsx"

    # Data loading
    X, y, raw_df, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print(f"✅ Data loaded. Samples: {len(y)}, Features: {len(feature_cols)}")

    # Training on full data
    preds_full, rules_full_df = train_predict_extract_rules_full(
        X, y, feature_cols, precision_min=0.8, recall_min=0.5
    )
    rules_full_df.to_excel("full_data_rules.xlsx", index=False)

    raw_df_full = raw_df.copy()
    raw_df_full["Predicted_is_MVP_Full"] = preds_full.astype(int)
    mvp_predicted_full = raw_df_full[raw_df_full["Predicted_is_MVP_Full"] == 1]
    print(f"\n Number of players predicted as MVP (Full Data): {len(mvp_predicted_full)}")
    print(mvp_predicted_full.head())

    raw_df_full.to_excel("MVP_predictions_full.xlsx", index=False)



##### ===================== MVPlike Rule Extraction =====================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVPtop3_rk.xlsx"
    NON_MVP_DATA_PATH = "NonMVPtop3_rk.xlsx"

    X, y, raw_df, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print(f"✅ Data loaded. Samples: {len(y)}, Features: {len(feature_cols)}")

    preds_full, rules_full_df = train_predict_extract_rules_full(
        X, y, feature_cols, precision_min=0.8, recall_min=0.63
    )
    rules_full_df.to_excel("full_data_MVPlikerules.xlsx", index=False)

    raw_df_full = raw_df.copy()
    raw_df_full["Predicted_is_MVP_Full"] = preds_full.astype(int)
    mvp_predicted_full = raw_df_full[raw_df_full["Predicted_is_MVP_Full"] == 1]
    print(f"\n Number of players predicted as MVP-like (Full Data): {len(mvp_predicted_full)}")
    print(mvp_predicted_full.head())

    raw_df_full.to_excel("MVPlike_predictions_full.xlsx", index=False)



##### ===================== MVP Prediction (Pre-season) Rule Extraction =====================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVP_rk.xlsx"
    NON_MVP_DATA_PATH = "NonMVP_rk.xlsx"

    X, y, raw_df, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print(f"✅ Data loaded. Samples: {len(y)}, Features: {len(feature_cols)}")

    preds_full, rules_full_df = train_predict_extract_rules_full(
        X, y, feature_cols, precision_min=0.8, recall_min=0.52
    )
    rules_full_df.to_excel("full_data_MVPprerules.xlsx", index=False)

    raw_df_full = raw_df.copy()
    raw_df_full["Predicted_is_MVP_Full"] = preds_full.astype(int)
    mvp_predicted_full = raw_df_full[raw_df_full["Predicted_is_MVP_Full"] == 1]
    print(f"\n Number of players predicted as MVP (Pre-season): {len(mvp_predicted_full)}")
    print(mvp_predicted_full.head())

    raw_df_full.to_excel("MVPpre_predictions_full.xlsx", index=False)
