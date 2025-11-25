from sklearn.ensemble import RandomForestRegressor  # Use regressor
from rulefit import RuleFit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import os


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

    # Drop rows with missing values
    if X.isnull().any().any():
        print(f"⚠ Detected {X.isnull().sum().sum()} missing values. Corresponding rows removed.")
        X = X.dropna()
        y = y[X.index]

    return X, y, feature_cols


# ============================== Model Training & Rule Extraction ==============================
def train_and_extract_rules(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train using a regressor
    tree_generator = RandomForestRegressor(
        n_estimators=300,   # More trees for stability
        max_depth=4,        # Slightly deeper trees for more refined rules
        min_samples_leaf=3  # Avoid overfitting
    )

    rf = RuleFit(
        tree_generator=tree_generator,
        max_rules=30,
        random_state=42
    )

    rf.fit(X_train.values, y_train, feature_names=feature_cols)
    rules = rf.get_rules()
    filtered_rules = rules[(rules["coef"] != 0) & (rules["support"] > 0)]
    sorted_rules = filtered_rules.sort_values("importance", ascending=False).copy()

    # === ① Generate readable rule content (condition) ===
    if "condition" not in sorted_rules.columns:
        if "rule" in sorted_rules.columns:
            sorted_rules["condition"] = sorted_rules["rule"].astype(str)
        else:
            sorted_rules["condition"] = "(No rule text available)"

    # === ② Human-readable prediction direction ===
    sorted_rules["prediction_direction"] = sorted_rules["coef"].apply(
        lambda c: "MVP" if c > 0 else "Non-MVP"
    )

    return rf, sorted_rules, X_test, y_test


# ============================== Model Evaluation ==============================
def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test.values)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)  # Precision for MVP class
    recall = recall_score(y_test, y_pred)        # Recall for MVP class
    
    return acc, precision, recall



##### True MVP Rule Extraction
# ============================== Main Script ==============================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVP_ture.xlsx"
    NON_MVP_DATA_PATH = "Non_MVP_ture.xlsx"
    RULES_OUTPUT_PATH = "MVP_rules.xlsx"

    X, y, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print("✅ Data loaded. Number of features:", len(feature_cols))

    rf_model, rules, X_test, y_test = train_and_extract_rules(X, y, feature_cols)
    print("✅ Rule extraction completed")

    acc, precision, recall = evaluate_model(rf_model, X_test, y_test)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    rules.to_excel(RULES_OUTPUT_PATH, index=False)
    print(f"Rules saved to: {RULES_OUTPUT_PATH}")

    print("Top 5 important rules:")
    print(rules.head(5)[["importance", "support", "prediction_direction", "condition"]])



##### MVPlike Rule Extraction
# ============================== Main Script ==============================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVPtop3_rk.xlsx"
    NON_MVP_DATA_PATH = "RNonMVPtop3_rk.xlsx"
    RULES_OUTPUT_PATH = "MVPtop3_rules.xlsx"

    X, y, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print("Data loaded. Number of features:", len(feature_cols))

    rf_model, rules, X_test, y_test = train_and_extract_rules(X, y, feature_cols)
    print("Rule extraction completed")

    acc, precision, recall = evaluate_model(rf_model, X_test, y_test)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    rules.to_excel(RULES_OUTPUT_PATH, index=False)
    print(f"Rules saved to: {RULES_OUTPUT_PATH}")

    print("Top 5 important rules:")
    print(rules.head(5)[["importance", "support", "prediction_direction", "condition"]])



##### MVP Prediction Rule Extraction
# ============================== Main Script ==============================
if __name__ == "__main__":
    os.chdir("C:/Users/86133/Desktop/software/MVP_Select")

    MVP_DATA_PATH = "MVP_rk.xlsx"
    NON_MVP_DATA_PATH = "NonMVP_rk.xlsx"
    RULES_OUTPUT_PATH = "MVPrk_rules.xlsx"

    X, y, feature_cols = load_and_preprocess_data(MVP_DATA_PATH, NON_MVP_DATA_PATH)
    print("✅ Data loaded. Number of features:", len(feature_cols))

    rf_model, rules, X_test, y_test = train_and_extract_rules(X, y, feature_cols)
    print("✅ Rule extraction completed")

    acc, precision, recall = evaluate_model(rf_model, X_test, y_test)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    rules.to_excel(RULES_OUTPUT_PATH, index=False)
    print(f"Rules saved to: {RULES_OUTPUT_PATH}")

    print("Top 5 important rules:")
    print(rules.head(5)[["importance", "support", "prediction_direction", "condition"]])
