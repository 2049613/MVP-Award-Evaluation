import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ensemble_ranking import EnsembleRanking
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import EnsembleRanking dependencies. The UFER script requires the "
        "Clus helper module used by ensemble_ranking.py. Please include helpers.py "
        "and the corresponding Clus runtime files in code/ufer before running this script."
    ) from exc

try:
    from relief import URelief
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import URelief. Please include relief.py in code/ufer before running this script."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CONFIG = {
    "data_path": CODE_DIR / "data" / "rawdata.xlsx",
    "save_path": BASE_DIR / "outputs" / "TOP3.xlsx",
    "compute_genie3": True,
    "compute_forest": True,
    "compute_urelief": True,
    "year_column": "Season",
    "player_column": "name",
    "genie3_params": {"score": ["Genie3"], "ensemble": "ExtraTrees", "ensemble_size": 3},
    "forest_params": {"score": ["RForest"], "ensemble": "ExtraTrees", "ensemble_size": 3},
    "relief_params": {"neighbours": 5},
}


def max_normalize(scores):
    scores_array = np.array(list(scores.values()), dtype=float)

    if len(scores_array) <= 1:
        return {key: 1.0 for key in scores}

    max_val = np.max(scores_array)
    if max_val == 0:
        return {key: 0.0 for key in scores}

    normalized = scores_array / max_val
    return {key: normalized[i] for i, key in enumerate(scores)}


def load_and_preprocess_data(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(data_path, engine="openpyxl")
    elif data_path.suffix.lower() == ".csv":
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {data_path.suffix}")

    data = data.dropna()
    logger.info("Loaded data from %s: %s", data_path, data.shape)

    for col in [CONFIG["year_column"], CONFIG["player_column"]]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    return data


def process_special_dict(scores_dict, feature_count, algorithm_name):
    unique_key = next(iter(scores_dict.keys()))
    unique_value = scores_dict[unique_key]

    if isinstance(unique_value, (list, np.ndarray)) and len(unique_value) == feature_count:
        return {i: unique_value[i] for i in range(feature_count)}
    if isinstance(unique_value, (list, np.ndarray)):
        logger.warning("%s returned fewer scores than expected; padding with zeros.", algorithm_name)
        return {i: (unique_value[i] if i < len(unique_value) else 0.0) for i in range(feature_count)}

    try:
        numeric_value = float(unique_value)
        return {i: numeric_value for i in range(feature_count)}
    except ValueError:
        logger.error("%s returned a nonnumeric score; replacing with zeros.", algorithm_name)
        return {i: 0.0 for i in range(feature_count)}


def process_scores(algorithm, algorithm_name, X, feature_count):
    try:
        algorithm.fit(X)
        scores = algorithm.feature_importance_ if hasattr(algorithm, "feature_importance_") else algorithm.fit(X)
    except Exception as exc:
        raise RuntimeError(f"{algorithm_name} failed: {str(exc)}") from exc

    if isinstance(scores, (list, np.ndarray)):
        scores_dict = {i: scores[i] for i in range(len(scores))}
    elif isinstance(scores, dict):
        scores_dict = scores
    else:
        raise TypeError(f"{algorithm_name} returned unsupported score type: {type(scores)}")

    if len(scores_dict) == 1 and isinstance(next(iter(scores_dict.keys())), str):
        scores_dict = process_special_dict(scores_dict, feature_count, algorithm_name)

    for idx in set(range(feature_count)) - set(scores_dict.keys()):
        scores_dict[idx] = 0.0

    normalized_scores = max_normalize(scores_dict)
    logger.info("%s feature importance scores were max-normalized.", algorithm_name)
    return normalized_scores


def calculate_player_scores(original_data, current_features, scores_dict, algorithm_name):
    logger.info("%s - calculating player scores using %d features.", algorithm_name, len(current_features))
    current_feature_data = original_data[current_features]
    importance_scores = np.array([float(scores_dict[idx]) for idx in range(len(current_features))])
    return np.dot(current_feature_data.values, importance_scores)


def find_top_players_per_year(scores_df, score_column, year_col, player_col, current_features, top_n=3):
    top_players = scores_df.groupby(year_col, group_keys=False).apply(
        lambda group: group.sort_values(by=score_column, ascending=False).head(top_n)
    )

    top_players["rank"] = top_players.groupby(year_col)[score_column].rank(
        method="dense", ascending=False
    ).astype(int)
    top_players = top_players.sort_values(by=[year_col, "rank"])

    keep_cols = [year_col, "rank", player_col, score_column] + [
        col for col in scores_df.columns if col not in current_features + [score_column, "rank"]
    ]
    return top_players[list(dict.fromkeys(keep_cols))]


def save_feature_importance(scores_dict, features, algorithm_name, writer):
    importance_df = pd.DataFrame(
        {
            "feature": [features[i] for i in range(len(features))],
            "importance_score": [scores_dict[i] for i in range(len(features))],
        }
    )
    importance_df = importance_df.sort_values(by="importance_score", ascending=False)
    sheet_name = f"{algorithm_name}_feature_importance"
    importance_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    logger.info("%s - feature importance sheet saved.", algorithm_name)


def main():
    CONFIG["save_path"].parent.mkdir(parents=True, exist_ok=True)
    original_data = load_and_preprocess_data(CONFIG["data_path"])

    non_feature_cols = [CONFIG["year_column"], CONFIG["player_column"]]
    current_features = [col for col in original_data.columns if col not in non_feature_cols]
    current_X = original_data[current_features].values
    current_feature_count = len(current_features)
    scenario_df = original_data.copy()

    score_columns = []
    enabled_algorithms = []

    with pd.ExcelWriter(CONFIG["save_path"], engine="openpyxl") as writer:
        for algo_name, algo_class, algo_params, enabled_key in [
            ("Genie3", EnsembleRanking, CONFIG["genie3_params"], "compute_genie3"),
            ("Forest", EnsembleRanking, CONFIG["forest_params"], "compute_forest"),
            ("URelief", URelief, CONFIG["relief_params"], "compute_urelief"),
        ]:
            if not CONFIG[enabled_key]:
                continue

            logger.info("Processing algorithm: %s", algo_name)
            algorithm = algo_class(**algo_params)
            scores_dict = process_scores(algorithm, algo_name, current_X, current_feature_count)
            save_feature_importance(scores_dict, current_features, algo_name, writer)

            score_col = f"{algo_name}_score"
            scenario_df[score_col] = calculate_player_scores(
                original_data, current_features, scores_dict, algo_name
            )
            enabled_algorithms.append(algo_name)
            score_columns.append(score_col)

            top_players = find_top_players_per_year(
                scenario_df,
                score_col,
                CONFIG["year_column"],
                CONFIG["player_column"],
                current_features,
                top_n=3,
            )
            top_players.to_excel(writer, sheet_name=f"{algo_name}_top3"[:31], index=False)

        if not score_columns:
            raise RuntimeError("No ranking algorithms were enabled or successfully executed.")

        scenario_df["Final average score"] = scenario_df[score_columns].mean(axis=1)
        output_cols = [CONFIG["year_column"], CONFIG["player_column"]] + score_columns + ["Final average score"]
        scenario_df[output_cols].to_excel(writer, sheet_name="All player scores"[:31], index=False)

        final_top_players = find_top_players_per_year(
            scenario_df,
            "Final average score",
            CONFIG["year_column"],
            CONFIG["player_column"],
            current_features,
            top_n=3,
        )
        final_top_players.to_excel(writer, sheet_name="Final average top3"[:31], index=False)

    logger.info("Saved UFER output to: %s", CONFIG["save_path"])


if __name__ == "__main__":
    main()
