import pandas as pd
import numpy as np
import logging
from pathlib import Path
from ensemble_ranking import EnsembleRanking
from relief import URelief

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CONFIG = {
    'data_path': Path("NBA player data.csv"),
    'save_path': Path("TOP3.xlsx"),
    'compute_genie3': True,
    'compute_forest': True,
    'compute_urelief': True,
    'year_column': 'Season',
    'player_column': 'name',
    'genie3_params': {'score': ["Genie3"], 'ensemble': "ExtraTrees", 'ensemble_size': 3},
    'forest_params': {'score': ["RForest"], 'ensemble': "ExtraTrees", 'ensemble_size': 3},
    'relief_params': {'neighbours': 5}
}


def max_normalize(scores):
    "MIN-MAX"
    scores_array = np.array(list(scores.values()))

    if len(scores_array) <= 1:
        return {k: 1.0 for k in scores.keys()}

    max_val = np.max(scores_array)

    if max_val == 0:
        return {k: 0.0 for k in scores.keys()}

    normalized = scores_array / max_val

    return {k: normalized[i] for i, k in enumerate(scores.keys())}


def load_and_preprocess_data(data_path):

    try:
        data = pd.read_csv(data_path)
        data = data.dropna()
        logger.info(f": {data.shape}")


        for col in [CONFIG['year_column'], CONFIG['player_column']]:
            if col not in data.columns:
                raise ValueError(f"no data: {col}，")

        return data
    except Exception as e:
        logger.error(f": {str(e)}", exc_info=True)
        raise


def process_special_dict(scores_dict, feature_count, algorithm_name):
    unique_key = next(iter(scores_dict.keys()))
    unique_value = scores_dict[unique_key]

    if isinstance(unique_value, (list, np.ndarray)) and len(unique_value) == feature_count:
        return {i: unique_value[i] for i in range(feature_count)}
    elif isinstance(unique_value, (list, np.ndarray)):
        logger.warning(f"{algorithm_name}")
        return {i: (unique_value[i] if i < len(unique_value) else 0.0) for i in range(feature_count)}
    else:
        try:
            numeric_value = float(unique_value)
            return {i: numeric_value for i in range(feature_count)}
        except:
            logger.error(f"{algorithm_name}")
            return {i: 0.0 for i in range(feature_count)}

#feature important
def process_scores(algorithm, algorithm_name, X, feature_count):
    """Extract, process and normalize the feature importance scores of the algorithm."""
    try:
        algorithm.fit(X)
        scores = algorithm.feature_importance_ if hasattr(algorithm, 'feature_importance_') else algorithm.fit(X)
    except Exception as e:
        raise RuntimeError(f"{algorithm_name}: {str(e)}")

    if isinstance(scores, (list, np.ndarray)):
        scores_dict = {i: scores[i] for i in range(len(scores))}
    elif isinstance(scores, dict):
        scores_dict = scores
    else:
        raise TypeError(f"{algorithm_name}: {type(scores)}")

    if len(scores_dict) == 1 and isinstance(next(iter(scores_dict.keys())), str):
        scores_dict = process_special_dict(scores_dict, feature_count, algorithm_name)


    all_indices = set(range(feature_count))
    for idx in all_indices - set(scores_dict.keys()):
        scores_dict[idx] = 0.0

    normalized_scores = max_normalize(scores_dict)
    logger.info(f"{algorithm_name}The feature importance has been normalized by dividing by the maximum value.")

    return normalized_scores


def calculate_player_scores(original_data, current_features, scores_dict, algorithm_name):
    """Calculate the comprehensive score of the players."""
    logger.info(f"{algorithm_name} - Calculate the comprehensive score of the players.（feature: {len(current_features)}）")

    current_feature_data = original_data[current_features]


    importance_scores = np.array([
        float(scores_dict[idx][0]) if isinstance(scores_dict[idx], (list, np.ndarray))
        else float(scores_dict[idx])
        for idx in range(len(current_features))
    ])

    # compute scores
    return np.dot(current_feature_data.values, importance_scores)


def find_top_players_per_year(scores_df, score_column, year_col, player_col, current_features, top_n=3):
    """find top3"""


    def get_top_n(group):
        return group.sort_values(by=score_column, ascending=False).head(top_n)

    top_players = scores_df.groupby(year_col, group_keys=False).apply(get_top_n)

    top_players['rank'] = top_players.groupby(year_col)[score_column].rank(
        method='dense', ascending=False
    ).astype(int)


    top_players = top_players.sort_values(by=[year_col, 'rank'])


    keep_cols = [year_col, 'rank', player_col, score_column] + [
        col for col in scores_df.columns if col not in current_features + [score_column, 'rank']
    ]
    return top_players[list(dict.fromkeys(keep_cols))]


def save_feature_importance(scores_dict, features, algorithm_name, writer):
    "Save the feature importance scores"

    importance_df = pd.DataFrame({
        'feature': [features[i] for i in range(len(features))],
        'important score': [scores_dict[i] for i in range(len(features))]
    })


    importance_df = importance_df.sort_values(by='score', ascending=False)

    sheet_name = f"{algorithm_name}_feature important"
    importance_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    logger.info(f"{algorithm_name} - save sheet: {sheet_name[:31]}")


def main():
    try:

        original_data = load_and_preprocess_data(CONFIG['data_path'])


        non_feature_cols = [CONFIG['year_column'], CONFIG['player_column']]
        all_features = [col for col in original_data.columns if col not in non_feature_cols]
        logger.info(f"feature number: {len(all_features)}，feature list: {all_features}")


        current_features = all_features
        scenario_name = "Original feature set"

        logger.info(f"\n===== begin: {scenario_name} =====")


        current_X = original_data[current_features].values
        current_feature_count = len(current_features)
        logger.info(f"Current feature count: {current_feature_count}")


        scenario_df = original_data.copy()


        enabled_algorithms = []

        score_columns = []


        with pd.ExcelWriter(CONFIG['save_path'], engine="openpyxl") as writer:

            for algo_name, algo_class, algo_params in [
                ('Genie3', EnsembleRanking, CONFIG['genie3_params']),
                ('Forest', EnsembleRanking, CONFIG['forest_params']),
                ('URelief', URelief, CONFIG['relief_params'])
            ]:
                if not CONFIG[f'compute_{algo_name.lower()}']:
                    continue

                try:
                    logger.info(f"\n----- processing algorithm: {algo_name} -----")
                    # Initialize the algorithm and calculate the importance of features
                    algorithm = algo_class(**algo_params)
                    scores_dict = process_scores(algorithm, algo_name, current_X, current_feature_count)

                    #Save the results of feature importance
                    save_feature_importance(scores_dict, current_features, algo_name, writer)

                    # Calculate the comprehensive score of the players and add it to the DataFrame
                    score_col = f'{algo_name}_score'
                    player_scores = calculate_player_scores(original_data, current_features, scores_dict, algo_name)
                    scenario_df[score_col] = player_scores


                    enabled_algorithms.append(algo_name)
                    score_columns.append(score_col)

                    # find TOP3
                    top_players = find_top_players_per_year(
                        scenario_df, score_col, CONFIG['year_column'],
                        CONFIG['player_column'], current_features, top_n=3
                    )

                    #
                    sheet_name = f"{algo_name}_top3"
                    top_players.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    logger.info(f"{algo_name} - top sheet: {sheet_name[:31]}")

                except Exception as e:
                    logger.error(f"{algo_name}: {str(e)}", exc_info=True)
                    continue

            # Calculate the average score of all enabled algorithms and use it as the final score.
            if len(score_columns) > 0:
                logger.info(f"\n----- calculate{len(score_columns)}the average score -----")


                scenario_df['Final average score'] = scenario_df[score_columns].mean(axis=1)


                all_scores_sheet = "Summary of all player scores"
                output_cols = [CONFIG['year_column'], CONFIG['player_column']] + score_columns + ['Final average score']
                scenario_df[output_cols].to_excel(writer, sheet_name=all_scores_sheet[:31], index=False)

                final_top_players = find_top_players_per_year(
                    scenario_df, 'Final average score', CONFIG['year_column'],
                    CONFIG['player_column'], current_features, top_n=3
                )

                
                final_sheet = "Final average score_top3"
                final_top_players.to_excel(writer, sheet_name=final_sheet[:31], index=False)
                logger.info(f"The top three players based on the final average score have been saved to the sheet.: {final_sheet[:31]}")
            else:
                logger.warning("wrong")

        logger.info(f"\nsave to: {CONFIG['save_path']}")

    except Exception as e:
        logger.error(f"Main process error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
