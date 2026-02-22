
# needs to be run with an environment conforming to src/submission/numerai_requirements.txt

import polars as pl
import xgboost as xgb

from datetime import datetime

from src.util.constants import DATA_PATH, FIXED_XGB_PARAMETERS
from src.util.common import load_from_pickle

selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
required_columns = ['era', 'target'] + selected_features

df = pl.read_parquet(f"{DATA_PATH}/df_all.parquet")

# FIXME: add sorting in train_models_for_ensembling, then fix this logic
ensemble_models = [12, 10, 16, 14, 10]  # TODO: automate
ensemble_models = [i-9 for i in ensemble_models]
ensemble_models_unique = list(set(ensemble_models))

parameters_dict = load_from_pickle(DATA_PATH / 'results/parameters_dict.pkl')

(DATA_PATH / 'models/xgb').mkdir(parents=True, exist_ok=True)

for i in parameters_dict:
    parameters = parameters_dict[i]
    parameters["min_child_weight"] = (10 ** parameters["min_child_weight_exponent"] - 1) / 10 ** 5
    parameters["alpha"] = (10 ** parameters["alpha_exponent"] - 1) / 10 ** 5
    parameters["lambda"] = (10 ** parameters["lambda_exponent"] - 1) / 10 ** 5
    parameters.update(FIXED_XGB_PARAMETERS)

    # extract num_boost_round (not a param in xgb.train params dict)
    num_boost_round = parameters.pop('num_boost_round')

    for key in ["min_child_weight_exponent", "alpha_exponent", "lambda_exponent"]:
        del parameters[key]

    dtrain = xgb.DMatrix(
        df[selected_features].to_numpy(),
        label=df['target'].to_numpy()
    )

    model = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=num_boost_round
    )

    model.save_model(f"{DATA_PATH}/models/xgb/xgb_model_{i}.json")

    print(f"{datetime.now().strftime('%H:%M:%S')} . . . Model {i} saved.")
