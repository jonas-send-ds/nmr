
# needs to be run with an environment conforming to src/submission/numerai_requirements.txt

import polars as pl
import lightgbm as lgb

from datetime import datetime

from src.util.constants import DATA_PATH, FIXED_LGB_PARAMETERS
from src.util.common import load_from_pickle

selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
required_columns = ['era', 'target'] + selected_features

df = pl.read_parquet(f"{DATA_PATH}/df_all.parquet")

ensemble_models = [0, 0, 1]
ensemble_models_unique = list(set(ensemble_models))

parameters_list = load_from_pickle(DATA_PATH / 'results/parameters_list.pkl')

(DATA_PATH / 'models/lgb').mkdir(parents=True, exist_ok=True)

for i in range(len(parameters_list)):
    parameters = parameters_list[i]
    parameters["min_sum_hessian_in_leaf"] = (10 ** parameters["min_sum_hessian_in_leaf_exponent"] - 1) / 10 ** 5,
    parameters["lambda_l1"] = (10 ** parameters["lambda_l1_exponent"] - 1) / 10 ** 5,
    parameters["lambda_l2"] = (10 ** parameters["lambda_l2_exponent"] - 1) / 10 ** 5
    parameters.update(FIXED_LGB_PARAMETERS)
    for key in ["min_sum_hessian_in_leaf_exponent", "lambda_l1_exponent", "lambda_l2_exponent"]:
            del parameters[key]

    lgb_train = lgb.Dataset(
        df[selected_features].to_numpy(),
        label=df['target'].to_numpy()
    )

    model = lgb.train(
        params=parameters,
        train_set=lgb_train,
        num_boost_round=parameters['num_boost_round']
    )

    index = ensemble_models_unique[i]
    model.save_model(f"{DATA_PATH}/models/lgb/lgb_model_{index}.txt")

    print(f"{datetime.now().strftime('%H:%M:%S')} . . . Model {index} saved.")
