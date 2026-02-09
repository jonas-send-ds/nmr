
from datetime import datetime

import polars as pl
import lightgbm as lgb
import xgboost as xgb

from src.util.constants import DATA_PATH, FIXED_LGB_PARAMETERS, FIXED_XGB_PARAMETERS
from src.util.common import load_from_pickle

NUMBER_OF_MODELS = 20

study_lgb = load_from_pickle(DATA_PATH / 'results/study_lgb.pkl')
study_xgb = load_from_pickle(DATA_PATH / 'results/study_xgb.pkl')
selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
required_columns = ['era', 'target'] + selected_features

df_train_list = []
for fold in range(3):
    df_train_fold = pl.read_parquet(f"{DATA_PATH}/folds/df_train_{fold}.parquet")
    df_train_fold = df_train_fold.select(required_columns)
    df_train_list.append(df_train_fold)
    del df_train_fold

trial_values = []
for trial in study_lgb.trials:
    if trial.value is not None and trial.state == 1:
        trial_values.append(trial.value)
for trial in study_xgb.trials:
    if trial.value is not None and trial.state == 1:
        trial_values.append(trial.value)

cutoff = sorted(trial_values)[-NUMBER_OF_MODELS]

parameters_list_lgb = [trial.params for trial in study_lgb.trials if trial.state == 1 and trial.value >= cutoff]
parameters_list_xgb = [trial.params for trial in study_xgb.trials if trial.state == 1 and trial.value >= cutoff]

(DATA_PATH / 'models/lgb').mkdir(parents=True, exist_ok=True)
(DATA_PATH / 'models/xgb').mkdir(parents=True, exist_ok=True)

for index in range(len(parameters_list_lgb)):
    parameters = parameters_list_lgb[index]
    parameters["min_sum_hessian_in_leaf"] = (10 ** parameters["min_sum_hessian_in_leaf_exponent"] - 1) / 10 ** 5
    parameters["lambda_l1"] = (10 ** parameters["lambda_l1_exponent"] - 1) / 10 ** 5
    parameters["lambda_l2"] = (10 ** parameters["lambda_l2_exponent"] - 1) / 10 ** 5
    parameters.update(FIXED_LGB_PARAMETERS)
    for key in ["min_sum_hessian_in_leaf_exponent", "lambda_l1_exponent", "lambda_l2_exponent"]:
            del parameters[key]

    for fold in range(3):
        lgb_train = lgb.Dataset(
            df_train_list[fold][selected_features].to_numpy(),
            label=df_train_list[fold]['target'].to_numpy()
        )

        model = lgb.train(
            params=parameters,
            train_set=lgb_train,
            num_boost_round=parameters['num_boost_round']
        )

        model.save_model(f"{DATA_PATH}/models/lgb/lgb_model_{index}_{fold}.txt")

        print(f"{datetime.now().strftime('%H:%M:%S')} . . . LightGBM model {index} saved for fold {fold}.")


for index in range(len(parameters_list_xgb)):
    parameters = parameters_list_xgb[index]
    parameters["min_child_weight"] = (10 ** parameters["min_child_weight_exponent"] - 1) / 10 ** 5
    parameters["alpha"] = (10 ** parameters["alpha_exponent"] - 1) / 10 ** 5
    parameters["lambda"] = (10 ** parameters["lambda_exponent"] - 1) / 10 ** 5
    parameters.update(FIXED_XGB_PARAMETERS)

    # extract num_boost_round (not a param in xgb.train params dict)
    num_boost_round = parameters.pop('num_boost_round')

    for key in ["min_child_weight_exponent", "alpha_exponent", "lambda_exponent"]:
        del parameters[key]

    for fold in range(3):
        dtrain = xgb.DMatrix(
            df_train_list[fold][selected_features].to_numpy(),
            label=df_train_list[fold]['target'].to_numpy()
        )

        model = xgb.train(
            params=parameters,
            dtrain=dtrain,
            num_boost_round=num_boost_round
        )

        model.save_model(f"{DATA_PATH}/models/xgb/xgb_model_{index}_{fold}.json")

        print(f"{datetime.now().strftime('%H:%M:%S')} . . . XGBoost model {index} saved for fold {fold}.")
