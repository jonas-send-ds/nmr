
import polars as pl
import lightgbm as lgb

from datetime import datetime

from src.util.constants import DATA_PATH, FIXED_LGB_PARAMETERS
from src.util.common import load_from_pickle


# Train the top 20 models from hyperparameter tuning (see src/notebooks/tune_hyperparameters.ipynb) for the 3 folds.


study = load_from_pickle(DATA_PATH / 'results/study.pkl')
selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
required_columns = ['era', 'target'] + selected_features

df_train_list = []
for fold in range(3):
    df_train_fold = pl.read_parquet(f"{DATA_PATH}/folds/df_train_{fold}.parquet")
    df_train_fold = df_train_fold.select(required_columns)
    df_train_list.append(df_train_fold)
    del df_train_fold

sorted_trials = sorted(study.trials, key=lambda trial: trial.value if trial.value is not None else float('-inf'), reverse=True)[:20]
parameters_list = [trial.params for trial in sorted_trials]

(DATA_PATH / 'models/lgb').mkdir(parents=True, exist_ok=True)

for index in range(len(parameters_list)):
    parameters = parameters_list[index]
    parameters["min_sum_hessian_in_leaf"] = (10 ** parameters["min_sum_hessian_in_leaf_exponent"] - 1) / 10 ** 5,
    parameters["lambda_l1"] = (10 ** parameters["lambda_l1_exponent"] - 1) / 10 ** 5,
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

        print(f"{datetime.now().strftime('%H:%M:%S')} . . . Model {index} saved for fold {fold}.")
