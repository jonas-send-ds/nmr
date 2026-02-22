
import cloudpickle
import numpy as np
import pandas as pd
import xgboost as xgb

from collections import Counter

from src.util.common import load_from_pickle
from src.util.constants import DATA_PATH

# TODO: this needs testing!

MULTIPLIER = -.8

selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
# FIXME: add sorting in train_models_for_ensembling, then fix this logic
ensemble_models = [12, 10, 16, 14, 10]  # TODO: automate
ensemble_models = [i-9 for i in ensemble_models]
model_dict = Counter(ensemble_models)

models = dict()
for index in model_dict.keys():
    model = xgb.Booster()
    model.load_model(f"{DATA_PATH}/models/xgb/xgb_model_{index}.json")
    models[index] = model


def normalise(prediction: np.array) -> np.array:
    return (prediction - prediction.min()) / (prediction.max() - prediction.min())


def get_linear_component(df: pd.DataFrame) -> np.array:
    X = df[selected_features].to_numpy()
    y = df['prediction'].to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    beta_hat = np.linalg.lstsq(X, y)[0]

    return X @ beta_hat


# Define prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    dmatrix_live = xgb.DMatrix(live_features[selected_features].to_numpy())
    predictions = []
    for index in model_dict.keys():
        prediction = normalise(models[index].predict(dmatrix_live))
        for i in range(model_dict[index]):
            predictions.append(prediction)

    live_prediction = np.mean(predictions, axis=0)
    live_features['prediction'] = pd.Series(live_prediction, index=live_features.index)
    submission = pd.Series(normalise(live_features['prediction'].to_numpy() + MULTIPLIER * get_linear_component(live_features)), index=live_features.index)

    return submission.to_frame("prediction")


(DATA_PATH / 'upload').mkdir(parents=True, exist_ok=True)
predict_pickle = cloudpickle.dumps(predict)
with open(DATA_PATH / "upload/upload_xgb.pkl", "wb") as f:
    f.write(predict_pickle)
