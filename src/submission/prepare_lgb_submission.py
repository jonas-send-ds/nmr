
# needs to be run with an environment conforming to src/submission/numerai_requirements.txt

import pandas as pd
import numpy as np
import lightgbm as lgb
import cloudpickle

from collections import Counter

from src.util.common import load_from_pickle
from src.util.constants import DATA_PATH


MULTIPLIER = -.75


selected_features = load_from_pickle(DATA_PATH / 'results/selected_features.pkl')
ensemble_models = [0, 0, 1]
model_dict = Counter(ensemble_models)

models = []
for index in model_dict.keys():
    models.append(lgb.Booster(model_file=f"{DATA_PATH}/models/lgb/lgb_model_{index}.txt"))


def normalise(prediction: np.array) -> np.array:
    return (prediction - prediction.min()) / (prediction.max() - prediction.min())


def get_linear_component(df: pd.DataFrame) -> np.array:
    X = df[selected_features].to_numpy()
    y = df['prediction'].to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    try:
        # this seems to be the fastest way to do this
        beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.lstsq(X, y)[0]

    return X @ beta_hat


# Define prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for index in model_dict.keys():
        prediction = normalise(models[index].predict(live_features[selected_features]))
        for i in range(model_dict[index]):
            predictions.append(prediction)

    live_prediction = np.mean(predictions, axis=0)
    live_features['prediction'] = pd.Series(live_prediction, index=live_features.index)
    submission = pd.Series(normalise(live_features['prediction'].to_numpy() + MULTIPLIER * get_linear_component(live_features)), index=live_features.index)

    return submission.to_frame("prediction")


(DATA_PATH / 'upload').mkdir(parents=True, exist_ok=True)
predict_pickle = cloudpickle.dumps(predict)
with open(DATA_PATH / "upload/upload_lgb.pkl", "wb") as f:
    f.write(predict_pickle)
