
import os

from numerapi import NumerAPI
from src.util.constants import DATA_PATH, PATH_RAW_TRAIN_SET, PATH_RAW_VALIDATE_SET, PATH_RAW_META_MODEL

VERSION = "v5.2"


def download_data() -> None:
    (DATA_PATH / 'raw').mkdir(parents=True, exist_ok=True)

    numer_api = NumerAPI()
    if os.path.exists(PATH_RAW_TRAIN_SET):
        os.remove(PATH_RAW_TRAIN_SET)
    if os.path.exists(PATH_RAW_VALIDATE_SET):
        os.remove(PATH_RAW_VALIDATE_SET)
    if os.path.exists(PATH_RAW_META_MODEL):
        os.remove(PATH_RAW_META_MODEL)

    numer_api.download_dataset(f"{VERSION}/train.parquet", PATH_RAW_TRAIN_SET)
    numer_api.download_dataset(f"{VERSION}/validation.parquet", PATH_RAW_VALIDATE_SET)
    numer_api.download_dataset(f"{VERSION}/meta_model.parquet", PATH_RAW_META_MODEL)

    print('Raw data downloaded.')
