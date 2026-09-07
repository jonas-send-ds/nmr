"""Module for splitting the data into train and validation (and meta-model) sets."""

import polars as pl
from tqdm import tqdm

from src.util.constants import PATH_RAW_TRAIN_SET, PATH_RAW_VALIDATE_SET, PATH_RAW_META_MODEL, DATA_PATH
from src.util.common import get_constant_features, get_redundant_features, save_as_pickle

ERAS_TO_PURGE = 8  # eight weeks
# TODO: make this more precise:
# The last meta-model era when Numerai switched to the new 60-day target was 1222 - leave 4 weeks buffer
FIRST_APPROXIMATED_ERA_WITH_CURRENT_SPEC = 1226


def split_data() -> None:
    """
    Concatenate Numerai's train and validation sets and split them into 3 folds.

    Purges weeks in between the training and validation data.
    In addition, a dataset containing the meta-model predictions is created.
    """
    (DATA_PATH / "folds").mkdir(parents=True, exist_ok=True)

    df_train: pl.DataFrame = (pl.read_parquet(PATH_RAW_TRAIN_SET)
                              .with_columns(pl.col("era").cast(pl.Int16)))
    df_validate: pl.DataFrame = (pl.read_parquet(PATH_RAW_VALIDATE_SET)
                                 .with_columns(pl.col("era").cast(pl.Int16))
                                 .filter(pl.col("target").is_not_null()))

    (pl.read_parquet(PATH_RAW_META_MODEL).with_columns(pl.col("era").cast(pl.Int16))
     .filter(pl.col("era") > FIRST_APPROXIMATED_ERA_WITH_CURRENT_SPEC)
     .drop(["era", "data_type"])
     .join(df_validate, on="id", how="inner")
     .write_parquet(DATA_PATH / "folds/df_meta_model.parquet"))

    df_all = pl.concat([df_train, df_validate])
    del df_train, df_validate

    df_all.write_parquet(DATA_PATH / "raw/df_all.parquet")
    feature_names = [x for x in df_all.columns if "feature" in x]
    eras = df_all["era"].unique().to_list()
    save_as_pickle(feature_names, DATA_PATH / "raw/feature_names.pkl")
    save_as_pickle(eras, DATA_PATH / "raw/eras.pkl")

    number_of_observations = df_all.shape[0]
    start_eras = [df_all["era"][round(number_of_observations * x)] for x in [.55, .7, .85]] + [df_all["era"].max() + 1]

    constant_features = get_constant_features(df_all)
    print(f"Found {len(constant_features)} constant features!")

    redundant_features = get_redundant_features(df_all)
    print(f"Found {len(redundant_features)} redundant features!")

    for fold in tqdm(range(3), desc="Folds"):
        # We save the folds as dataframes for potential experiments and
        # the training data as matrices for quick and memory-efficient access
        # The prepared training data samples eras to avid memory issues
        df_train_fold = df_all.filter(pl.col("era") < (start_eras[fold] - ERAS_TO_PURGE))
        df_train_fold.write_parquet(DATA_PATH / f"folds/df_train_{fold}.parquet")

        df_train_fold = df_train_fold.drop(
            [col for col in df_train_fold.columns if "target_" in col] + ["data_type", "id"]
        ).filter(pl.col("era") % 2 == 0)  # sample every second era
        X_train = df_train_fold[feature_names].to_numpy()
        y_train = df_train_fold["target"].to_numpy()
        save_as_pickle(X_train, DATA_PATH / f"folds/X_sampled_train_{fold}.pkl")
        save_as_pickle(y_train, DATA_PATH / f"folds/y_sampled_train_{fold}.pkl")
        del df_train_fold

        (df_all.filter((pl.col("era") >= start_eras[fold]) & (pl.col("era") < start_eras[fold + 1]))
         .write_parquet(DATA_PATH / f"folds/df_validate_{fold}.parquet"))

    print("Data saved in folds.")
