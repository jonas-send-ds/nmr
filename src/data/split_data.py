
import polars as pl

from src.util.constants import PATH_RAW_TRAIN_SET, PATH_RAW_VALIDATE_SET, PATH_RAW_META_MODEL, DATA_PATH


ERAS_TO_PURGE = 26  # half a year


def split_data() -> None:
    (DATA_PATH / 'folds').mkdir(parents=True, exist_ok=True)

    df_train: pl.DataFrame = pl.read_parquet(PATH_RAW_TRAIN_SET)
    df_validate: pl.DataFrame = pl.read_parquet(PATH_RAW_VALIDATE_SET)
    df_meta_model: pl.DataFrame = pl.read_parquet(PATH_RAW_META_MODEL)
    df_validate = df_validate.filter(pl.col("target").is_not_null())

    df_train = df_train.with_columns(pl.col("era").cast(pl.Int16))
    df_validate = df_validate.with_columns(pl.col("era").cast(pl.Int16))

    df_meta_model = df_meta_model.drop(['era', 'data_type'])
    df_meta_model = df_meta_model.join(df_validate, on='id', how='inner')
    df_meta_model.write_parquet(f"{DATA_PATH}/folds/df_meta_model.parquet")
    del df_meta_model

    df_all = pl.concat([df_train, df_validate])
    del df_train, df_validate

    df_all.write_parquet(f"{DATA_PATH}/df_all.parquet")

    number_of_observations = df_all.shape[0]
    start_eras = [df_all['era'][round(number_of_observations * x)] for x in [.55, .7, .85]] + [df_all['era'].max() + 1]

    for fold in range(3):
        (df_all.filter((pl.col('era') < (start_eras[fold] - ERAS_TO_PURGE)))
         .write_parquet(f'{DATA_PATH}/folds/df_train_{fold}.parquet'))
        (df_all.filter(((pl.col('era') >= start_eras[fold]) & (pl.col('era') < start_eras[fold + 1])))
         .write_parquet(f'{DATA_PATH}/folds/df_validate_{fold}.parquet'))

    print('Data saved in folds.')
