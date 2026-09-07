
import polars as pl
import numpy as np
import pickle
import matplotlib as plt

from pathlib import Path
from typing import Any

from scipy.stats import norm
from tqdm import tqdm

PRECISION = 1e-10  # very low tolerance to exclude features that are most likely unhelpful as well


def customise_plot_style() -> None:
    """
    Customise Matplotlib and Seaborn plots
    """
    plt.rc('axes.spines', right=False, top=False)  # hide top and right spine
    plt.rc('axes', grid=True, linewidth=.5)  # activate grid and slim axis lines
    plt.rc('xtick.major', width=.5)  # slim x-ticks
    plt.rc('ytick.major', width=.5)  # slim y-ticks
    plt.rc('grid', alpha=.8, linewidth=.5)  # slim, transparent grid


def save_as_pickle(x: Any, path: str | Path) -> None:
    """
    Save a given object to a specified file path in pickle format.

    :param x: The object to be saved
    :param path: The path to the pickle file
    """
    with open(path, "wb") as open_file:
        # noinspection PyTypeChecker
        pickle.dump(x, open_file)


def load_from_pickle(path: str | Path) -> Any:
    """
    Load an object from a pickle file.

    The file must have been created using Python's `pickle` module,
    and its format  should be compatible with the current Python version and environment.

    :param path: The path to the pickle file
    """
    with open(path, "rb") as open_file:
        return pickle.load(open_file)


def mean_grouped_spearman_correlation(
        prediction: pl.Series | np.ndarray,
        target: pl.Series | np.ndarray,
        era: pl.Series | np.ndarray
) -> float:
    """
    Calculate the mean of Spearman correlation coefficients applied grouped by era.

    :param prediction: prediction
    :param target: target
    :param era: era
    """
    return grouped_spearman_correlation(prediction, target, era).mean()


def grouped_spearman_correlation(
        prediction: pl.Series | np.ndarray,
        target: pl.Series | np.ndarray,
        era: pl.Series | np.ndarray
) -> pl.Series:
    """
    Calculate Spearman correlation for prediction and target within each era.

    :param prediction: prediction
    :param target: target
    :param era: era
    """
    _df: pl.DataFrame = pl.DataFrame({"era": era, "prediction": prediction, "target": target})

    calculate_numerai_corr = (pl.struct(pl.col("prediction"), pl.col("target"))
                              .map_batches(numerai_corr_struct, returns_scalar=True, return_dtype=pl.Float64))

    return (_df.group_by("era", maintain_order=True)
            .agg(calculate_numerai_corr.alias("correlation"))
            .select("correlation")
            .to_series())


def numerai_corr_struct(struct: pl.Series) -> np.float64:
    """
    Apply the numerai_corr function to a Polars struct column.

    :param struct: a Polars struct column containing the "prediction" and "prediction_mm" fields
    """
    prediction: pl.Series = struct.struct.field("prediction")
    target: pl.Series = struct.struct.field("target")

    return numerai_corr(prediction, target)



def numerai_corr(prediction: pl.Series, target: pl.Series) -> np.float64:
    """
    Correlation function that puts more weight on the tails provided by Numerai.
    See https://docs.numer.ai/numerai-tournament/scoring/correlation-corr.
    """
    ranked_prediction = (prediction.rank(method="average") - 0.5) / prediction.count()
    gauss_ranked_prediction = norm.ppf(ranked_prediction)  # gaussianise predictions
    centered_target = target - target.mean()  # make targets centered around 0


    prediction_heavy_tails = np.sign(gauss_ranked_prediction) * np.abs(gauss_ranked_prediction) ** 1.5
    target_heavy_tails = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    # noinspection PyTypeChecker
    return np.corrcoef(prediction_heavy_tails, target_heavy_tails)[0, 1]


def orthogonalise_by_era(prediction: pl.Series, prediction_mm: pl.Series, era: pl.Series) -> pl.Series:
    """
    Orthogonalises predictions with respect to the meta-model predictions by era.
    """
    df: pl.DataFrame = pl.DataFrame({"era": era, "prediction": prediction, "prediction_mm": prediction_mm})

    orthogonalise_lazy = (pl.struct(pl.col("prediction"), pl.col("prediction_mm"))
                          .map_batches(orthogonalise, return_dtype=pl.List(pl.Float64)))

    return (df.group_by("era", maintain_order=True)
            .agg(orthogonalise_lazy.alias("prediction_orthogonalised")).explode("prediction_orthogonalised")
            .select("prediction_orthogonalised")
            .to_series())


def orthogonalise(struct: pl.Series) -> np.ndarray:
    """
    Orthogonalises predictions with respect to the meta-model predictions by projecting
    prediction onto mm_prediction, then subtracting that projection from prediction.

    Arguments:
        struct: a Polars struct column containing the "prediction" and "prediction_mm" fields

    Returns:
        np.ndarray - the orthogonalised series as ndarray/list
    """
    prediction: pl.Series = struct.struct.field("prediction")
    prediction_mm: pl.Series = struct.struct.field("prediction_mm")

    ranked_prediction = (prediction.rank(method="average") - 0.5) / prediction.count()
    gauss_ranked_prediction = norm.ppf(ranked_prediction)  # gaussianise predictions
    ranked_prediction_mm = (prediction_mm.rank(method="average") - 0.5) / prediction_mm.count()
    gauss_ranked_prediction_mm = norm.ppf(ranked_prediction_mm)  # gaussianise meta-model predictions

    return gauss_ranked_prediction - np.outer(gauss_ranked_prediction_mm,
                                              (gauss_ranked_prediction.T @ gauss_ranked_prediction_mm) /
                                              (gauss_ranked_prediction_mm.T @ gauss_ranked_prediction_mm)).flatten()


def get_constant_features(df: pl.DataFrame, features_to_ignore: list[str] | None=None) -> list[str]:
    """
    Get features without variation in the dataframe.

    Checks only columns that contain the substring 'feature'.

    :param df: The dataframe to check for constant features
    :param features_to_ignore: a list of features to ignore
    :return: A list of constant features
    """
    if features_to_ignore is None:
        features_to_ignore = []
    feature_names = [x for x in df.columns if "feature" in x]
    feature_names = [feature for feature in feature_names if feature not in features_to_ignore]

    constant_features = []
    for feature in tqdm(feature_names, desc="Features"):
        if df[feature].var() < PRECISION:
            constant_features.append(feature)

    return constant_features


def get_redundant_features(df: pl.DataFrame, features_to_ignore: list[str] | None=None) -> list[str]:
    """
    Get features which are duplicated in a dataframe.

    Checks only columns that contain the substring 'feature'.

    :param df: The dataframe to check for redundant features
    :param features_to_ignore: a list of features to ignore
    :return: A list of redundant features
    """
    if features_to_ignore is None:
        features_to_ignore = []
    feature_names = [x for x in df.columns if "feature" in x]
    feature_names = [feature for feature in feature_names if feature not in features_to_ignore]
    df_sample = df.sample(n=1000).select(feature_names)

    candidate_pairs = []
    for i in tqdm(range(len(feature_names)), desc="Features"):
        for j in range(i + 1, len(feature_names)):
            corr = df_sample.select(pl.corr(feature_names[i], feature_names[j])).to_series().item()
            if corr > (1 - PRECISION):
                candidate_pairs.append((feature_names[i], feature_names[j]))

    redundant_features = set()
    if candidate_pairs:
        print(f"Found {len(candidate_pairs)} candidate pairs for duplicated features...")
        for feature1, feature2 in tqdm(candidate_pairs, desc="Candidate pairs"):
            corr_full = df.select(pl.corr(feature1, feature2)).to_series().item()
            if corr_full > (1 - PRECISION):
                redundant_features.add(feature2)
    return list(redundant_features)
