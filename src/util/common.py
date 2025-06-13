
import polars as pl
import numpy as np
import pickle
import matplotlib as plt

from pathlib import Path
from typing import Any
from scipy.stats import norm


def customise_plot_style() -> None:
    """
    customise Matplotlib and Seaborn plots
    """
    plt.rc('axes.spines', right=False, top=False)  # hide top and right spine
    plt.rc('axes', grid=True, linewidth=.5)  # activate grid and slim axis lines
    plt.rc('xtick.major', width=.5)  # slim x-ticks
    plt.rc('ytick.major', width=.5)  # slim y-ticks
    plt.rc('grid', alpha=.8, linewidth=.5)  # slim, transparent grid


def save_as_pickle(x: Any, path: str | Path) -> None:
    with open(path, "wb") as open_file:
        # noinspection PyTypeChecker
        pickle.dump(x, open_file)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as open_file:
        return pickle.load(open_file)


def mean_grouped_spearman_correlation(prediction: pl.Series, target: pl.Series, era: pl.Series) -> float:
    return grouped_spearman_correlation(prediction, target, era).mean()


def grouped_spearman_correlation(prediction: pl.Series, target: pl.Series, era: pl.Series) -> pl.Series:
    _df: pl.DataFrame = pl.DataFrame({"era": era, "prediction": prediction, "target": target})

    calculate_numerai_corr = (pl.struct(pl.col('prediction'), pl.col('target'))
                              .map_elements(numerai_corr_struct, return_dtype=pl.Float64))

    return (_df.group_by("era", maintain_order=True)
            .agg(calculate_numerai_corr.alias('correlation'))
            .select('correlation')
            .to_series())


def numerai_corr_struct(_df: pl.Series) -> np.float64:
    """
    :param _df: a Polars struct column
    :return: the correlation coefficient between the predictions and the target as float
    """
    prediction: pl.Series = _df.struct.field("prediction")
    target: pl.Series = _df.struct.field("target")

    return numerai_corr(prediction, target)


# correlation function that puts more weight on the tails
# provided by numerai (see https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)
def numerai_corr(prediction: pl.Series, target: pl.Series) -> np.float64:
    ranked_prediction = (prediction.rank(method="average") - 0.5) / prediction.count()
    gauss_ranked_prediction = norm.ppf(ranked_prediction)  # gaussianise predictions
    centered_target = target - target.mean()  # make targets centered around 0


    prediction_heavy_tails = np.sign(gauss_ranked_prediction) * np.abs(gauss_ranked_prediction) ** 1.5
    target_heavy_tails = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    # noinspection PyTypeChecker
    return np.corrcoef(prediction_heavy_tails, target_heavy_tails)[0, 1]


def orthogonalise_by_era(prediction: pl.Series, prediction_mm: pl.Series, era: pl.Series) -> pl.Series:
    df: pl.DataFrame = pl.DataFrame({"era": era, "prediction": prediction, "prediction_mm": prediction_mm})

    orthogonalise_lazy = (pl.struct(pl.col('prediction'), pl.col('prediction_mm'))
                          .map_elements(orthogonalise, return_dtype=pl.List(pl.Float64)))

    return (df.group_by("era", maintain_order=True)
            .agg(orthogonalise_lazy.alias('prediction_orthogonalised')).explode('prediction_orthogonalised')
            .select('prediction_orthogonalised')
            .to_series())


def orthogonalise(series: pl.Series) -> np.ndarray:
    """
    Orthogonalises predictions with respect to the meta-model predictions by projecting
    prediction onto mm_prediction, then subtracting that projection from prediction.

    Arguments:
        series: a Polars struct column

    Returns:
        np.ndarray - the orthogonalised series as ndarray/list
    """
    prediction: pl.Series = series.struct.field("prediction")
    prediction_mm: pl.Series = series.struct.field("prediction_mm")

    ranked_prediction = (prediction.rank(method="average") - 0.5) / prediction.count()
    gauss_ranked_prediction = norm.ppf(ranked_prediction)  # gaussianise predictions
    ranked_prediction_mm = (prediction_mm.rank(method="average") - 0.5) / prediction_mm.count()
    gauss_ranked_prediction_mm = norm.ppf(ranked_prediction_mm)  # gaussianise meta-model predictions

    return gauss_ranked_prediction - np.outer(gauss_ranked_prediction_mm,
                                              (gauss_ranked_prediction.T @ gauss_ranked_prediction_mm) /
                                              (gauss_ranked_prediction_mm.T @ gauss_ranked_prediction_mm)).flatten()
