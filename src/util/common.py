
import polars as pl
import numpy as np

from scipy.stats import norm


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
