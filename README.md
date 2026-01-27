# Numerai (v5.0)

### Dependency management
Dependencies are managed via Poetry.

To create the correct virtual environment, run the following in the project's root directory:
```console
poetry env activate
```

### Data
Data is downloaded from Numerai via the Python package `NumerAPI`.

Simply run `src/data/data_pipeline.py` 
which also splits data in three training/validation folds and a dataset containing meta-model predictions.

### Analysis plan
All experiments and analyses are included as notebooks found in `src/notebooks` which are run in the following order:
* `ranking_vs_regression.ipynb` tests whether predicting ranks can outperform regression (both with LightGBM). Regression consistently outperforms ranking.
* `performance_over_time.ipynb` analyses both how average performance varies over eras and how the performance ranking of LightGBM models is correlated between groups of eras.
* `mmc_stability_and_approximation.ipynb` checks how stable correlation of LightGBM models with the meta model is and finds `CORR_PRED_TARGET - (CORR_PRED_MM * CORR_MM_TARGET)` as a good approximation for MMC which is subsequently used in training etc.
* `benchmarks_and_mmc_approximation.ipynb` establishes correlation benchmarks for the three folds for linear models and LightGBM models trained with Numerai's suggested hyperparameters. The "Numerai models" are used to approximate MMC for the three folds.
* `compare_frameworks.ipynb` briefly compares LightGBM, XGBoost, and CatBoost to choose one framework for feature selection.
* `feature_selection.ipynb` runs the Boruta algorithm with simple LightGBM models to iteratively delete features. The resulting selected feature set is stored in `data/results/selected_features.pkl`.
* `tune_hyperparameters.ipynb` runs hyperparameter optimisation (for LightGBM) via `optuna` and stored the result in `data/results/study.pkl`.

The top 20 LightGBM models from hyperparameter optimisation are then trained in `src/model_training/train_models_for_ensembling.py`.

* `test_feature_neutralisation.ipynb` tests whether (partially) subtracting the component of predictions which can be explained with a linear model (by era) improves performance (for LightGBM models). It indicates that this "feature neutralisation" can improve performance.
* `train_ensemble.ipynb` selects models to include in the final ensemble and further analyses the best multiplier for "feature neutralisation". The parameters required to train the models for the ensemble are stored in `data/results/parameters_list.pkl`.

### Submision
For all submission steps, a distinct Python environment that adheres to `src/submission/numerai_requirements.txt` must be used. See also `src/submission/README.md`.

Models are trained in `src/submission/train_models_for_submission.py` and then pickled in a submission function (including feature neutralisation) in `src/submission/prepare_lgb_submission.py`.

### Linter
The project can be linted with ruff via `ruff check`. The Rules are configured in `pyproject.toml`.