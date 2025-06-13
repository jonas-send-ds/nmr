### Dependency management
Dependencies are managed via Poetry.

To create the correct virtual environment, run the following in the project's root directory:
```console
poetry env activate
```

### Analysis
* `ranking_vs_regression.ipynb`
* `performance_over_time.ipynb`
* `mmc_stability_and_approximation.ipynb`

### Linter
The project can be linted with ruff via `ruff check`. The Rules are configured in `pyproject.toml`.