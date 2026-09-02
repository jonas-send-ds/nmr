
## Principles
- Optimise for maintainability: code is read more often than written.
- Documentation is paramount. Always keep the README etc. informative, concise, and up to date.


## Coding Standards
- Type hints are required on all functions. Use type hints also elsewhere whenever the type is not clear from the instantiation code.
- Add docstrings to all classes and functions.
- Express intent in naming and avoid abbreviations.
- Use comments (only) for what the code cannot say.

### Testing
- Test behaviour, not implementation
- Whenever possible, a test should test one specific thing. Group related assertions only when they're testing the same behaviour.
- Tests should follow a clear Given-When-Then structure. Leave one blank line between each of these three sections of a test.
- Make tests independent and order-agnostic.
- Do not mock test data.
- Test data should be:
  - relevant (valid and consistent with the domain)
  - minimal (only create data that is required for the test)
  - isolated (no dependencies with other tests)
  - random where possible
  - reproducible
  - generated in a central location (e.g. a test data factory)


## Preferred Tools
Python and the following packages:
- **Poetry** for dependency management
- **Polars** for data loading and mining (prefer over Pandas whenever possible)
- **Seaborn** for plotting (usually works with Polars data as input)
- **Ruff** for linting
- **ty** for type-checking
- **tqdm** for progress tracking
- **Optuna** for hyperparameter tuning
- **Pydantic** for object validation
- **FastAPI** for APIs
- **pytest** for testing


## Claude-specific instructions
- **README**: After each change, check against the README.md to see whether it needs to be updated.
