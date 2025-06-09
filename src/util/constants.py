
from pathlib import Path


def find_project_root(current_path, marker='src')-> Path | None:
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError("Project root not found.")


DATA_PATH: Path = find_project_root(Path.cwd()) / 'data'
PATH_RAW_TRAIN_SET: str = str(DATA_PATH) + "/raw/train.parquet"
PATH_RAW_VALIDATE_SET: str = str(DATA_PATH) + "/raw/validate.parquet"
PATH_RAW_META_MODEL: str = str(DATA_PATH) + "/raw/meta_model.parquet"
