
from pathlib import Path


def find_project_root(current_path: str | Path, marker: str = 'src')-> Path:
    """
    Finds the root directory of a project by searching for a specific marker file or
    directory upward from a given path.

    :param current_path: The starting directory path from which to begin searching
        for the project root.
    :param marker: The name of the file or directory to look for as a project marker.
        Defaults to "src".
    :return: The path of the project root directory containing the marker as a
        Path object if found, otherwise a FileNotFound exception is raised.
    """
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError("Project root not found.")


DATA_PATH: Path = find_project_root(Path.cwd()) / 'data'
PATH_RAW_TRAIN_SET: str = str(DATA_PATH) + "/raw/train.parquet"
PATH_RAW_VALIDATE_SET: str = str(DATA_PATH) + "/raw/validate.parquet"
PATH_RAW_META_MODEL: str = str(DATA_PATH) + "/raw/meta_model.parquet"

META_MODEL_PERFORMANCE = [.051, .048, .035]
