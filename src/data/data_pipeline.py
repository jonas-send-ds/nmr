
from src.data.download_data import download_data
from src.data.split_data import split_data


def main() -> None:
    download_data()
    split_data()


if __name__ == '__main__':
    main()
