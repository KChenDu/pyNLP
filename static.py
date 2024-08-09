from pathlib import Path
from urllib.request import urlretrieve


HANLP_DATA_PATH = Path(__file__).parent / 'data'


def download(data_url: str, dest_path: Path) -> None:
    urlretrieve(data_url, dest_path)


def remove_file(path: Path) -> None:
    path.unlink()
