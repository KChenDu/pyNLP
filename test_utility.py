from pathlib import Path
from static import HANLP_DATA_PATH, download, remove_file
from zipfile import ZipFile


def test_data_path() -> Path:
    """
    获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
    :return:
    """
    data_path = HANLP_DATA_PATH / 'test'
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def ensure_data(data_name: str, data_url: str) -> Path:
    root_path = test_data_path()
    dest_path = root_path / data_name
    if dest_path.exists():
        return dest_path
    if data_url.endswith('.zip'):
        dest_path = dest_path.parent / (dest_path.name + '.zip')
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path.with_suffix('')
    return dest_path
