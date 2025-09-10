# built-in dependencies
import hashlib

# package dependencies
from deepface.commons.logger import Logger

logger = Logger()


def find_file_hash(file_path: str, hash_algorithm: str = "sha256") -> str:
    """
    Find the hash of a given file with its content
    Args:
        file_path (str): exact path of a given file
        hash_algorithm (str): hash algorithm
    Returns:
        hash (str)
    """
    hash_func = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()
