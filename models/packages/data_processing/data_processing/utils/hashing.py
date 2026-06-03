import hashlib


def hash_string(a_string: str) -> str:
    """Hashing function that provided unique entities and relations ids"""
    return hashlib.md5(a_string.encode()).hexdigest()
