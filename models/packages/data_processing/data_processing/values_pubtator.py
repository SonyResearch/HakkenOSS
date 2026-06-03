PUBTATOR_API_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
)
SLEEP_BETWEEN_CALLS = 0.5

CACHE_PATH = "cache"
PUBTATOR_CACHE_FILE = f"{CACHE_PATH}/pubtator_cache.jsonl"
PUBTATOR_PUBLICATION_METADATA_CACHE_FILE = f"{CACHE_PATH}/pubtator_publication_metadata_cache.jsonl"
PMID_TRACK_FILE = f"{CACHE_PATH}/processed_pmids.txt"
