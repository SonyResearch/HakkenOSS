import os
import uuid
from typing import TypedDict

import pytest
from dotenv import load_dotenv
from pymilvus import MilvusClient


class _MilvusConnectionInfo(TypedDict):
    host: str
    user: str
    password: str
    collection_name: str


@pytest.fixture
def milvus_connection_info(test_data_root):
    test_env_path = test_data_root / "test_milvus_config.env"
    load_dotenv(test_env_path, override=True)

    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    user = os.getenv("MILVUS_USER", "root")
    password = os.getenv("MILVUS_PASSWORD", "Milvus")
    collection_name = "_" + uuid.uuid4().hex[:16]

    yield _MilvusConnectionInfo(
        host=uri, user=user, password=password, collection_name=collection_name
    )

    client = MilvusClient(uri=uri, user=user, password=password)
    client.drop_collection(collection_name)
