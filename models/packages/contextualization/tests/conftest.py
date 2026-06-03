import os
import uuid
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv


@pytest.fixture
def test_data_root():
    return Path(__file__).parent / "test_data"


@pytest.fixture
def contextualization_container_config_env_path(test_data_root):
    return test_data_root / "test_contextualization_container_config.env"


@pytest.fixture
def encoding_container_config_yaml_path(test_data_root):
    return test_data_root / "test_encoding_container_config.yaml"


@pytest.fixture
def ndjson_publications_path(test_data_root):
    return test_data_root / "test_publications_data.ndjson"


@pytest.fixture
def ndjson_publication_concept_links_path(test_data_root):
    return test_data_root / "test_publication_concept_links_data.ndjson"


@pytest.fixture
def parquet_publications_directory(test_data_root):
    return test_data_root / "parquet" / "publications"


@pytest.fixture
def parquet_publication_concept_links_directory(test_data_root):
    return test_data_root / "parquet" / "publication_concepts"


@pytest.fixture
def ndjson_publications_directory(test_data_root):
    return test_data_root / "ndjson" / "publications"


@pytest.fixture
def ndjson_publication_concept_links_directory(test_data_root):
    return test_data_root / "ndjson" / "publication_concepts"


@pytest.fixture
def postgres_sql_path(test_data_root):
    return test_data_root / "test_postgres_db.sql"


@pytest.fixture
def postgres_connection_string(test_data_root, postgres_sql_path):
    test_env_path = test_data_root / "test_postgres_config.env"
    load_dotenv(test_env_path, override=True)

    host = os.getenv("POSTGRES_HOST", "localhost:5432")
    user = os.getenv("POSTGRES_USER", "test_user")
    password = os.getenv("POSTGRES_PASSWORD", "test_password")
    db_name = "_" + uuid.uuid4().hex[:16]

    conn_str = f"postgresql://{user}:{password}@{host}"
    conn_str_with_db_name = f"postgresql://{user}:{password}@{host}/{db_name}"

    with psycopg.connect(conn_str, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        cur.execute(f'CREATE DATABASE "{db_name}"')

    with psycopg.connect(conn_str_with_db_name, autocommit=True) as conn:
        cur = conn.cursor()
        with open(postgres_sql_path) as f:
            sql = f.read()
        cur.execute(sql)

    yield conn_str_with_db_name

    with psycopg.connect(conn_str, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
