from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from hakken_ml_toolkit.file_manager import S3Manager
from hakken_ml_toolkit.file_manager.core.values.exceptions import (
    BucketNotFoundError,
    InvalidCredentialsError,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_init_with_invalid_credentials() -> None:
    # Arrange
    with patch("boto3.client") as mock_client:
        mock_client.side_effect = ClientError(
            error_response={"Error": {"Code": "InvalidAccessKeyId"}},
            operation_name="CreateClient",
        )

        # Act & Assert
        with pytest.raises(InvalidCredentialsError):
            S3Manager()


def test_list_buckets_success() -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_paginator = MagicMock()
        mock_boto3_client_return.get_paginator.return_value = mock_paginator

        mock_pages = [
            {"Buckets": [{"Name": "bucket1", "CreationDate": datetime(2024, 1, 1)}]},
            {"Buckets": [{"Name": "bucket2", "CreationDate": datetime(2024, 1, 2)}]},
        ]
        mock_paginator.paginate.return_value = mock_pages

        mock_boto3_client.return_value = mock_boto3_client_return
        s3_manager = S3Manager()
        result: list[dict] = s3_manager.find()

        mock_boto3_client_return.get_paginator.assert_called_once_with("list_buckets")

        mock_paginator.paginate.assert_called_once()

        assert isinstance(result, list)
        for result_i in result:
            assert isinstance(result_i, dict)
        assert len(result) == len(mock_pages)
        result_df = pd.DataFrame(result)
        assert list(result_df.columns) == ["Name", "CreationDate"]
        assert result_df["Name"].tolist() == ["bucket1", "bucket2"]


def test_list_buckets_empty() -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_paginator = MagicMock()
        mock_boto3_client_return.get_paginator.return_value = mock_paginator

        mock_pages: list[dict] = []
        mock_paginator.paginate.return_value = mock_pages

        mock_boto3_client.return_value = mock_boto3_client_return
        s3_manager = S3Manager()

        result = s3_manager.find()

        assert isinstance(result, list)
        assert len(result) == 0


def test_list_objects_success() -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_paginator = MagicMock()
        mock_boto3_client_return.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "Contents": [
                    {
                        "Key": "file1.txt",
                        "LastModified": datetime(2024, 1, 1),
                        "Size": 100,
                        "StorageClass": "STANDARD",
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        mock_boto3_client.return_value = mock_boto3_client_return

        s3_manager = S3Manager()

        result = s3_manager.find("test-bucket/prefix")

        assert isinstance(result, list)
        assert len(result) == 1

        result_df = pd.DataFrame(result)
        assert list(result_df.columns) == [
            "Key",
            "LastModified",
            "Size",
            "StorageClass",
        ]
        assert result_df["Key"].iloc[0] == "file1.txt"
        assert result_df["StorageClass"].iloc[0] == "STANDARD"


def test_list_objects_bucket_not_found() -> None:
    with patch("boto3.client") as mock_boto3_client:
        # Set up the mock client
        mock_boto3_client_return = MagicMock()

        # Create the mock paginator that will be used in _list_objects
        mock_paginator = MagicMock()
        mock_boto3_client_return.get_paginator.return_value = mock_paginator

        # Set up the error that will be raised during pagination
        error_response: dict[str, dict[str, str]] = {
            "Error": {
                "Code": "NoSuchBucket",
                "Message": "The specified bucket does not exist",
            }
        }
        mock_paginator.paginate.side_effect = ClientError(
            error_response=cast("Any", error_response),
            operation_name="ListObjectsV2",  # type: ignore[arg-type]
        )

        # Return our mocked client when boto3.client is called
        mock_boto3_client.return_value = mock_boto3_client_return

        # Create the S3Manager instance
        s3_manager = S3Manager()

        # Test that BucketNotFoundError is raised when find is called
        with pytest.raises(BucketNotFoundError):
            s3_manager.find("non-existent-bucket/prefix")


def test_download_file_success(tmp_path: Path) -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_boto3_client.return_value = mock_boto3_client_return

        s3_manager = S3Manager()
        local_path = tmp_path / "downloads/local_test.txt"
        remote_path = "test-bucket/test.txt"

        success = s3_manager.download_file(remote_path=remote_path, local_path=str(local_path))

        assert success
        assert local_path.parent.exists()

        mock_boto3_client_return.download_file.assert_called_once_with(
            Bucket="test-bucket",
            Key="test.txt",
            Filename=str(local_path),
        )


@pytest.mark.parametrize(
    "remote_path",
    [
        "test-bucket/prefix/test.txt",
        "test-bucket/prefix",
    ],
)
def test_upload_file_success(tmp_path: Path, remote_path: str) -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_boto3_client.return_value = mock_boto3_client_return
        s3_manager = S3Manager()

        local_path = tmp_path / "test.txt"
        local_path.write_text("test content")

        result = s3_manager.upload_file(local_path=str(local_path), remote_path=remote_path)

        assert result is True
        mock_boto3_client_return.upload_file.assert_called_once_with(
            str(local_path), "test-bucket", "prefix/test.txt"
        )


def test_upload_file_nonexistent_file(tmp_path: Path) -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client_return = MagicMock()

        mock_boto3_client.return_value = mock_boto3_client_return
        s3_manager = S3Manager()

        local_path = tmp_path / "nonexistent.txt"
        remote_path = "test-bucket/prefix"

        result = s3_manager.upload_file(str(local_path), remote_path)

        assert result is False


def test_download_files() -> None:
    with patch("boto3.client") as mock_boto3_client:
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        s3_manager = S3Manager()

        with patch.object(s3_manager, "download_file") as mock_download_file:
            mock_download_file.return_value = True

            remote_folder = "my-bucket/folder"
            local_folder = "/local/path"
            basename_list = ["file1.txt", "file2.csv", "file3.pdf"]

            result = s3_manager.download_files(
                remote_folder=remote_folder,
                local_folder=local_folder,
                basename_list=basename_list,
            )

            assert mock_download_file.call_count == 3
            mock_download_file.assert_any_call(
                remote_path=os.path.join(remote_folder, "file1.txt"),
                local_path=os.path.join(local_folder, "file1.txt"),
            )
            mock_download_file.assert_any_call(
                remote_path=os.path.join(remote_folder, "file2.csv"),
                local_path=os.path.join(local_folder, "file2.csv"),
            )
            mock_download_file.assert_any_call(
                remote_path=os.path.join(remote_folder, "file3.pdf"),
                local_path=os.path.join(local_folder, "file3.pdf"),
            )

            assert result == [True, True, True]

            mock_download_file.reset_mock()
            mock_download_file.side_effect = [True, False, True]

            result = s3_manager.download_files(
                remote_folder=remote_folder,
                local_folder=local_folder,
                basename_list=basename_list,
            )

            assert mock_download_file.call_count == 3

            assert result == [True, False, True]

            mock_download_file.reset_mock()

            result = s3_manager.download_files(
                remote_folder=remote_folder, local_folder=local_folder, basename_list=[]
            )

            # Verify download_file was not called
            assert mock_download_file.call_count == 0

            # Check the empty result list
            assert result == []
