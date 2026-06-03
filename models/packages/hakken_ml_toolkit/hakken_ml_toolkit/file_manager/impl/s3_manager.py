from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from hakken_ml_toolkit.file_manager.core.contracts.file_manager import FileManager
from hakken_ml_toolkit.file_manager.core.values.exceptions import (
    BucketNotFoundError,
    InvalidCredentialsError,
)


class S3Manager(FileManager):
    def __init__(self) -> None:
        try:
            self.client = boto3.client("s3")
        except ClientError as e:
            raise InvalidCredentialsError() from e

    def _strip_s3_prefix(self, remote_path: str) -> str:
        if remote_path.startswith("s3://"):
            return remote_path[5:]
        return remote_path

    def extract_bucket_name(self, remote_path: str) -> str:
        if remote_path.startswith("s3://"):
            msg = "Remove the s3 prefix s3:// first"
            raise RuntimeError(msg)

        parts = remote_path.split("/", 1)

        return parts[0]

    def extract_prefix(self, remote_path: str) -> str:
        parts = remote_path.split("/", 1)

        if len(parts) == 1:
            return ""
        return parts[1]

    def find(self, folder_path: str | None = None, recursive: bool = False) -> list[dict]:
        if recursive:
            raise NotImplementedError()

        if folder_path is None:
            return self._list_buckets()

        folder_path_ = self._strip_s3_prefix(folder_path)

        bucket_name = self.extract_bucket_name(remote_path=folder_path_)

        prefix = self.extract_prefix(remote_path=folder_path_)

        return self._list_objects(bucket_name=bucket_name, prefix=prefix)

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        remote_path_obj = Path(self._strip_s3_prefix(remote_path))

        local_path_obj = Path(local_path)

        if not bool(remote_path_obj.suffix):
            remote_path_obj = remote_path_obj / local_path_obj.name

        logger.info(f"Uploading fie from {local_path} to {remote_path_obj}...")

        if not local_path_obj.is_file():
            logger.info("...failed!")
            return False

        if metadata is not None:
            raise NotImplementedError()

        try:
            bucket_name = self.extract_bucket_name(remote_path=str(remote_path_obj))

            key = str(self.extract_prefix(remote_path=str(remote_path_obj)))

            self.client.upload_file(str(local_path_obj), bucket_name, key)

        except ClientError:
            logger.info("...failed!")
            return False
        else:
            logger.info("...success!")
            return True

    def upload_folder(
        self,
        local_path: str,
        remote_path: str,
    ) -> bool:
        local_path_obj = Path(local_path)
        remote_path_obj = Path(self._strip_s3_prefix(remote_path))
        if not local_path_obj.is_dir():
            return False

        try:
            bucket_name = self.extract_bucket_name(str(remote_path_obj))
            prefix = self.extract_prefix(str(remote_path_obj))
            for file_path in local_path_obj.glob("**/*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path_obj)
                    key = f"{prefix}/{rel_path}"
                    self.client.upload_file(str(file_path), bucket_name, key)

        except ClientError:
            return False
        else:
            return True

    def download_file(self, remote_path: str, local_path: str) -> bool:
        remote_path_obj = Path(self._strip_s3_prefix(remote_path))

        local_path_obj = Path(local_path)

        bucket_name = self.extract_bucket_name(remote_path=str(remote_path_obj))
        key = self.extract_prefix(remote_path=str(remote_path_obj))

        local_path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(
                Bucket=bucket_name, Key=str(key), Filename=str(local_path_obj)
            )
        except ClientError:
            logger.exception(f"{bucket_name}/{key} --> {local_path}")
            return False
        else:
            logger.info(f"{bucket_name}/{key} --> {local_path}")
            return True

    def download_folder(self, remote_path: str, local_path: str) -> bool:
        remote_path_obj = Path(self._strip_s3_prefix(remote_path))

        local_path_obj = Path(local_path)

        bucket_name = self.extract_bucket_name(remote_path=str(remote_path_obj))
        prefix = self.extract_prefix(remote_path=str(remote_path_obj))

        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            objects = self._list_objects(bucket_name, prefix)
            if len(objects) == 0:
                return False

            local_path_obj.mkdir(parents=True, exist_ok=True)

            for obj in objects:
                file_path = local_path_obj / Path(obj["Key"]).name
                self.client.download_file(bucket_name, obj["Key"], str(file_path))

        except ClientError:
            return False
        else:
            return True

    def _list_buckets(self) -> list[dict]:
        try:
            paginator = self.client.get_paginator("list_buckets")
            pages = paginator.paginate()
            all_buckets = []
            for page in pages:
                if "Buckets" in page:
                    all_buckets.extend(page["Buckets"])

        except ClientError as e:
            raise InvalidCredentialsError() from e
        else:
            return all_buckets

    def _list_objects(self, bucket_name: str, prefix: str) -> list[dict]:
        try:
            paginator = self.client.get_paginator("list_objects_v2")

            all_objects = []

            pages = paginator.paginate(Bucket=bucket_name, Prefix=str(prefix))
            for page in pages:
                if "Contents" in page:
                    all_objects.extend(page["Contents"])

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                raise BucketNotFoundError(bucket_name) from e
            raise
        else:
            return all_objects
