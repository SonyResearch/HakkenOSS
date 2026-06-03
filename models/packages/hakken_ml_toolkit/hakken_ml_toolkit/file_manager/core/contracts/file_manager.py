import os
from abc import ABC, abstractmethod


class FileManager(ABC):
    @abstractmethod
    def find(self, folder_path: str | None = None, recursive: bool = False) -> list[dict]:
        """
        Find resources in the storage system.

        Args:
            folder_path: Path to the folder to search in. If None, lists root-level resources.
            recursive: Whether to search recursively through subfolders.

        Returns:
            A list of dictionaries containing information about the found resources.
        """
        pass

    @abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        pass

    @abstractmethod
    def upload_folder(
        self,
        local_path: str,
        remote_path: str,
    ) -> bool:
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> bool:
        pass

    def download_files(
        self, remote_folder: str, local_folder: str, basename_list: list[str]
    ) -> list[bool]:
        success_list = []

        for basename in basename_list:
            remote_path = os.path.join(remote_folder, basename)
            local_path = os.path.join(local_folder, basename)
            success = self.download_file(remote_path=remote_path, local_path=local_path)

            success_list.append(success)

        return success_list

    @abstractmethod
    def download_folder(self, remote_path: str, local_path: str) -> bool:
        pass
