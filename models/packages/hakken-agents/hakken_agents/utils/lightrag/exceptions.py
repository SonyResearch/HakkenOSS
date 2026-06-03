class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ""):
        msg = (
            f"Pipeline namespace '{namespace}' not found.\n"
            f"\n"
            f"Pipeline status should be auto-initialized by initialize_storages().\n"
            f"If you see this error, please ensure:\n"
            f"\n"
            f"  1. You called await rag.initialize_storages()\n"
            f"  2. For multi-workspace setups, each LightRAG instance was properly initialized\n"
            f"\n"
            f"Standard initialization:\n"
            f"  rag = LightRAG(workspace='your_workspace')\n"
            f"  await rag.initialize_storages()  # Auto-initializes pipeline_status\n"
            f"\n"
            f"If you need manual control (advanced):\n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status(workspace='your_workspace')"
        )
        super().__init__(msg)


class StorageNotInitializedError(RuntimeError):
    """Raised when storage operations are attempted before initialization."""

    def __init__(self, storage_type: str = "Storage"):
        super().__init__(
            f"{storage_type} not initialized. Please ensure proper initialization:\n"
            f"\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()  # Required - auto-initializes pipeline_status\n"
            f"\n"
            f"See: https://github.com/HKUDS/LightRAG#important-initialization-requirements"
        )
