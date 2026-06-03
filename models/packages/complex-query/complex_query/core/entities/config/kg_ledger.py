from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import KGLedgerType


class KGLedgerConfigBase(BaseModel):
    config_type: KGLedgerType


class HDF5KGLedgerConfig(KGLedgerConfigBase):
    config_type: Literal[KGLedgerType.HDF5] = KGLedgerType.HDF5

    file_path: Path = Path(".querying_cache/kg.h5")


class InMemoryKGLedgerConfig(KGLedgerConfigBase):
    config_type: Literal[KGLedgerType.IN_MEMORY] = KGLedgerType.IN_MEMORY


class SqliteKGLedgerConfig(KGLedgerConfigBase):
    config_type: Literal[KGLedgerType.SQLITE] = KGLedgerType.SQLITE

    file_path: Path = Path(".querying_cache/kg.sqlite")


KGLedgerConfig: TypeAlias = Annotated[
    HDF5KGLedgerConfig | InMemoryKGLedgerConfig | SqliteKGLedgerConfig,
    Field(discriminator="config_type"),
]
