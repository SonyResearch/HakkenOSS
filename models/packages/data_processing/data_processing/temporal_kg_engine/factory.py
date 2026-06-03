from loguru import logger

from .base import TemporalKGEngine
from .enums import TKGEngine
from .in_memory import InMemoryTKGEngine, InMemoryTKGSettings
from .settings import TemporalKGSettings

# Registry: engine name → (SettingsClass, EngineClass)
_ENGINE_REGISTRY: dict[TKGEngine, tuple[type[TemporalKGSettings], type[TemporalKGEngine]]] = {
    TKGEngine.IN_MEMORY: (InMemoryTKGSettings, InMemoryTKGEngine),
}


class TKGFactory:
    @staticmethod
    def from_env(name: str) -> TemporalKGEngine:
        """
        Create a Temporal KG engine instance based on the given name.
        Supported engines: 'in memory'
        """
        try:
            engine = TKGEngine(name.lower().strip())
        except ValueError as exc:
            msg = f"Unknown engine '{name}'. Supported: {', '.join(e.value for e in TKGEngine)}"
            raise ValueError(msg) from exc
        settings_cls, engine_cls = _ENGINE_REGISTRY[engine]

        settings = settings_cls()

        logger.debug(settings)

        return engine_cls.from_settings(settings)

    @staticmethod
    def list_engines() -> list[str]:
        """Return sorted list of supported engine names."""
        return sorted(TKGEngine.values())
