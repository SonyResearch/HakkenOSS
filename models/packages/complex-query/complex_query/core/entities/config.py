from pydantic_settings import BaseSettings


class OctopusConfig(BaseSettings):
    host: str
    port: int
    username: str
    password: str
    output_host: str | None = None
    output_port: int = 8888
    num_oc_workers: int = 12


class CoreModelConfig(BaseSettings):
    host: str
    port: int
    route: str


class BeamSearchConfig(BaseSettings):
    batched: bool = True
    batch_size: int = 32
    beam_size: int = 5
