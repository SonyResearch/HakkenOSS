from dependency_injector import containers, providers
from pydantic_settings import BaseSettings

from embeddings.core.contracts.embeddings import ITransformer, Sampler, Walker
from embeddings.core.contracts.graph_converter import IGraphConverter
from embeddings.core.contracts.graph_loader import IGraphLoader
from embeddings.impl.embeddings import Transformer
from embeddings.impl.pyrdf2vec_graph_converter import RDFConverter
from embeddings.impl.rdf_graph_loader import RDFLoader

# Externals
from pyrdf2vec.samplers import ObjFreqSampler, PredFreqSampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker, WalkletWalker, WLWalker


# TODO: Convert str to enums
class ContainerConfiguration(BaseSettings):
    sampler_type: str = "uniform"
    walker_type: str = "random"
    type_word2vec: str = "skip-gram"
    vector_size: int = 200
    nr_walks: int = 200
    walk_depth: int = 4


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["embeddings"])

    config = providers.Configuration(strict=True, default=ContainerConfiguration().model_dump())

    sampler: providers.Provider[Sampler] = providers.Selector(
        config.sampler_type,
        uniform=providers.Singleton(UniformSampler),
        objfreq=providers.Singleton(ObjFreqSampler),
        predfreq=providers.Singleton(PredFreqSampler),
    )

    walker: providers.Provider[Walker] = providers.Selector(
        config.walker_type,
        random=providers.Singleton(
            RandomWalker,
            max_depth=config.walk_depth(),
            max_walks=config.nr_walks(),
            sampler=sampler(),
        ),
        wl=providers.Singleton(
            WLWalker,
            max_depth=config.walk_depth(),
            max_walks=config.nr_walks(),
            sampler=sampler(),
        ),
        walklet=providers.Singleton(
            WalkletWalker,
            max_depth=config.walk_depth(),
            max_walks=config.nr_walks(),
            sampler=sampler(),
        ),
    )
    transformer: providers.Provider[ITransformer] = providers.Singleton(
        Transformer, config.vector_size, config.type_word2vec, [walker()]
    )

    graph_loader: providers.Provider[IGraphLoader] = providers.Factory(RDFLoader)

    graph_converter: providers.Provider[IGraphConverter] = providers.Factory(RDFConverter)
