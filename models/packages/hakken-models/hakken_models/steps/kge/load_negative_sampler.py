from zenml import step

from hakken_models.core.configs.train_common import NegSamplerConfig
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.fact_validator.tensor_based import TensorFactValidator
from hakken_models.negative_samplers import NegativeSampler, neg_sampler_registry


@step(enable_cache=False)
def load_negative_sampler_step(
    dataset: DatasetDeployment, neg_sampler_config: NegSamplerConfig
) -> NegativeSampler:
    facts_tensor = dataset.get_facts_tensor(split_name="train")
    fact_validator = TensorFactValidator(
        positive_facts=facts_tensor, fact_key_length=3, device=None, batch_size=10_000
    )

    neg_sampler_cls = neg_sampler_registry.get(neg_sampler_config.name)
    return neg_sampler_cls(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        fact_validator=fact_validator,
        **neg_sampler_config.kwargs,
    )
