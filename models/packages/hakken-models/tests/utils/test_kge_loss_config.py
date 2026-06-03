from hakken_models.core.configs.negative_strategy import NegativeAggregation, NegativeStrategyConfig
from hakken_models.core.configs.train_common import LossConfig


def test_loss_config_dict_overwrites_neg_strategy_from_negative_strategy() -> None:
    loss = LossConfig(
        name="RankingRelationLoss",
        kwargs={
            "entity_loss": "MarginRankingLoss",
            "entity_loss_kwargs": {"margin": 1.0},
            "neg_strategy": "hardest",
            "rel_loss_weight": 0.0,
        },
    )
    neg = NegativeStrategyConfig(name=NegativeAggregation.MEAN)

    out = loss.with_kge_negative_strategy(neg)

    assert out["kwargs"]["neg_strategy"] == "mean"


def test_loss_config_dict_matches_negative_strategy_hardest() -> None:
    loss = LossConfig(name="RankingRelationLoss", kwargs={"rel_loss_weight": 0.0})
    neg = NegativeStrategyConfig(name=NegativeAggregation.HARDEST)

    out = loss.with_kge_negative_strategy(neg)

    assert out["kwargs"]["neg_strategy"] == "hardest"
