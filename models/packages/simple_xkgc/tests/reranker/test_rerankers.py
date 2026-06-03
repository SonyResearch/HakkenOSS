import pandas as pd
import pytest

from hakken_explainer.reranker import ExplanationReranker, PathwayReranker, ScoreReranker


@pytest.fixture
def sample_explanations() -> pd.DataFrame:
    """Sample explanations DataFrame with mixed positive/negative scores."""
    return pd.DataFrame(
        {
            "pathway": ["A", "A", "B", "B", "C"],
            "score": [0.8, -0.2, -0.1, -0.9, 0.3],
            "explanation": ["exp1", "exp2", "exp3", "exp4", "exp5"],
        }
    )


def test_score_reranker_mixed_scores(sample_explanations: pd.DataFrame) -> None:
    """Test ScoreReranker sorts by score descending with mixed pos/neg scores."""
    reranker = ScoreReranker()
    result = reranker.rerank(sample_explanations)

    expected_scores = [0.8, 0.3, -0.1, -0.2, -0.9]
    assert result["score"].tolist() == expected_scores


def test_pathway_reranker_mixed_scores(sample_explanations: pd.DataFrame) -> None:
    """Test PathwayReranker prioritizes highest score per pathway (pos/neg)."""
    reranker = PathwayReranker()
    result = reranker.rerank(sample_explanations)

    # Top per pathway: A(0.8), C(0.3), B(-0.1) - highest from each pathway
    top_pathways = result.head(3)
    assert set(top_pathways["pathway"]) == {"A", "B", "C"}
    assert top_pathways["score"].tolist() == [0.8, 0.3, -0.1]

    # Remaining rows: A(-0.2), B(-0.9) - sorted by score descending
    remaining = result.tail(2)
    assert remaining["score"].tolist() == [-0.2, -0.9]


def test_pathway_reranker_all_negative_scores() -> None:
    """Test PathwayReranker works correctly with all negative scores."""
    df = pd.DataFrame(
        {
            "pathway": ["X", "X", "Y"],
            "score": [-0.1, -0.5, -0.3],
            "explanation": ["a", "b", "c"],
        }
    )

    reranker = PathwayReranker()
    result = reranker.rerank(df)

    assert result["score"].tolist() == [-0.1, -0.3, -0.5]
    assert result["pathway"].tolist() == ["X", "Y", "X"]


@pytest.mark.parametrize("reranker", [ScoreReranker(), PathwayReranker()])
def test_preserves_all_rows(
    sample_explanations: pd.DataFrame, reranker: ExplanationReranker
) -> None:
    """Test both rerankers preserve all original rows."""
    result = reranker.rerank(sample_explanations)

    assert len(result) == len(sample_explanations)
    assert set(result["explanation"]) == set(sample_explanations["explanation"])
