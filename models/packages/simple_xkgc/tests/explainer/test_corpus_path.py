import pytest
from torch import Tensor

from hakken_explainer.candidate_finder.corpus.path import CorpusPathFinder
from tests.explainer.base import BaseHakkenExplainerTest


class TestHakkenExplainerWithCorpusPathFinder(BaseHakkenExplainerTest):
    @pytest.fixture
    def candidate_finder(self, search_space: Tensor):
        path_finder = CorpusPathFinder(max_candidates=5000, undirected=True)

        path_finder.setup(facts_batch=search_space, cache_folder=None)
        return path_finder
