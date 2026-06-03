from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyrdf2vec.graphs import Vertex  # noqa: F401

Hop = Tuple[Any, Any]

SWalk = Tuple[str, ...]
Walk = Tuple[Any, ...]

Embeddings = List[np.ndarray]

EntityWalks = Dict[str, List[SWalk]]
Entities = List[str]

Literal = Union[float, str]
Literals = List[List[Union[Literal, Tuple[Literal, ...]]]]

Response = List[Dict[str, Dict[str, str]]]
