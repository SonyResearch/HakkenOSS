from typing import List, Tuple, Union

import numpy as np

Embeddings = List[np.ndarray]

Entities = List[str]
Literal = Union[float, str]

Literals = List[List[Union[Literal, Tuple[Literal, ...]]]]
