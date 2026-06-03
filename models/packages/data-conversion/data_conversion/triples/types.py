from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

NPEntity = np.int_
NPRelation = np.int_
NPTriple = Annotated[npt.NDArray[np.int_], Literal[3]]

NPArrayOfEntities = Annotated[npt.NDArray[np.int_], Literal["N"]]
NPArrayOfRelations = Annotated[npt.NDArray[np.int_], Literal["N"]]
NPArrayOfTriples = Annotated[npt.NDArray[np.int_], Literal["N", 3]]
