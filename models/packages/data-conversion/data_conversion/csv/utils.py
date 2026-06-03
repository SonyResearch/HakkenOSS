from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable


class CSVUtils:
    @staticmethod
    def get_number_of_rows(csv_filepath: str, has_header: bool = True) -> int:
        with open(os.path.expanduser(csv_filepath)) as f:
            n_rows = sum(1 for _ in f)
        if has_header:
            n_rows -= 1
        return n_rows

    @staticmethod
    def apply_function_to_csv_in_chunks(
        csv_filepath: str,
        callable_function: Callable[..., Any],
        chunk_size: int,
        read_csv_kwargs: dict[str, Any] | None = None,
        callable_kwargs: dict[str, Any] | None = None,
        show_progress_bar: bool = True,
    ) -> list[Any]:
        """
        Applies iteratively a function to a large csv by dividing it in chunks.

        Args:
            csv_filepath: The path to the large csv file.
            callable_function: The callable function. It must accept the dataframe chunk as first
                positional argument.
            chunk_size: The chunk size.
            read_csv_kwargs: Arguments to pass to pd.read_csv.
            callable_kwargs: Arguments to pass to the callable function.
            show_progress_bar: Whether to show progress using tqdm.

        Returns:
            Returns the results of the functions in a list (where each item is the result on a
            chunk).
        """
        results = []
        if read_csv_kwargs is None:
            read_csv_kwargs = {}
        if callable_kwargs is None:
            callable_kwargs = {}
        if show_progress_bar:
            logging.info("Calculating file size...")
            total = math.ceil(CSVUtils.get_number_of_rows(csv_filepath) / chunk_size)
        else:
            total = None
        with pd.read_csv(csv_filepath, chunksize=chunk_size, **read_csv_kwargs) as reader:
            for chunk in tqdm.tqdm(reader, desc="Processing csv chunks", total=total):
                result = callable_function(chunk, **callable_kwargs)
                results.append(result)
        return results
