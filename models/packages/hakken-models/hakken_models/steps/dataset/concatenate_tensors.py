import polars as pl
from zenml import step
from zenml.integrations.polars.materializers import PolarsMaterializer


@step(output_materializers=PolarsMaterializer)
def concatenate_dataframes(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> pl.DataFrame:
    return pl.concat([train_df, val_df, test_df])
