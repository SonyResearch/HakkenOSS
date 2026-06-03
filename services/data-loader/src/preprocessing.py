"""
This script processes the raw relations_cleaned.csv
file to nodes.csv and edges.csv files. Those files are 
in the correct format for the neo4j-admin import tool.
"""

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    collect_list,
    concat,
    concat_ws,
    initcap,
    monotonically_increasing_id,
    regexp_extract,
    regexp_replace,
    split,
    when,
    upper,
    lit,
)


def connect() -> SparkSession:
    """Creates or gets Spark session object.

    Returns:
        SparkSession: the spark session
    """
    spark_session = (
        SparkSession.builder.appName("Cleaner")
        .master("local[*]")
        .config("spark.executor.memory", "7g")
        .config("spark.driver.memory", "200g")
        .config("spark.driver.maxResultSize", "150g")
        .getOrCreate()
    )
    return spark_session


def load_csv(spark_session: SparkSession, path: str, sep: str = "\t") -> DataFrame:
    """Loads a CSV file into a dataframe.

    Args:
        spark (SparkSession): spark session.
        path (str): path of CSV file.
        sep (str, optional): separator of CSV file. Defaults to "\t".

    Returns:
        Dataframe: the loaded dataframe
    """
    df = spark_session.read.option("delimiter", sep).csv(
        path, header=True, inferSchema=True
    )

    # create id column to make the joins
    df = df.withColumn("id", monotonically_increasing_id())

    # clean columns from special characters that may interfer with the neo4j-admin import
    df = df.withColumn("store_srcsent", regexp_replace("store_srcsent", '\\"', ""))
    df = df.withColumn("store_srcsent", regexp_replace("store_srcsent", "\\n", ""))
    df = df.withColumn("subject", regexp_replace("subject", '\\"', ""))
    df = df.withColumn("subject", regexp_replace("subject", "\\n", ""))
    df = df.withColumn("object", regexp_replace("object", '\\"', ""))
    df = df.withColumn("subject", regexp_replace("subject", "\\n", ""))

    return df


def write_csv(df: DataFrame, path: str) -> None:
    """Writes a dataframe into a single CSV file separated by ";".

    Args:
        df (DataFrame): dataframe to be written.
        path (str): path to write the CSV file.
    """
    df.repartition(1).write.option("sep", "\t").csv(path, header=True, mode="overwrite")

    df.unpersist()
    return


def prepare_nodes(df: DataFrame, node_type: str) -> DataFrame:
    """Extracts nodes, their lablels, and parameters from the raw data.

    Args:
        df (DataFrame): raw data
        node_type (str): whether a node is "subject" or "object"

    Returns:
        DataFrame: dataframe that contains nodes information (ocid, concept, label).
    """

    relation_col = df.select("relation")
    if node_type == "subject":
        labels_df = relation_col.withColumn(
            "relation", regexp_extract("relation", r"\[([a-zA-Z]+)\]", 1)
        )
        # Handle the case where the relation is malformed: [compound-relatesTo-protein-IC50]
        labels_df = labels_df.withColumn(
            "relation",
            when(col("relation") == "", "Compound").otherwise(col("relation")),
        )
    elif node_type == "object":
        labels_df = relation_col.withColumn(
            "relation", regexp_extract("relation", r"\[([a-zA-Z\s?]+)\]$", 1)
        )
        # Handle the case where the relation is malformed: [compound-relatesTo-protein-IC50]
        labels_df = labels_df.withColumn(
            "relation",
            when(col("relation") == "", "Protein-IC50").otherwise(col("relation")),
        )
    # Convert space separated values (e.g. population group) to concatenated title case (i.e. PopulationGroup)
    labels_df = labels_df.withColumn("relation", initcap(col("relation")))
    labels_df = labels_df.withColumn(
        "relation",
        when(
            col("relation").contains(" "), concat_ws("", split("relation", " "))
        ).otherwise(col("relation")),
    )

    labels_df = labels_df.withColumn("id", monotonically_increasing_id())
    labels_df = labels_df.withColumnRenamed("relation", "label_" + node_type)

    nodes = df.select("id", "ocid_" + node_type, node_type)
    nodes = nodes.join(labels_df, "id", "outer")
    nodes = nodes.withColumnRenamed(node_type, "concept")
    nodes = nodes.withColumnRenamed("ocid_" + node_type, "ocid")

    return nodes


def extract_nodes(df: DataFrame) -> DataFrame:
    """Extracts nodes information from the raw dataframe.

    Args:
        df (DataFrame): original dataframe that contains raw information.

    Returns:
        DataFrame: dataframe that contains node infomration
    """
    subject_nodes = prepare_nodes(df, "subject")
    object_nodes = prepare_nodes(df, "object")

    nodes = subject_nodes.union(object_nodes)
    nodes = nodes.drop("id")
    # some nodes may have the same ocid but different labels, we make them to multilabel nodes.
    nodes = nodes.groupBy("ocid", "concept").agg(
        concat_ws("|", collect_list(nodes.label_subject)).alias(":LABEL")
    )
    # add :ID column for neo4j admin-import
    nodes = nodes.withColumn("ocid:ID", nodes["ocid"])
    nodes = nodes.dropDuplicates()
    return nodes


def extract_edges(df: DataFrame) -> DataFrame:
    """
    Extracts edges information from the raw dataframe.

    Args:
        df (Dataframe): raw dataframe that contains raw information.
    Return:
        DataFrame: dataframe that contains all edge related information.
    """

    df = df.withColumn(
        "relation", regexp_extract("relation", r"(\])\s([\w\s]+)\s(\[)", 2)
    )
    df = df.withColumn("relation", upper(col("relation"))).withColumn(
        "relation", regexp_replace(col("relation"), " ", "_")
    )

    df = df.withColumnRenamed("store_source_date", "date:date")
    df = df.withColumn(
        "date:date", regexp_extract(df["date:date"], "(\d+)?-?(\d+)?-?(\d\d\d\d)?", 1)
    )
    df = df.withColumn("date:date", concat(df["date:date"], lit("-01-01")))
    df = df.withColumnRenamed("#ocid_relation", "ocid_relation")

    edges = df.select(
        "relation",
        "ocid_subject",
        "ocid_object",
        "ocid_relation",
        "date:date",
        "srcrep",
        "val_doc_id",
        "srcsection",
        "store_source_id",
        "store_source_link",
        "store_srcsent",
        "store_source_db",
    )
    # rename relation to :TYPE and ocid_subject to :START_ID, and ocid_object to :END_ID,
    edges = edges.withColumnsRenamed(
        {"relation": ":TYPE", "ocid_subject": ":START_ID", "ocid_object": ":END_ID"}
    )
    edges = edges.dropDuplicates()
    return edges


if __name__ == "__main__":
    try:
        spark = connect()

        raw_data = load_csv(spark, "/import/relations_cleaned.csv")

        nodes = extract_nodes(raw_data)
        write_csv(nodes, "/import/nodes")

        edges = extract_edges(raw_data)
        raw_data.unpersist()
        write_csv(edges, "/import/edges")

        spark.stop()

    except Exception as error:
        if spark is not None:
            spark.stop()

        print(error)
