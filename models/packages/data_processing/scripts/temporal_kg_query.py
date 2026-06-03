import argparse
import time

import yaml
from dotenv import load_dotenv
from loguru import logger

from data_processing.temporal_kg_engine.factory import TKGFactory
from data_processing.temporal_kg_engine.settings import QuerySettings

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Temporal Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s in-memory --recreate
        """,
    )

    parser.add_argument(
        "tkg_name",
        type=str,
        choices=TKGFactory.list_engines(),
        help="TKG engine name",
    )
    args = parser.parse_args()
    logger.info(f"Creating {args.tkg_name} engine from environment")
    engine = TKGFactory.from_env(args.tkg_name)

    query_settings = QuerySettings()
    logger.debug(query_settings)

    engine.connect()
    engine.set_result_limit(-1)

    with open(query_settings.FILE) as f:
        queries: list[str] = yaml.safe_load(f)

    for query in queries:
        print(f"Query:\n{query}")

        tic = time.time()

        # Example query
        results = engine.query(query, timeout=query_settings.TIMEOUT_MS)
        delay = time.time() - tic

        print(f"[{delay:.2f} secs]Query results ({len(results)}):")
        for i, row in enumerate(results):
            print(f"  {row}")
            if i > 4:
                break

        print("===============================")


if __name__ == "__main__":
    main()
