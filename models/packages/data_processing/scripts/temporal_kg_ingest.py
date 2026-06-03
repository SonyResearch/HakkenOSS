import argparse

from dotenv import load_dotenv
from loguru import logger

from data_processing.temporal_kg_engine.factory import TKGFactory
from data_processing.temporal_kg_engine.settings import IngestSettings

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest data into Temporal Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s in_memory --recreate
        """,
    )

    parser.add_argument(
        "tkg_name",
        type=str,
        choices=TKGFactory.list_engines(),
        help="TKG engine name",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Clean the database before ingestion",
    )

    args = parser.parse_args()

    # Create engine from environment
    logger.info(f"Creating {args.tkg_name} engine from environment")
    engine = TKGFactory.from_env(args.tkg_name)

    ingest_settings = IngestSettings()
    logger.debug(ingest_settings)

    # Override recreate setting if provided via CLI
    recreate = args.recreate if args.recreate else ingest_settings.RECREATE_GRAPH

    engine.connect()

    logger.info("=" * 60)
    logger.info(f"Starting ingestion with {args.tkg_name} engine")
    logger.info(f"Recreate mode: {recreate}")
    logger.info("=" * 60)

    engine.ingest_from_s3(recreate=recreate)

    logger.info("=" * 60)
    logger.info("✅ Ingestion completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
