import argparse
import sys
from pathlib import Path

from dependency_injector.wiring import inject

from embeddings import WiringConfig, setup_container
from embeddings.core.actions.embeddings import (
    ExtractEmbeddingsAction,
    ExtractEmbeddingsActionInput,
)


@inject
def call_action(args):
    action_input = ExtractEmbeddingsActionInput(
        ontology_file_path=args.ontology_file,
        entities_file_path=args.entities_file,
        output_file_path=args.output_file,
    )
    ExtractEmbeddingsAction.run(input=action_input)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run embeddings")
    parser.add_argument(
        "--ontology_file",
        type=str,
        required=True,
        help="Please provide the path to the ontology file",
    )
    parser.add_argument(
        "--entities_file",
        type=str,
        required=True,
        help="Please provide the path to the entities file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Please provide the path to the yaml configuration file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Please provide the path to the output file",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not Path(args.ontology_file).exists():
        raise Exception(f"Ontology file {args.ontology_file} does not exist")
    if not Path(args.entities_file).exists():
        raise Exception(f"Entities file {args.entities_file} does not exist")
    if args.config_file and not Path(args.config_file).exists():
        raise Exception(f"Configuration file {args.config_file} does not exist")
    if not Path("/".join(args.output_file.split("/")[:-1])).exists():
        Path("/".join(args.output_file.split("/")[:-1])).mkdir(parents=True)

    if args.config_file:
        setup_container(
            Path(args.config_file), wiring_config=WiringConfig(modules=[sys.modules[__name__]])
        )
    else:
        setup_container(wiring_config=WiringConfig(modules=[sys.modules[__name__]]))

    call_action(args)


if __name__ == "__main__":
    main()
