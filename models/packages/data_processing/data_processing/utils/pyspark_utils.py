import shutil
from pathlib import Path

from data_processing.values import DEFAULT_SEPARATOR, SEPARATOR_SUFFIX_DICT, UNKNOWN_SUFFIX


def write_to_single_dsv_file(output_dir: str, sep: str = DEFAULT_SEPARATOR):
    """Writes to a single file a CSV or TSV-like file with header"""
    output_path = Path(output_dir)
    suffix = SEPARATOR_SUFFIX_DICT.get(sep, UNKNOWN_SUFFIX)
    final_file = output_path.with_suffix(suffix)

    first_file = True
    with final_file.open("wb") as outfile:
        for file_path in sorted(output_path.iterdir()):
            if file_path.name.startswith("part-") and file_path.suffix == ".csv":
                with file_path.open("rb") as infile:
                    if first_file:
                        shutil.copyfileobj(infile, outfile)
                        first_file = False
                    else:
                        # Skip header line
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)

    # Remove the temp folder
    shutil.rmtree(output_path)
