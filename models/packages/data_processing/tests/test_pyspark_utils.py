from pathlib import Path

from data_processing.utils.pyspark_utils import write_to_single_dsv_file
from data_processing.values import DEFAULT_SEPARATOR, SEPARATOR_SUFFIX_DICT


class TestWriteToSingleDSVFile:
    def test_write_to_single_dsv_file_basic(self, tmp_path: Path):
        """Test writing a single DSV file from multiple CSV parts with default separator"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Create two CSV parts with headers
        (output_dir / "part-000.csv").write_text("header\nrow1\n")
        (output_dir / "part-001.csv").write_text("header\nrow2\n")
        (output_dir / "readme.txt").write_text("ignore me")

        write_to_single_dsv_file(str(output_dir), sep=DEFAULT_SEPARATOR)

        suffix = SEPARATOR_SUFFIX_DICT.get(DEFAULT_SEPARATOR, ".unknown")
        final_file = output_dir.with_suffix(suffix)

        assert final_file.exists()
        assert final_file.read_text() == "header\nrow1\nrow2\n"
        assert not output_dir.exists()  # temp folder removed

    def test_write_to_single_dsv_file_no_csv_files(self, tmp_path: Path):
        """Test when directory contains no CSV files"""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        (output_dir / "file.txt").write_text("hello")
        (output_dir / "file.json").write_text("{}")

        write_to_single_dsv_file(str(output_dir), sep=DEFAULT_SEPARATOR)

        suffix = SEPARATOR_SUFFIX_DICT.get(DEFAULT_SEPARATOR, ".unknown")
        final_file = output_dir.with_suffix(suffix)

        assert final_file.exists()
        assert final_file.read_text() == ""  # empty
        assert not output_dir.exists()

    def test_write_to_single_dsv_file_custom_separator(self, tmp_path: Path):
        """Test with custom separator not in dict"""
        sep = "|"
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        (output_dir / "part-000.csv").write_text("header\nrowX\n")

        write_to_single_dsv_file(str(output_dir), sep=sep)

        suffix = SEPARATOR_SUFFIX_DICT.get(sep, ".unknown")
        final_file = output_dir.with_suffix(suffix)

        assert final_file.exists()
        content = final_file.read_text()
        assert "rowX" in content
        assert not output_dir.exists()
