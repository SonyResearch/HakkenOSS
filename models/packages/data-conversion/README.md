# Data Conversion package
Library functions for transforming data from one structure or file format to another.


## Testing

To run the unit tests for this project, you can use the following commands:

```bash
python -m unittest discover tests
```

This command will automatically discover and execute all the unit tests located in the "tests" directory.

Alternatively, you can run specific unit tests by specifying the test module:
```bash
python -m unittest tests.test_triples_conversion
```

This command will run only the unit tests defined in the `tests/test_triples_conversion.py` file.

## TODO

This section lists pending tasks that need to be addressed in the package.

- [ ] Implement support for GPU in Torch and PyTorch Geometric.
