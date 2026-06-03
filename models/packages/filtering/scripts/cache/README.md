# Degree Cache Generation

**This contains code for old data structure. Will be update afterwards.**

Generate a cache file of node degrees for performance improvement.
Basically it computes degrees by year, but plus considering train-validation-test split.

- Train data: Before 2021/12/24 (Until timestamp `1640300400`)
- Validation data: Between 2021/12/24 and 2022/05/12 (Until timestamp `1652306400`)
- Test data: After 2022/05/12

Note that those cutoff timestamp values are also defined in `filtering.core.values.data_split`.
