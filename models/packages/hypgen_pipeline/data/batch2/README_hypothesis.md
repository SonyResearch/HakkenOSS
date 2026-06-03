# Overwiew of the Hypothesis File named `hypothesis.tsv`
First of all the file is a tsv, so it is tab (`\t`) separated as commas might be present in the
entities names. In the files the entities are referred to as `nodes`.

## Loading instructions
The easiest way to load the data is with python and the library `pandas`. The code is the following:
```bash
import pandas as pd
import ast

df = pd.read_csv("hypothesis.tsv", delimiter="\t")
df['node_pair'] = df['node_pair'].apply(ast.literal_eval)
df['node_pair_ocids'] = df['node_pair_ocids'].apply(ast.literal_eval)
df['recency_median'] = df['recency_median'].apply(ast.literal_eval)
df['recency_mode'] = df['recency_mode'].apply(ast.literal_eval)
df['papers_count'] = df['papers_count'].apply(ast.literal_eval)
```

The conversions perfomed with the library `ast` are necessary to have lists instead of
strings. We also have a version of this file as `pickle` in case you find it more convenient.

## Overview of the data
Below a throughtful description of each column in the datafile:
* `hypothesis_idx`: This is an internal id to be able to trace the hypothesis.
* `node_pair`: The subject and object of the hypothesis (preferred name). Recorded as a list of two elements.
* `node_pair_ocids`: The subject and object of the hypothesis expressed as OCID (DS ids). Recorded as list of two elements.
* `predicted_relation`: The relation type predicted between the two nodes. Note two things:
    1. The formatting of the relation name was changed for easier processing (e.g. has underscores `_`).
    2. The prediction IT IS NOT DIRECTIONAL because the model is not directional, so subject and object can be interchanged.
* `confidence_score`: The confidence score provided by the model for the given hypothesis. It is not a probability, but can be used for ranking the hypothesis.
* `recency_mode`: The mode year of the subject and the object respectively. Recorded as list of two elements.
* `recency_median`: The median year of the subject and the object respectively. Recorded as list of two elements.
* `papers_count`: The unique number of publications citing the subject and the object respectively. Recorded as list of two elements.
* `shortest_path_length`: The shortest path length between the subject and the object in the original graph. E.g. if the two nodes (entities)
    were already connected by a relation in the original graph, the shortest_path_length would be 3 (includes the extrema). If they were connected through another node, the shortest path length would be 5 (includes extrema, two hops). 
