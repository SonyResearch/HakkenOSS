#!/bin/bash

export JAVA_OPTS='-Xms330G -Xmx330G'
neo4j-admin database import full --overwrite-destination --verbose  --delimiter="\t" --array-delimiter="|" --read-buffer-size=900MB --max-off-heap-memory=90% --nodes=/var/lib/neo4j/import/nodes.csv --relationships=/var/lib/neo4j/import/edges.csv neo4j