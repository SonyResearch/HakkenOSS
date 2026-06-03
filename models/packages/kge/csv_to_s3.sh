#!/bin/bash

# Configure these variables (or set S3_BUCKET_NAME as an environment variable)
BUCKET_NAME="${S3_BUCKET_NAME:?S3_BUCKET_NAME environment variable is required}"
PREFIX="ds_project_files/ds_data/nodes_selection/aging/inference_files/shortest_path/"
BASE_DIR="experiments/data/hypotheses_clean"

# Find all CSV files and upload them
find "$BASE_DIR" -name "*_spath.csv" | while read file; do
    filename=$(basename "$file")
    echo "Uploading $file to s3://$BUCKET_NAME/$PREFIX$filename"
    aws s3 cp "$file" "s3://$BUCKET_NAME/$PREFIX$filename"
done

echo "Upload complete"