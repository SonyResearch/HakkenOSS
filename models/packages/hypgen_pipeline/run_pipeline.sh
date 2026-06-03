# Script to update the pipeline with new files
BATCH=2
echo Pipeline for batch number: ${BATCH}

# STEP 1: Fetch Raw Hypothesis Predictions from s3
# Set S3_BUCKET_NAME as an environment variable before running this script
S3_BUCKET_NAME="${S3_BUCKET_NAME:?S3_BUCKET_NAME environment variable is required}"
inputfile=s3://${S3_BUCKET_NAME}/ds_project_files/ds_data/batch${BATCH}/batch${BATCH}_hypothesis/c813a962_inference_output.csv
outputfile=data/batch${BATCH}/raw/inference_output.csv

aws s3 cp ${inputfile} ${outputfile}
echo Fetched data from ${inputfile} to : ${outputfile}

# STEP 2: Reproduce pipeline. Make sure the batch number is the same as above in params.yaml
dvc repro

# STEP 3: Put on s3 the final list after the pipeline
finalfile_s3=s3://${S3_BUCKET_NAME}/ds_project_files/ds_data/batch${BATCH}/batch${BATCH}_hypothesis_postprocess/hypothesis.pkl
aws s3 cp data/batch${BATCH}/finalize/hypothesis.pkl ${finalfile_s3}
echo Synced data to ${finalfile_s3}