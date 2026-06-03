#!/bin/bash

# Define versions
HADOOP_AWS_VERSION=3.3.4
AWS_SDK_VERSION=1.12.375

# Define download URLs
HADOOP_AWS_URL="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar"
AWS_SDK_URL="https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

# Define destination folder (e.g. $SPARK_HOME/jars)
DEST_DIR="$SPARK_HOME/jars"

mkdir -p "$DEST_DIR"

# Download jars
curl -L -o "$DEST_DIR/hadoop-aws-${HADOOP_AWS_VERSION}.jar" "$HADOOP_AWS_URL"
curl -L -o "$DEST_DIR/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" "$AWS_SDK_URL"

echo "Downloaded hadoop-aws and aws-java-sdk-bundle jars to $DEST_DIR"
