# Neo4j Data Loader

This project provides a simple step-by-step guide to loading the **Digital Sciene** data into a `neo4j` instance.
It uses `spark` and `neo4j-admin database import` to guarantee fast processing of big data. 
This project is a supplementary step-by-step guide for loading data into a `neo4j` instance.
Please familiarize yourself with the prerequisites below before working with `neo4j`. 

## Prerequisits 
* Access to the data S3 bucket
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
* [Docker](https://docs.docker.com/engine/install/ubuntu/)
* [neo4j community edition](https://neo4j.com/docs/operations-manual/current/installation/linux/debian/#debian-installation) 
* Clone this repository into a directory your current user owns. 

## How to Use
You will need extensive computing resources. Make sure to have enough computing resources available (e.g., `EC2` instance `r7iz.12xlarge` with `1TB` volume). 
Commands assume the `data-loader/` directory as the working directory, except stated otherwise. 

### Data Preprocessing

You want to start by getting the raw data.
1. Download `relations_cleaned.csv` from the data S3 bucket into `raw_data/`
    ``` 
    aws s3 cp s3://<YOUR_BUCKET>/relations_cleaned.csv raw_data/relations_cleaned.csv
    ```
Next, you will set up the spark preprocessing environment. 

2. Pull the [apache/spark-py image](https://hub.docker.com/r/apache/spark-py/tags).
    ```
    docker pull apache/spark-py:v3.4.0
    ```
3. Run the `pyspark docker container`.
    ```
    docker run --name pyspark -u root -v {InsertCorrectPathtoRepo}/data-loader/raw_data:/import -v {InsertCorrectPathtoRepo}/data-loader/src:/code -it apache/spark-py bash
    ```
    :exclamation: :exclamation: :exclamation: Change `{InsertCorrectPathtoRepo}` to the absolute path of this repository on your host machine (e.g. `/home/ubuntu/`) :exclamation: :exclamation: :exclamation:
4. Change your working directory inside the docker container to root and execute the preprocessing script there.
    ```
    cd 
    /opt/spark/bin/spark-submit --conf spark.executor.memory=7g --conf spark.driver.memory=200g --conf spark.driver.maxResultSize=150g --conf spark.sql.objectHashAggregate.sortBased.fallbackThreshold=3000 /code/preprocessing.py
    ```
    You may adapt the config settings. They are not heavily optimized yet. The script takes about 30 minutes to execute on the recommended hardware. While running the script, it is recommended not to issue any other computing-intensive tasks on the same instance. The script outputs two folders, `raw_data/edges` (with columns: [":TYPE" - str with relation type, "ocid_subject" - ":START_ID" - str with ocid of subject, ":END_ID" - str with ocid of object, "store_source_date" - date, "src_rep" - str, "val_doc_id" - int, "srcsection"- str, "store_source_id" - int, "store_source_link" - str, "store_srcsent" - str, "store_source_db" - str
]) and `raw_data/nodes` (with columns: ["ocid:ID" - stirng with ocid, ":LABEL" - List of str with labels(types/classes), "concept" - str with name]).  

5. It is best to stop and delete the pyspark docker container once the script is finished.
   ```
   docker stop pyspark
   docker rm pyspark
   ```
Congratulations :partying_face:, you successfully preprocessed the data. 

### Import Preprocessed Data into neo4j

Ensure that you have successfully executed the data preprocessing steps. 

6. Move and rename the preprocessed data to the `neo4j` import directory. 
    ```
    sudo mv raw_data/nodes/part-{InsertCorrectFileNameHere}.csv /var/lib/neo4j/import/nodes.csv
    sudo mv raw_data/edges/part-{InsertCorrectFileNameHere}.csv /var/lib/neo4j/import/edges.csv
    ```
    :exclamation: :exclamation: :exclamation: Change `{InsertCorrectFileNameHere}` to the individual filename assigned by pyspark :exclamation: :exclamation: :exclamation:

7. Finally, you can import the data into the `neo4j` database.
    ```
    sudo bash src/import.sh
    ```
    You may adapt the config settings. They are not heavily optimized yet. The script takes about 30 minutes to execute on the recommended hardware. While running the script, it is recommended not to issue any other computing-intensive tasks on the same instance.

Congratulations :partying_face:, you are done. 	:tada: 	:tada: 	:tada: 

### Starting the neo4j Instance
Start the `neo4j` instance with the following command: 
    
```
docker compose up
```

### Interacting with the neo4j Instance
You have several ways to interact with the `neo4j` instance. 
For most ways you have to forward the interface to your client. 

```
ssh -N -L 7474:localhost:7474 -L 7687:localhost:7687 {hostName}
```

:exclamation: :exclamation: :exclamation: Change `{hostName}` to the host name of the `EC2` instance with the `neo4j` database running :exclamation: :exclamation: :exclamation:

1. [Browser Interface](https://neo4j.com/developer/neo4j-browser/) 
2. [neo4j Python Driver](https://neo4j.com/developer/python/)
3. [Cypher Shell](https://neo4j.com/docs/operations-manual/current/tools/cypher-shell/) (only useable inside the docker container!)

### Stopping the neo4j Instance
Stop the `neo4j` instance with the following command: 
```
docker compose down
```

### Usefull Commands
* `watch df -h` - watch disk space while spark does its thing. 
* `/opt/spark/bin/pyspark --conf spark.executor.memory=7g --conf spark.driver.memory=200g --conf spark.driver.maxResultSize=150g --conf spark.sql.objectHashAggregate.sortBased.fallbackThreshold=300` - open pyspark shell inside the pyspark docker container

### (Optional) APOC Plugin Installation Guide for Neo4j

APOC is an important plugin for Neo4j that facilitates batching and parallelizing large queries. This guide will help you install and use the APOC plugin seamlessly.

#### Installation

1. **Default Installation**: APOC comes pre-installed with your Neo4j installation, located in the labs folder.

2. **Moving APOC to Plugins Folder**:
   - To utilize APOC, you need to move it from the labs folder to the plugins folder. You can do this by executing the following command in your terminal:
     ```
     sudo mv /var/lib/neo4j/labs/{insert-your-apoc-version.jar} /var/lib/neo4j/plugins/
     ```
   - Replace `{insert-your-apoc-version.jar}` with the appropriate APOC version you have in your installation.

3. **Restart Neo4j instance** to activate your new plugin!

With APOC activated, you can now use its functions and procedures in your Neo4j queries to efficiently handle large datasets and parallelize operations.

That's it! You've successfully installed and activated the APOC plugin for Neo4j.


### (Optional) Adding NeoSemantics to Your Neo4j Installation

NeoSemantics is a powerful Neo4j plugin designed to seamlessly integrate Semantic Web technologies with Neo4j graph databases. By incorporating NeoSemantics into your Neo4j environment, you unlock a range of functionalities, including RDF import/export, SPARQL endpoint support, ontology management, reasoning capabilities, and more.

For detailed information and usage guidelines, refer to the [NeoSemantics user manual](https://neo4j.com/labs/neosemantics/4.0/).

To add NeoSemantics to your Neo4j installation, follow these steps:

1. **Download the Plugin:** Start by downloading the NeoSemantics plugin into your Neo4j `plugins` folder. Visit the [NeoSemantics releases page](https://github.com/jbarrasa/neosemantics/releases) to find the latest release. Copy the link to the JAR file for your preferred release and use `wget` to download it into your plugins folder:
   
    ```
    wget -P /var/lib/neo4j/plugins {link_to_jar_release}
    ```


2. **Configure RDF Endpoint:** Add the RDF endpoint configuration to your Neo4j installation by running the following command:

    ```
    echo 'server.unmanaged_extension_classes=n10s.endpoint=/rdf' | sudo tee -a /var/lib/neo4j/conf/neo4j.conf
    ```


3. **Restart Neo4j Instance:** **If your Neo4j instance is currently running**, restart it to apply the changes:

    ```
    docker compose restart
    ```


4. **Verify Installation:** Confirm that the installation was successful by pinging the RDF endpoint. In the Neo4j Browser Interface, execute the following Cypher query:

    ```
    :GET http://localhost:7474/rdf/ping
    ```

If the installation was successful, you should receive a response indicating the endpoint is accessible.

You are now ready to use NeoSemantics!! 

## What this Project is not
* It does not explain the neo4j-admin import tool. Consult [this resource](https://neo4j.com/docs/operations-manual/current/tutorial/neo4j-admin-import/) for a comprehensive guide on the neo4j-admin import tool. 
* It does not propose optimal parameters for pyspark, the neo4j database, and the neo4j-admin import tool. You may change them. 
* It does not explain the chosen data model. Change the data model by adapting the `src/preprocessing.py` script to your needs. 

## How to get Help
If you encounter any issues please open a GitHub issue.

## Acknowledgements
This approach is based on the neo4j-admin bulk import guidelines for loading large datasets into neo4j. 
