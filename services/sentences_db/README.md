# PostgreSQL Service Setup
The service/sentencesdb directory contains the necessary configuration files to set up and run a PostgreSQL database service using Docker. This setup ensures a consistent and reproducible environment for your database needs.

## Getting Started

Follow these steps to quickly set up and run the PostgreSQL service:

1. **Clone this repository** (if applicable):
    ```sh
   git clone <REPO_URL>
   cd services/sentences_db
   ```
2. **Create a .env file** in the service/sentences_db directory with the following content. Make sure to replace yourpassword with your own password:
    ```
    POSTGRES_PASSWORD=yourpassword
    PGPORT=8080
    PGDATA=/var/lib/postgresql/data/
    ```
3. **Download the PostgreSQL dump file**
    ```sh
    aws s3 cp s3://<YOUR_BUCKET>/sentences_db/latest/pg_data.tar.gz ./data/pg_data.tar.gz
    cd ./data
    tar --no-same-owner -xvzf pg_data.tar.gz
    ```

4. **Start the PostgreSQL service** using Docker Compose:
    ```sh
    docker-compose up -d
    ```

5. **Verify your DB schema**. If the database sentence_db exists, it is running correctly. :
    ```sh
    docker ps  # Get container id
    docker exec -it [container-id] psql -h localhost -p [PGPORT] -U postgres  # Log-in to your postgresql
    postgres=# \l
                                  List of databases
        Name     |  Owner   | Encoding | Collate |  Ctype  |   Access privileges   
    -------------+----------+----------+---------+---------+-----------------------
    postgres    | postgres | UTF8     | C.UTF-8 | C.UTF-8 | 
    sentence_db | postgres | UTF8     | C.UTF-8 | C.UTF-8 | 
    template0   | postgres | UTF8     | C.UTF-8 | C.UTF-8 | =c/postgres          +
                |          |          |         |         | postgres=CTc/postgres
    template1   | postgres | UTF8     | C.UTF-8 | C.UTF-8 | =c/postgres          +
                |          |          |         |         | postgres=CTc/postgres
    (4 rows)
    ```


## Configuration Files
+ **docker-compose.yml**: Defines the PostgreSQL service configuration, including environment variables, volumes, and ports.
+ **.env**: Contains environment-specific variables such as the database password, port, and data directory.

## Environment Variables
The following environment variables should be defined in the .env file:
+ POSTGRES_PASSWORD: The password for the PostgreSQL database user.
+ PGPORT: The port on which PostgreSQL will run.
+ PGDATA: The directory where PostgreSQL data will be stored.

