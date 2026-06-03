# PostgreSQL Data Insertion

This directory contains scripts related to inserting data into PostgreSQL.

Currently, the following tables are constructed:

- Publication-concept
- Publication text

## Usage

- Copy `.env.sample` file to `.env` and change values as you wish
- Copy `postgresql.conf.sample` file to `postgresql.conf` and change values as you wish
  - The ones changed in the sample file is `shared_buffers` and `max_wal_size`
- Run `sudo docker compose up -d` (or without `sudo` on rootless docker)
  - This will create database named and tables automatically by running queries in `schema.sql`
