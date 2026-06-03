CREATE DATABASE contextualization;

\connect contextualization;

CREATE TABLE publication (
    pk integer NOT NULL,
    publication_id character varying(32),
    year integer,
    title text,
    abstract text,
    doi text,
    authors jsonb,
    citations_count integer
);

CREATE TABLE publication_concept (
    pk bigint NOT NULL,
    publication_id character varying(32),
    concept_id character varying(32)
);

ALTER TABLE ONLY publication_concept
    ADD CONSTRAINT publication_concept_pkey PRIMARY KEY (pk);

ALTER TABLE ONLY publication
    ADD CONSTRAINT publication_pkey PRIMARY KEY (pk);

CREATE INDEX publication_concept_publication_id_idx ON publication_concept USING btree (publication_id);

CREATE INDEX publication_concept_concept_id_idx ON publication_concept USING btree (concept_id);

CREATE UNIQUE INDEX publication_publication_id_idx ON publication USING btree (publication_id);

CREATE INDEX publication_year_idx ON publication USING btree (year);

CREATE INDEX publication_citations_count_index ON publication USING btree (citations_count);
