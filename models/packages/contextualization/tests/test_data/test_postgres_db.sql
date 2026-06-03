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

INSERT INTO publication_concept (pk, publication_id, concept_id)
VALUES
     (0, 'id1', 'concept_id1'),
     (1, 'id2', 'concept_id2'),
     (2, 'id3', 'concept_id3'),
     (3, 'id1', 'concept_id2');

INSERT INTO publication (pk, publication_id, year, title, abstract, doi, authors, citations_count)
VALUES
    (0, 'id1', 2008, 'title1', 'abstract1', 'doi1', '[{"first_name":"AF1","last_name":"AL1"}]', 10),
    (1, 'id2', 2015, 'title2', 'abstract2', 'doi2', '[{"first_name":"AF2","last_name":"AL2"}]', 20),
    (2, 'id3', 2019, 'title3', 'abstract3', 'doi3', '[{"first_name":"AF3","last_name":"AL3"}, {"last_name":"AL4"}]', 15);
