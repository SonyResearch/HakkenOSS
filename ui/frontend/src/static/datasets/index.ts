import { domains as domainsPubtator } from './pubtator-data/domains';
import { domains as domainsDs } from './ds-data/domains';
import { relations as relationsPubtator } from './pubtator-data/relations';
import { relations as relationsDs } from './ds-data/relations';
import { defaultConfig } from '../../configI';

export type DataSets = 'ds' | 'pubtator';

export interface Domain {
  node_domain: string;
  node_domain_id: string;
}

export interface Relation {
  relation_type: string;
  relation_id: string;
  object_domain: string;
  subject_domain: string;
}

export interface SimpleConcept {
  id: string;
  domain?: string;
}

export interface Concept extends SimpleConcept {
  name: string;
}

export type Concepts = Record<string, Concept>;
export type Relations = Record<string, Relation>;
export type Domains = Record<string, Domain>;

export const datasets = {
  ds: {
    domains: domainsDs as Record<string, Domain>,
    relations: relationsDs as Record<string, Relation>,
  },
  pubtator: {
    domains: domainsPubtator as Record<string, Domain>,
    relations: relationsPubtator as Record<string, Relation>,
  },
};

export const { domains, relations } = datasets[defaultConfig.dataSet];
