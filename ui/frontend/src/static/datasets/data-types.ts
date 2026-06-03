import { Concept } from '.';

export interface Domain {
  node_domain: string;
  node_domain_id: string;
}

export interface Relation {
  relation_type: string;
  relation_id: string;
}

export interface ConditionedRelation extends Relation {
  object_domain: string;
  subject_domain: string;
}

export interface ConceptNode {
  node_id: string;
  node_domain: string;
  node_id_raw: string;
  node_name: string;
}

export interface ConstraintsFilteringResponse {
  variable: 'X' | 'R' | 'Y';
  type: 'relation' | 'concept';
  values: ConstraintRelation[] | ConstraintConcept[];
}

export type ConstraintRelation = {
  label: string;
  identifier: string;
};

export type ConstraintConcept = {
  label: string;
  identifier: string;
  domain_identifier?: string;
};

export type Concepts = Record<string, Concept>;
export type Relations = Record<string, Relation>;
export type Domains = Record<string, Domain>;
export type ConditionedRelations = Record<string, ConditionedRelation>;
