import { Triple } from '../../../contexts/QueryContext/types';
import { LinkType } from '../Common/types';

export interface ExplanationTriple extends Triple {
  head: string;
  relation: string;
  tail: string;
  paths?: number[];
  hop: number;
}

export type NodeGroups = 'head' | 'tail' | 'middle' | 'relation';

export interface ExplanationNodeType extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  hop: number;
  group: NodeGroups;
  fx?: number | null;
  fy?: number | null;
}

export interface ExplanationLinkType extends LinkType {
  source: string | ExplanationNodeType;
  target: string | ExplanationNodeType;
  relationName: string;
}
