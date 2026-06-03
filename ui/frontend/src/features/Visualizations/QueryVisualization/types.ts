import { ConditionType, InputType } from '../../../contexts/QueryContext/types';
import { LinkType, NodeType } from '../Common/types';

export interface QueryNodeType extends NodeType {
  group: InputType.VARIABLE | InputType.CONCEPT | InputType.RELATION;
  conditionIndex: number | null;
  type: ConditionType;
}

export interface QueryLinkType extends LinkType {
  source: string | QueryNodeType;
  target: string | QueryNodeType;
  conditionIndex: number | null;
  type: ConditionType;
  midX?: number;
  midY?: number;
}
