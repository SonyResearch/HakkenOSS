import * as d3 from 'd3';
import { ConditionType } from '../../../contexts/QueryContext/types';
import { QueryLinkType, QueryNodeType } from './types';

const CONSTANTS = {
  RELATION_LINK_LENGTH: 40,
  CONSTRAINTS_LINK_LENGTH: 180,
  HYPOTHESES_LINK_LENGTH: 200,
  REPULSION_STRENGTH: -1500,
  CONSTRAINTS_BOTTOM_STRENGTH: 500,
  HYPOTHESES_BOTTOM_STRENGTH: 100,
  CONSTRAINTS_RIGHT_STRENGHT: 0,
  HYPOTHESES_RIGHT_STRENGHT: 800,
};

export const createSimulation = (
  nodes: QueryNodeType[],
  linksData: QueryLinkType[],
) => {
  const simulation = d3
    .forceSimulation<QueryNodeType>(nodes)
    .force(
      'link',
      d3
        .forceLink<QueryNodeType, QueryLinkType>(linksData)
        .id((link) => link.id)
        .distance((link) =>
          !link.relationName
            ? CONSTANTS.RELATION_LINK_LENGTH
            : link.type === ConditionType.constraints
              ? CONSTANTS.CONSTRAINTS_LINK_LENGTH
              : CONSTANTS.HYPOTHESES_LINK_LENGTH,
        ),
    )
    .force(
      'charge',
      d3.forceManyBody<QueryNodeType>().strength(CONSTANTS.REPULSION_STRENGTH),
    ) //repulsion between nodes
    .force(
      'down',
      d3.forceY((link) =>
        link.type === ConditionType.constraints
          ? CONSTANTS.CONSTRAINTS_BOTTOM_STRENGTH
          : CONSTANTS.HYPOTHESES_BOTTOM_STRENGTH,
      ),
    )
    .force(
      'left',
      d3.forceX(
        (
          link, //hypotheses to the right and constraints to the left
        ) =>
          link.type === ConditionType.constraints
            ? CONSTANTS.CONSTRAINTS_RIGHT_STRENGHT
            : CONSTANTS.HYPOTHESES_RIGHT_STRENGHT,
      ),
    );
  return simulation;
};
