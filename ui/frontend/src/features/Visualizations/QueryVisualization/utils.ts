import { CandidateResultType } from '../../CandidateDashboard/types';
import {
  AddValue,
  Condition,
  ConditionType,
  InputType,
  TripleEntity,
} from '../../../contexts/QueryContext/types';
import { domains, relations } from '../../../static/datasets';
import { QueryLinkType, QueryNodeType } from './types';

export const relationsArray = [
  ...new Set(
    Object.values(relations).map((relation) => relation.relation_type),
  ),
];
const domainsArray = Object.values(domains).map((domain) => domain.node_domain);

export const calculateCandidateScore = (
  //maybe we don't need it
  candidate: CandidateResultType,
  groupNodes: QueryNodeType[],
) => {
  const conditionIndexes = [
    ...new Set(
      groupNodes
        .map((node) => node.conditionIndex)
        .filter((index) => typeof index === 'number'),
    ),
  ];
  const product = conditionIndexes
    .map((index) => Number(candidate.conditionsScores[index ?? 0]))
    .reduce((acc, value) => acc * value, 1);
  return product;
};

export const separateConditionsByOR = (
  conditions: Record<number, Condition>,
) => {
  const conditionsGroups = [];
  let currentGroup: Condition[] = [];
  for (const condition of Object.values(conditions)) {
    if (condition.addValue === 'OR') {
      if (currentGroup.length > 0) {
        conditionsGroups.push(currentGroup);
      }
      currentGroup = [condition as Condition];
    } else {
      currentGroup.push(condition as Condition);
    }
  }
  if (currentGroup.length > 0) {
    conditionsGroups.push(currentGroup);
  }
  return conditionsGroups;
};

export const getConditionGroups = (
  //get groups separated by or operator at 'root level'
  hypotheses: Record<number, Condition>,
  constraints: Record<number, Condition>,
) => {
  const hypothesesGroups: Condition[][] = separateConditionsByOR(hypotheses);
  const constraintsGroups: Condition[][] = separateConditionsByOR(constraints);

  const mergedGroups = mergeGroups(hypothesesGroups, constraintsGroups); //merge to get all combinations
  return [...mergedGroups];
};

export const mergeGroups = (
  hypothesesGroups: Condition[][],
  constraintsGroups: Condition[][],
) => {
  if (hypothesesGroups.length > 0 && constraintsGroups.length > 0) {
    return hypothesesGroups.flatMap((hypothesesGroup) =>
      constraintsGroups.map((constraintsGroup) => [
        ...constraintsGroup,
        ...hypothesesGroup,
      ]),
    );
  } else if (hypothesesGroups.length > 0) {
    return [...hypothesesGroups];
  } else if (constraintsGroups.length > 0) {
    return [...constraintsGroups];
  } else {
    return [];
  }
};

const formatConditionLabel = (index: number, concept: TripleEntity) => {
  return `${index + concept.label} # ${concept.domain}`;
};

export const getLinksDataFromConditions = (
  conditionGroup: Condition[],
  startingConditionIndex: number,
) => {
  //get edges data for d3
  let conditionIndex = startingConditionIndex;
  const linksData: QueryLinkType[] = Object.values(conditionGroup).flatMap(
    (condition, index) => [
      {
        target: condition.condition.tail.isVariable
          ? condition.condition.tail.domain
          : formatConditionLabel(index, condition.condition.tail),
        source: condition.condition.head.isVariable
          ? condition.condition.head.domain
          : formatConditionLabel(index, condition.condition.head),
        conditionIndex:
          condition.conditionType === ConditionType.constraints
            ? null
            : conditionIndex++,
        type: condition.conditionType,
        relationName: `${condition.addValue === AddValue.AND_NOT ? 'NOT ' : ''}${condition.condition.relation}`,
      },
    ],
  );

  const nodeNames: Set<string> = new Set();
  linksData.forEach((node) => {
    if (typeof node.source === 'string') {
      nodeNames.add(node.source);
    }
    if (typeof node.target === 'string') {
      nodeNames.add(node.target);
    }
  });
  return { linksData, nodeNames };
};

export const constructNodesFromData = (
  nodeNames: Set<string>,
  linksData: QueryLinkType[],
) => {
  const nodes: QueryNodeType[] = Array.from(nodeNames).map((name) => ({
    id: name,
    type: (() => {
      const matchingNode = linksData.find(
        (node) => node.target === name || node.source === name,
      );
      return matchingNode ? matchingNode.type : ConditionType.hypotheses;
    })(),
    conditionIndex: (() => {
      //get condition index so later we know which node goes with each candidate score
      const matchingNode = linksData.find(
        (node) => node.target === name || node.source === name,
      );
      return matchingNode?.conditionIndex ?? null;
    })(),
    group: domainsArray.includes(name)
      ? InputType.VARIABLE
      : relationsArray.includes(
            name
              .replace(/\b(NOT|OR)\b/g, '')
              .replace(/\s+/g, ' ')
              .trim(),
          )
        ? InputType.RELATION
        : InputType.CONCEPT,
  }));
  return nodes;
};
