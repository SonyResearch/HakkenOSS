import {
  getLinksDataFromConditions,
  constructNodesFromData,
  separateConditionsByOR,
  mergeGroups,
  calculateCandidateScore,
} from '../../../src/features/Visualizations/QueryVisualization/utils';
import {
  AddValue,
  InputType,
  ConditionType,
  Condition,
  Query,
} from '../../../src/contexts/QueryContext/types';
import { describe, it, expect, vi } from 'vitest';
import { CandidateResultType } from '../../../src/features/CandidateDashboard/types';
import { QueryNodeType } from '../../../src/features/Visualizations/QueryVisualization/types';

const mockCondition = (
  head: string,
  relation: string,
  tail: string,
  conditionType: ConditionType,
  addValue: AddValue = AddValue.AND,
) => ({
  condition: {
    head: {
      label: 'x',
      domain: head,
      id: '1234',
      isVariable: true,
    },
    relation: relation,
    tail: {
      label: tail,
      domain: 'protein',
      id: '1234',
      isVariable: false,
    },
  },
  conditionType: conditionType,
  addValue,
});

describe('getLinksDataFromConditions', () => {
  describe('data returned from just hypotheses', () => {
    it('should return the links data with relationName property and node names', () => {
      const mockConditionGroup = [
        mockCondition('A', 'relates to', 'B', ConditionType.hypotheses),
        mockCondition('A', 'treats', 'C', ConditionType.hypotheses),
      ];
      const { linksData, nodeNames } = getLinksDataFromConditions(
        mockConditionGroup,
        0,
      );
      expect(linksData).toEqual([
        {
          conditionIndex: 0,
          relationName: 'relates to',
          source: 'A',
          target: '0B # protein',
          type: ConditionType.hypotheses,
        },
        {
          conditionIndex: 1,
          relationName: 'treats',
          source: 'A',
          target: '1C # protein',
          type: ConditionType.hypotheses,
        },
      ]);
      expect(nodeNames).toEqual(new Set(['A', '0B # protein', '1C # protein']));
    });
  });
  describe('data returned from hypotheses and constraints', () => {
    it('should return a numeric condition index for hypotheses and a null one for constraints', () => {
      const mockConditionGroup = [
        mockCondition('A', 'relates to', 'B', ConditionType.hypotheses),
        mockCondition('A', 'relates to', 'C', ConditionType.hypotheses),
        mockCondition('A', 'treats', 'D', ConditionType.constraints),
      ];
      const { linksData, nodeNames } = getLinksDataFromConditions(
        mockConditionGroup,
        0,
      );
      expect(linksData).toEqual([
        {
          conditionIndex: 0,
          relationName: 'relates to',
          source: 'A',
          target: '0B # protein',
          type: ConditionType.hypotheses,
        },
        {
          conditionIndex: 1,
          relationName: 'relates to',
          source: 'A',
          target: '1C # protein',
          type: ConditionType.hypotheses,
        },
        {
          conditionIndex: null,
          relationName: 'treats',
          source: 'A',
          target: '2D # protein',
          type: ConditionType.constraints,
        },
      ]);
      expect(nodeNames).toEqual(
        new Set(['A', '0B # protein', '1C # protein', '2D # protein']),
      );
    });
  });
});

describe('constructNodesGromData', () => {
  const mockLinksData = [
    {
      conditionIndex: 0,
      relationName: 'relates to',
      source: 'A',
      target: '0B # protein',
      type: ConditionType.hypotheses,
    },
    {
      conditionIndex: null,
      relationName: '',
      source: 'A',
      target: 'ASSOCIATE',
      type: ConditionType.constraints,
    },
  ];
  it('should return an array with a concept node and a relation node by merging names and links', () => {
    const mockNodeNames = new Set(['0B # protein', 'ASSOCIATE']);
    const nodes = constructNodesFromData(mockNodeNames, mockLinksData);
    expect(nodes).toEqual([
      {
        id: '0B # protein',
        type: ConditionType.hypotheses,
        conditionIndex: 0,
        group: InputType.CONCEPT,
      },
      {
        id: 'ASSOCIATE',
        type: ConditionType.constraints,
        conditionIndex: null,
        group: InputType.RELATION,
      },
    ]);
  });
  it('should return an empty array if the list of node names is empty', () => {
    const mockNodeNames = new Set([]);
    const nodes = constructNodesFromData(mockNodeNames, mockLinksData);
    expect(nodes).toEqual([]);
  });
});

describe('separateConditionsByOR', () => {
  it('should return an array with two array groups of conditions divided by "OR"', () => {
    const mockConditions: Record<number, Condition> = {
      0: mockCondition('A', 'relates to', 'B', ConditionType.hypotheses),
      1: mockCondition(
        'A',
        'treats',
        'C',
        ConditionType.hypotheses,
        AddValue.OR,
      ),
      2: mockCondition('A', 'inhibits', 'D', ConditionType.hypotheses),
    };
    const mockGroups = separateConditionsByOR(mockConditions);
    expect(mockGroups).toStrictEqual([
      [mockConditions[0]],
      [mockConditions[1], mockConditions[2]],
    ]);
  });
  it('should return an array with only one array containing all the condions', () => {
    const mockConditions: Record<number, Condition> = {
      0: mockCondition('A', 'relates to', 'B', ConditionType.hypotheses),
      1: mockCondition(
        'A',
        'treats',
        'C',
        ConditionType.hypotheses,
        AddValue.AND_NOT,
      ),
      2: mockCondition('A', 'inhibits', 'D', ConditionType.hypotheses),
    };
    const mockGroups = separateConditionsByOR(mockConditions);
    expect(mockGroups).toStrictEqual([
      [mockConditions[0], mockConditions[1], mockConditions[2]],
    ]);
  });
  it('Should return an empty array if no conditions are given', () => {
    const mockConditions: Record<number, Condition> = [];
    const mockGroups = separateConditionsByOR(mockConditions);
    expect(mockGroups).toStrictEqual([]);
  });
});

describe('mergeGroups', () => {
  describe('conditions with more than one group', () => {
    let hypothesesGroup = [
      [mockCondition('A', 'relates to', 'B', ConditionType.hypotheses)],
      [
        mockCondition(
          'A',
          'treats',
          'C',
          ConditionType.hypotheses,
          AddValue.OR,
        ),
        mockCondition('A', 'inhibits', 'D', ConditionType.hypotheses),
      ],
    ];
    let constraintsGroup = [
      [
        mockCondition('A', 'relates to', 'F', ConditionType.constraints),
        mockCondition('A', 'inhibits', 'G', ConditionType.constraints),
      ],
    ];
    it('should merge both hypotheses and constraints groups and return a two arrays with all possible group combinations', () => {
      const mergedGroups = mergeGroups(hypothesesGroup, constraintsGroup);
      expect(mergedGroups).toStrictEqual([
        [...constraintsGroup[0], ...hypothesesGroup[0]],
        [...constraintsGroup[0], ...hypothesesGroup[1]],
      ]);
    });
    it('should return the groups as is if one of the condition types are empty', () => {
      constraintsGroup = [];
      const mergedGroups = mergeGroups(hypothesesGroup, constraintsGroup);
      expect(mergedGroups).toStrictEqual([
        hypothesesGroup[0],
        hypothesesGroup[1],
      ]);
    });
  });
  describe('condition types with only one group', () => {
    let hypothesesGroup = [
      [
        mockCondition('A', 'relates to', 'B', ConditionType.hypotheses),
        mockCondition(
          'A',
          'treats',
          'C',
          ConditionType.hypotheses,
          AddValue.OR,
        ),
        mockCondition('A', 'inhibits', 'D', ConditionType.hypotheses),
      ],
    ];
    let constraintsGroup: Condition[][] = [
      [
        mockCondition('A', 'relates to', 'F', ConditionType.constraints),
        mockCondition('A', 'inhibits', 'G', ConditionType.constraints),
      ],
    ];
    it('should return an array containing an array with all conditions', () => {
      const mergedGroups = mergeGroups(hypothesesGroup, constraintsGroup);
      expect(mergedGroups).toStrictEqual([
        [...constraintsGroup[0], ...hypothesesGroup[0]],
      ]);
    });
  });
});

describe('calculateCandidateScore', () => {
  const mockCandidate: CandidateResultType = {
    variableAssignments: { X: '23423423' },
    queryScore: 9.99,
    conditionsScores: {
      0: 2,
      1: 3,
      2: 4,
    },
    name: 'ABC',
    domain: 'DISEASE',
  };
  const node1: QueryNodeType = {
    id: 'ABC',
    group: InputType.VARIABLE,
    conditionIndex: 0,
    type: ConditionType.hypotheses,
  };
  let node2: QueryNodeType = {
    id: 'BCD',
    group: InputType.RELATION,
    conditionIndex: 1,
    type: ConditionType.hypotheses,
  };
  const node3: QueryNodeType = {
    id: 'CDF',
    group: InputType.RELATION,
    conditionIndex: 2,
    type: ConditionType.hypotheses,
  };
  it('Should return the product of condition scores for all nodes', () => {
    const score = calculateCandidateScore(mockCandidate, [node1, node2, node3]);
    expect(score).toBe(24);
  });
  it('Should exclude the score from the calculation if no node contains that condition index', () => {
    node2 = { ...node2, conditionIndex: null };
    const score = calculateCandidateScore(mockCandidate, [node1, node2, node3]);
    expect(score).toBe(8);
  });
});
