import {
  AddValue,
  Condition,
  ConditionType,
  Variable,
} from '../contexts/QueryContext/types';

export type Conditions = Record<number, Condition>;
export type Variables = Record<number, Variable>;

export type useCases = {
  title: string;
  explanation: string;
  variables: Variables;
  hypotheses: Conditions;
  constraints?: Conditions;
};

export const useCases: Record<string, useCases[]> = {
  ds: [
    {
      title: 'Drug Discovery',
      explanation:
        'identify molecules that can inhibit the progression of glioblastoma',
      variables: {
        0: {
          label: 'X',
          domain: { node_domain_id: 'id108020003292', node_domain: 'protein' },
        },
      },
      hypotheses: {
        0: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id208000005017',
              domain: 'disease',
              isVariable: false,
              label: 'glioblastoma',
            },
            relation: 'Attenuates',
          },
          conditionType: ConditionType.hypotheses,
          addValue: AddValue.AND,
        },
      },
      constraints: {
        0: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id102100006130',
              domain: 'human_genes',
              isVariable: false,
              label: 'CDK6',
            },
            relation: 'Binds to',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
        1: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id101700012265',
              domain: 'protein',
              isVariable: false,
              label: 'EGFR',
            },
            relation: 'Decreases expression of',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
      },
    },
    {
      title: 'Target selection',
      explanation:
        'Find proteins crucial to the treatment of metastatic melanoma',
      variables: {
        0: {
          label: 'X',
          domain: { node_domain_id: 'id108020003292', node_domain: 'protein' },
        },
      },
      hypotheses: {
        0: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id208000021039',
              domain: 'disease',
              isVariable: false,
              label: 'metastatic melanoma',
            },
            relation: 'Treats',
          },
          conditionType: ConditionType.hypotheses,
          addValue: AddValue.AND,
        },
      },
      constraints: {
        0: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id102100019764',
              domain: 'human_genes',
              isVariable: false,
              label: 'PDCD1',
            },
            relation: 'Affects expression of',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
        1: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id102100004328',
              domain: 'human_genes',
              isVariable: false,
              label: 'BRAF',
            },
            relation: 'Increases expression of',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
      },
    },
    {
      title: 'Genetic Interactions',
      explanation: 'Identify genetic diseases responsive to PARP inhibitors',
      variables: {
        0: {
          label: 'X',
          domain: { node_domain_id: 'id201000010099', node_domain: 'disease' },
        },
      },
      hypotheses: {
        0: {
          condition: {
            head: {
              id: 'id239000012661',
              domain: 'substance',
              isVariable: false,
              label: 'PARP inhibitor',
            },
            tail: {
              id: 'id201000010099',
              domain: 'disease',
              isVariable: true,
              label: 'X',
            },
            relation: 'Treats',
          },
          conditionType: ConditionType.hypotheses,
          addValue: AddValue.AND,
        },
      },
      constraints: {
        0: {
          condition: {
            head: {
              id: 'id206010071980',
              domain: 'disease',
              isVariable: false,
              label: 'BRCA1 gene mutation',
            },
            tail: {
              id: 'id201000010099',
              domain: 'disease',
              isVariable: true,
              label: 'X',
            },
            relation: 'Relates to',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
        1: {
          condition: {
            head: {
              id: 'id201000010099',
              domain: 'disease',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id102000000872',
              domain: 'human_genes',
              isVariable: false,
              label: 'ATM',
            },
            relation: 'Is a target for',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
      },
    },
    {
      title: 'Drug Repurposing',
      explanation:
        'discover new protein targets for the antiviral drug, Remibrutinib',
      variables: {
        0: {
          label: 'X',
          domain: { node_domain_id: 'id108020003292', node_domain: 'protein' },
        },
      },
      hypotheses: {
        0: {
          condition: {
            head: {
              id: 'id190069416048',
              domain: 'chemistry',
              isVariable: false,
              label: 'remibrutinib',
            },
            tail: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            relation: 'Binds to',
          },
          conditionType: ConditionType.hypotheses,
          addValue: AddValue.AND,
        },
      },
      constraints: {
        0: {
          condition: {
            head: {
              id: 'id108020003292',
              domain: 'protein',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id101700034277',
              domain: 'protein',
              isVariable: false,
              label: 'JAK1',
            },
            relation: 'Relates to',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
      },
    },
    {
      title: 'Biomarker Discovery',
      explanation:
        'Find effective biomarkers for early-stage Alzheimer’s disease',
      variables: {
        0: {
          label: 'X',
          domain: {
            node_domain_id: 'id400500000045',
            node_domain: 'biomarker',
          },
        },
      },
      hypotheses: {
        0: {
          condition: {
            head: {
              id: 'id400500000045',
              domain: 'biomarker',
              isVariable: true,
              label: 'X',
            },
            tail: {
              id: 'id208000020814',
              domain: 'disease',
              isVariable: false,
              label: 'Prodromal Alzheimer disease',
            },
            relation: 'Of',
          },
          conditionType: ConditionType.hypotheses,
          addValue: AddValue.AND,
        },
      },
      constraints: {
        0: {
          condition: {
            head: {
              id: 'id102000009265',
              domain: 'human_genes',
              isVariable: false,
              label: 'Cerebrospinal Fluid Proteins',
            },
            tail: {
              id: 'id400500000045',
              domain: 'biomarker',
              isVariable: true,
              label: 'X',
            },
            relation: 'Is a',
          },
          conditionType: ConditionType.constraints,
          addValue: AddValue.AND,
        },
      },
    },
  ],
};
