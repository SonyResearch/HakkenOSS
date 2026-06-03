/*Hook that handles query submission*/
import { useEffect, useState } from 'react';

import {
  getTripleFromCondition,
  isNestedObjectEqual,
} from '../../features/QueryForm/components/MainForm/utils';
import { useQueryContext } from '../../contexts/QueryContext';
import {
  Condition,
  ConditionType,
  Query,
  Variable,
} from '../../contexts/QueryContext/types';
import { useQueryService } from '../../services/query';
import { Domain } from '../../static/datasets/data-types';
import { editAndShiftConditions } from '../../features/QueryForm/utils';

export const useHandleQueryForm = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const {
    hypotheses,
    constraints,
    setHypotheses,
    setConstraints,
    variables,
    setVariables,
    queryApi,
    query,
    queryMode,
    candidatesNumber,
    setQuery,
    setQueryApi,
    setCandidatesResult,
  } = useQueryContext();

  const { submitQuery } = useQueryService();

  const handleAddNewConditions = (
    newConditions: Record<number, Condition>,
    conditionType: ConditionType,
  ) => {
    const [conditions, setConditions] =
      conditionType === ConditionType.hypotheses
        ? [hypotheses, setHypotheses]
        : [constraints, setConstraints];

    let allConditions: Record<number, Condition> = {};
    if (Object.keys(newConditions).length > 1) {
      allConditions = editAndShiftConditions(conditions, newConditions);
    } else {
      allConditions = { ...conditions, ...newConditions };
    }

    setConditions({
      ...allConditions,
    });
  };

  useEffect(() => {
    if (
      Object.values(hypotheses).length > 0 &&
      Object.values(variables).length > 0
    ) {
      const [, queryAsId] = updateQueryStringApi();
      setQueryApi(queryAsId);
    }
  }, [hypotheses, constraints, variables]);

  const handleAddNewVariable = (label: string, domain: Domain) => {
    console.log('Adding variable', variables, domain, label);
    const currentIndex = 0;
    //const newVariables = { ...variables, [currentIndex]: { label, domain } }; */ //change when we accept more than one variable
    setVariables({ [currentIndex]: { label, domain } });
  };

  const checkExistingVariable = (label: string, domain: Domain) => {
    console.log('Checking...', domain, variables);
    const len = Object.keys(variables)?.length;
    if (len && isNestedObjectEqual(variables[0].domain, domain)) {
      return 'exists';
    } else if (len) {
      return 'not-empty';
    }
    return 'empty';
  };

  const findLabelOfVariable = (name: string) => {
    return Object.values(variables).find(
      (variable) => variable.domain.node_domain === name,
    );
  };
  const updateQueryStringApi = () => {
    const variablesAttribute = Object.values(variables)
      ?.map(
        (variable: Variable) =>
          `{"label": "${variable.label || ''}", "domain": "${variable.domain?.node_domain}"}`,
      )
      .join(',');
    const allConditions = [
      ...Object.values(hypotheses),
      ...Object.values(constraints),
    ];
    const conditionsAttribute = allConditions.map(
      (condition: Condition, index) => {
        const { variable, concept, relation, isSubjectPrediction } =
          getTripleFromCondition(condition);
        const variableFound = findLabelOfVariable(
          variable.domain.replace(/_/g, ' '),
        );
        return {
          asId: `${index ? condition?.addValue : ''} ${condition?.conditionType}(${isSubjectPrediction ? variableFound?.label : concept.id}, ${relation}, ${!isSubjectPrediction ? variableFound?.label : concept.id})`,
          asString: `${index ? condition?.addValue : ''}  ${condition?.conditionType}(${isSubjectPrediction ? variableFound?.label : concept.label}, ${relation}, ${!isSubjectPrediction ? variableFound?.label : concept.label})`,
        };
      },
    );

    const queryAsString = `{"variables":[ ${variablesAttribute} ],
      "formula": "${conditionsAttribute.map((item) => item.asString).join(' ')}"}`;
    const queryAsIds = `{"variables":[ ${variablesAttribute} ],
      "formula": "${conditionsAttribute.map((item) => item.asId).join(' ')}"}`;

    const query = JSON.parse(queryAsString) as Query;
    setQuery(
      `[${Object.values(variables)
        .map(
          (variable: Variable) =>
            `${variable.label} ∈ ${variable.domain.node_domain}`,
        )
        .join(', ')}]: ${query.formula}`,
    );
    setQueryApi(queryAsString);
    return [queryAsString, queryAsIds];
  };

  const createQuery = async () => {
    setLoading(true);
    try {
      const candidates = await submitQuery(
        queryApi,
        query,
        hypotheses,
        constraints,
        candidatesNumber,
        queryMode,
      );
      setCandidatesResult(candidates);
      setLoading(false);
    } catch (error) {
      // TODO: Update once we have centralized error handling
      console.error('createQueryError = ', error);
      setLoading(false);
      throw new Error(
        (error as Error).message ??
          'Something went wrong while processing this query, please try again later',
      );
    }
  };

  return {
    createQuery,
    loading,
    handleAddNewConditions,
    handleAddNewVariable,
    checkExistingVariable,
    updateQueryStringApi,
  };
};
