/*Hook to access all possible options for a query in the form depending on what is already selected, which will be fetched from Data API*/

import React, { useEffect, useState } from 'react';
import {
  ConditionType,
  PredictionType,
} from '../../contexts/QueryContext/types';
import {
  filterOptionsOfConstraints,
  filterOptionsOfDomains,
  filterOptionsOfRelations,
  getOptionsFromConcepts,
  updateConcepts,
} from '../../features/QueryForm/components/MainForm/utils';
import {
  FormAction,
  QueryFormType,
} from '../../contexts/QueryFormContext/types';
import { appConfig } from '../../config';

export const useFilterInputOptions = (
  form: QueryFormType,
  variableDomain: string,
  formType: ConditionType,
  dispatch: React.Dispatch<FormAction>,
) => {
  const [loadingRelations, setLoadingRelations] = useState(false);
  const [loadingConcepts, setLoadingConcepts] = useState(false);
  const [possibleVariableDomains, setPossibleVariableDomains] = useState<
    string[]
  >([]);
  const [possibleConceptDomains, setPossibleConceptDomains] = useState<
    string[]
  >([]);
  const [possibleRelations, setPossibleRelations] = useState<string[]>([]);
  const [possibleConceptNames, setPossibleConceptNames] = useState<string[]>(
    [],
  );

  useEffect(() => {
    const runFilteringLogic = async () => {
      setLoadingRelations(true);
      setLoadingConcepts(true);
      try {
        await loadVariableDomains();
        if (
          formType === ConditionType.hypotheses ||
          !form.tailValue[0] ||
          appConfig.queryDemo
        ) {
          await loadRelations();
        } else {
          await loadRelationConstraintOptions();
        }
        setLoadingRelations(false);

        if (
          formType === ConditionType.hypotheses ||
          !form.selectedConceptDomain ||
          appConfig.queryDemo
        ) {
          await loadConceptDomainsAndConcepts();
        } else if (!form.tailValue[0]) {
          await loadConceptConstraintOptions();
        }
      } catch (err) {
        console.error('Unexpected error in filtering logic:', err);
        dispatchError('Unexpected error while loading filter options');
      } finally {
        setLoadingRelations(false);
        setLoadingConcepts(false);
      }
    };
    const loadVariableDomains = async () => {
      try {
        const varDomains = await filterOptionsOfDomains(
          form.selectedConceptDomain,
          form.selectedRelation,
          form.predictionType === PredictionType.SUBJECT,
        );
        setPossibleVariableDomains(varDomains);
      } catch (err) {
        console.error('Failed to fetch variable domains', err);
        dispatchError('Failed to update variable domains');
      }
    };

    const loadRelations = async () => {
      try {
        const filteredRelations = await filterOptionsOfRelations(
          variableDomain,
          form.selectedConceptDomain,
          form.predictionType,
        );
        setPossibleRelations(filteredRelations);
      } catch (err) {
        console.error('Failed to fetch relations', err);
        dispatchError('Failed to update relations');
      }
    };

    const loadConceptDomainsAndConcepts = async () => {
      try {
        const conceptDomains = await filterOptionsOfDomains(
          variableDomain,
          form.selectedRelation,
          form.predictionType !== PredictionType.SUBJECT,
        );
        setPossibleConceptDomains(conceptDomains);
      } catch (err) {
        console.error('Failed to fetch concept domains', err);
        dispatchError('Failed to update concept domains');
      }

      if (form.selectedConceptDomain && !form.tailValue[0]) {
        try {
          const filteredConcepts = await updateConcepts(
            form.selectedConceptDomain,
          );
          setPossibleConceptNames(getOptionsFromConcepts(filteredConcepts));
        } catch (err) {
          console.error('Failed to update concepts', err);
          dispatchError('Failed to update concepts');
        }
      }
    };

    const loadConceptConstraintOptions = async () => {
      try {
        const concepts = await updateConcepts(
          form.selectedConceptDomain,
          500000,
        );
        const result = await filterOptionsOfConstraints(
          form,
          variableDomain,
          concepts,
        );
        if (result) {
          setPossibleConceptNames(result.possibleConceptNames);
        }
      } catch (err) {
        console.error('Failed to filter concepts in constraints', err);
        dispatchError('Failed to update constraint concept options');
      }
    };

    const loadRelationConstraintOptions = async () => {
      try {
        const concepts = await updateConcepts(
          form.selectedConceptDomain,
          500000,
        );
        const result = await filterOptionsOfConstraints(
          form,
          variableDomain,
          concepts,
        );
        if (result) {
          console.log('Relation constraint result:', result.possibleRelations);
          setPossibleRelations(result.possibleRelations);
        }
      } catch (err) {
        console.error('Failed to filter relations in constraints', err);
        dispatchError('Failed to update constraint relation options');
      }
    };

    const dispatchError = (message: string) => {
      dispatch({
        type: 'SET_ERROR',
        formType,
        value: message,
      });
    };

    runFilteringLogic();
  }, [
    formType,
    form.selectedRelation,
    form.selectedConceptDomain,
    form.predictionType,
    form.tailValue,
    variableDomain,
  ]);

  return {
    loadingConcepts,
    loadingRelations,
    possibleVariableDomains,
    possibleConceptDomains,
    possibleRelations,
    possibleConceptNames,
    setPossibleConceptNames,
    setLoadingConcepts,
  };
};
