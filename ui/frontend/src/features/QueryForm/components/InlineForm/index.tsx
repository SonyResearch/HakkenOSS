/*Component that gathers all condition inputs */

import './index.css';
import BoolAndTypeInput from '../FormInputs/BoolAndTypeInput';
import VariableInputs from '../FormInputs/VariableInputs';
import PredictionTypeInput from '../FormInputs/PredictionTypeInput';
import ConceptInputs from '../FormInputs/ConceptInputs';
import { ConditionType } from '../../../../contexts/QueryContext/types';
import React, { RefObject, SetStateAction } from 'react';
import { InputType } from '../../../../contexts/QueryContext/types';
import RelationInput from '../FormInputs/RelationInput';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { useFilterInputOptions } from '../../../../hooks/useFilterFormTriples/useFilterInputOptions';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';

interface InlineFormProps {
  conditionType: ConditionType;
  isFirstConditionOfType: boolean;
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  refs: Record<InputType, RefObject<HTMLDivElement>>;
}

const InlineForm = ({
  conditionType,
  isFirstConditionOfType,
  setFocusedInput,
  refs,
}: InlineFormProps) => {
  const { queryMode } = useQueryContext();
  const { state, dispatch } = useQueryFormContext();
  const {
    loadingConcepts,
    loadingRelations,
    possibleVariableDomains,
    possibleRelations,
    possibleConceptDomains,
    possibleConceptNames,
    setPossibleConceptNames,
    setLoadingConcepts,
  } = useFilterInputOptions(
    state[conditionType].form,
    state.selectedVariableDomain,
    conditionType,
    dispatch,
  );
  return (
    <div className="input-container">
      {/*<PredictionTypeInput />*/}
      {(conditionType === ConditionType.constraints ||
        queryMode === 'complex') && (
        <BoolAndTypeInput
          conditionType={conditionType}
          isFirstConditionOfType={isFirstConditionOfType}
        ></BoolAndTypeInput>
      )}
      <VariableInputs
        stackRef={refs.variable}
        setFocusedInput={setFocusedInput}
        conditionType={conditionType}
        possibleVariableDomains={possibleVariableDomains}
      />

      <div className="input-wrapper relation" ref={refs.relation}>
        <PredictionTypeInput conditionType={conditionType} />
        <RelationInput
          setFocusedInput={setFocusedInput}
          conditionType={conditionType}
          possibleRelations={possibleRelations}
          loadingRelations={loadingRelations}
        />
      </div>

      <ConceptInputs
        stackRef={refs.concept}
        setFocusedInput={setFocusedInput}
        conditionType={conditionType}
        possibleConceptDomains={possibleConceptDomains}
        possibleConceptNames={possibleConceptNames}
        loadingConcepts={loadingConcepts}
        setLoadingConcepts={setLoadingConcepts}
        setPossibleConceptNames={setPossibleConceptNames}
      />
    </div>
  );
};

export default InlineForm;
