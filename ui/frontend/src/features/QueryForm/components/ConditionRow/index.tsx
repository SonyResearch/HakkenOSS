import './index.css';
import {
  Condition,
  ConditionType,
  InputType,
  PredictionType,
} from '../../../../contexts/QueryContext/types';
import {
  getAddValue,
  getConditionText,
  getTripleFromCondition,
} from '../MainForm/utils';
import editIcon from '../../../../assets/images/icons/edit-regular.svg';
import deleteIcon from '../../../../assets/images/icons/trash-can-regular.svg';
import copyIcon from '../../../../assets/images/icons/copy-regular.svg';
import squareIcon from '../../../../assets/images/icons/square-regular-full.svg';
import checkIcon from '../../../../assets/images/icons/square-check-regular.svg';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { addCondition } from '../../utils';
import { useHandleQueryForm } from '../../../../hooks/useHandleQueryForm';
import React, { SetStateAction, useRef, useState } from 'react';
import { Conditions } from '../../../../static/useCases';
import { PyramidGuide } from '../Pyramids';
import InlineForm from '../InlineForm';
import { getNextIndex } from '../../utils';

interface ConditionRowProps {
  condition: Condition;
  index: number;
  listing: number;
  conditions: Conditions;
  setConditions: (prevConditions: Record<number, Condition>) => void;
  conditionType: ConditionType;
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  isOnlyCondition: boolean;
  isGuideVisible: boolean;
  focusedInput: InputType | null;
  canAddMoreConditions: boolean;
}

const ConditionRow = ({
  conditions,
  setConditions,
  condition,
  index,
  listing,
  conditionType,
  setFocusedInput,
  isOnlyCondition,
  isGuideVisible,
  focusedInput,
  canAddMoreConditions,
}: ConditionRowProps) => {
  const { state, dispatch } = useQueryFormContext();
  const refs = {
    variable: useRef<HTMLDivElement>(null),
    relation: useRef<HTMLDivElement>(null),
    concept: useRef<HTMLDivElement>(null),
    filter: useRef<HTMLDivElement>(null),
  };
  const {
    handleAddNewVariable,
    handleAddNewConditions,
    checkExistingVariable,
  } = useHandleQueryForm();
  const [addingCondition, setAddingCondition] = useState<boolean>(false);
  const conditionText = `${getAddValue(listing, condition.addValue)} (${getConditionText(condition)})`;
  const nextIndex = getNextIndex(conditions);
  const editingThisCondition =
    state[conditionType].conditionToEdit?.index === index;
  const [addButtonHovered, setAddButtonHovered] = useState<boolean>(false);

  const shouldGuideBeVisible =
    conditionType === ConditionType.hypotheses &&
    isGuideVisible &&
    editingThisCondition;

  const handleAddButton = async () => {
    setAddingCondition(true);
    try {
      await addCondition(
        state,
        dispatch,
        checkExistingVariable,
        handleAddNewVariable,
        handleAddNewConditions,
        nextIndex,
        conditionType,
      );
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        formType: conditionType,
        value:
          (error as { message?: string })?.message ??
          'Something went wrong, please try again later',
      });
    } finally {
      setAddingCondition(false);
    }
  };

  const handleDelete = () => {
    const newConditions = { ...conditions };
    delete newConditions[index];
    setConditions(newConditions);
    if (isOnlyCondition) {
      dispatch({ type: 'RESET' });
    }
    if (editingThisCondition) {
      dispatch({ type: 'CANCEL_EDITING', formType: conditionType });
      dispatch({ type: 'CLEAR', formType: conditionType });
    }
  };
  const handleDuplicate = () => {
    const newConditions = {
      ...conditions,
      [nextIndex]: { ...conditions[index] },
    };
    setConditions(newConditions);
  };

  const handleEdit = () => {
    const condition = conditions[index];
    const { variable, concept, isSubjectPrediction } =
      getTripleFromCondition(condition);
    dispatch({
      type: 'SET_EDITING_CONDITION',
      formType: conditionType,
      payload: {
        ...state[conditionType],
        form: {
          ...state[conditionType].form,
          searchType: condition.conditionType,
          addValue: condition.addValue,
          selectedRelation: condition.condition.relation,
          tailValue: [concept.label],
          headValue: variable.label,
          predictionType: isSubjectPrediction
            ? PredictionType.SUBJECT
            : PredictionType.OBJECT,
          selectedConceptDomain: concept.domain,
        },
        conditionToEdit: { condition: conditions[index], index },
      },
    });
  };

  return (
    <>
      <div
        data-testid="condition-item"
        className={`${editingThisCondition ? 'editing' : ''} condition-row-wrapper`}
      >
        <div className="condition-row">
          <strong>{listing + 1}.</strong>
          {editingThisCondition ? (
            <InlineForm
              isFirstConditionOfType={listing === 0}
              setFocusedInput={setFocusedInput}
              conditionType={conditionType}
              refs={refs}
            />
          ) : (
            <span
              className={
                state[conditionType].conditionToEdit ? 'light-text' : ''
              }
            >
              {conditionText}
            </span>
          )}
        </div>
        <div className={`icon-box ${index === nextIndex - 1 ? 'first' : ''}`}>
          {editingThisCondition ? (
            <div className="check-icon" onClick={handleAddButton}>
              {addingCondition ? (
                <div
                  className="loading-spinner"
                  data-testid="loading-spinner"
                ></div>
              ) : (
                <img
                  src={addButtonHovered ? checkIcon : squareIcon}
                  onMouseEnter={() => setAddButtonHovered(true)}
                  onMouseLeave={() => setAddButtonHovered(false)}
                  onClick={handleAddButton}
                  alt="check icon"
                ></img>
              )}
            </div>
          ) : (
            <div className="edit-icon">
              <img
                data-testid={`edit-icon-${listing}`}
                src={editIcon}
                onClick={handleEdit}
                alt="edit icon"
              ></img>
            </div>
          )}
          {canAddMoreConditions && (
            <div
              className="copy-icon"
              style={
                state[conditionType].conditionToEdit?.index === index
                  ? { opacity: 0 }
                  : { opacity: 1 }
              }
            >
              <img
                data-testid={`duplicate-icon-${listing}`}
                src={copyIcon}
                style={
                  editingThisCondition
                    ? { opacity: '0', pointerEvents: 'none' }
                    : { opacity: '' }
                }
                onClick={handleDuplicate}
                alt="duplicate icon"
              ></img>
            </div>
          )}
          <div className="delete-icon">
            <img
              data-testid={`delete-icon-${listing}`}
              src={deleteIcon}
              onClick={handleDelete}
              alt="delete icon"
            ></img>
          </div>
        </div>
      </div>
      {shouldGuideBeVisible && (
        <PyramidGuide
          predictionType={
            state[conditionType].form.predictionType as PredictionType
          }
          highlight={focusedInput}
          refs={refs}
        />
      )}
    </>
  );
};

export default ConditionRow;
