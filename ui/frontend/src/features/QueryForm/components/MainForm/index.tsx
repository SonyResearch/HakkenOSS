/*Main form component, displaying the condition list for both hypotheses and constraints and handling the query submission*/

import React, { useState, useEffect } from 'react';
import './index.css';
import ConditionList from '../ConditionList';
import { Button, Stack } from '@mui/material';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { useNavigate } from 'react-router-dom';
import HistoryList from '../../../History/components/HistoryList';
import {
  Condition,
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';
import { QueryVisualization } from '../../../Visualizations/QueryVisualization';
import { defaultData } from '../../../../contexts/QueryContext/utils';
import LightTooltip from '../../../../shared/components/LightToolTip';
import { useHandleQueryForm } from '../../../../hooks/useHandleQueryForm';
import { getTripleFromCondition } from './utils';
import ErrorBox from '../../../../shared/components/ErrorBox';
import LoadingQuery from '../../../../shared/components/LoadingQueryModal';
import { QueryModeToggle } from '../QueryModeToggle';

const MainForm = () => {
  const { state, dispatch } = useQueryFormContext();
  const { createQuery } = useHandleQueryForm();
  const [showConstraints, setShowConstraints] = useState<boolean>(false);
  const [isGuideVisible, setIsGuideVisible] = useState<boolean>(true);
  const [focusedInput, setFocusedInput] = useState<InputType | null>(null);
  const [queryError, setQueryError] = useState<{
    message: string;
    level: 'low' | 'high';
  } | null>(null);
  const {
    resetToDefault,
    setCandidatesNumber,
    candidatesNumber,
    query,
    hypotheses,
    setHypotheses,
    constraints,
    setConstraints,
    setSearchedParameters,
    example,
    setExample,
    queryMode,
    isSearching,
    setIsSearching,
  } = useQueryContext();
  const [tempConstraints, setTempConstraints] = useState<
    Record<number, Condition>
  >(defaultData.constraints);
  const isSearchButtonDisabled =
    !query ||
    state[ConditionType.hypotheses].conditionToEdit !== null ||
    state[ConditionType.constraints].conditionToEdit !== null;
  const isResetSearchButtonDisabled = !query;
  const navigate = useNavigate();
  const conditionsLengths: Record<ConditionType, number> = {
    [ConditionType.hypotheses]: Object.keys(hypotheses).length,
    [ConditionType.constraints]: Object.keys(constraints).length,
  };

  const clearQuery = () => {
    dispatch({ type: 'RESET' });
    resetToDefault();
  };
  const showVisualization =
    (conditionsLengths[ConditionType.hypotheses] > 0 &&
      !state[ConditionType.hypotheses].conditionToEdit) ||
    (conditionsLengths[ConditionType.constraints] > 0 &&
      !state[ConditionType.constraints].conditionToEdit) ||
    conditionsLengths[ConditionType.hypotheses] > 1 ||
    conditionsLengths[ConditionType.constraints] > 1;

  useEffect(() => {
    if (conditionsLengths[ConditionType.constraints] > 0)
      setShowConstraints(true);
  }, [constraints]);

  useEffect(() => {
    setTempConstraints(defaultData.constraints);
  }, [state.selectedVariableDomain]);

  useEffect(() => {
    if (Object.values(hypotheses).length && !state.selectedVariableDomain) {
      const { variable } = getTripleFromCondition(Object.values(hypotheses)[0]);
      dispatch({ type: 'UPDATE_VARIABLE_DOMAIN', value: variable.domain });
    }
  }, []);

  const handleSearch = async () => {
    setIsSearching(true);
    try {
      await createQuery();
      setSearchedParameters({
        query: query,
        hypotheses: hypotheses,
        constraints: constraints,
        candidatesNumber: candidatesNumber,
      });
      if (example) setExample(null);
      navigate('/query/results');
    } catch (error) {
      console.log('error', error);
      if ((error as Error).message === 'No results found for this query') {
        setQueryError({
          message:
            'We could not find any results for this query, please try with another one',
          level: 'low',
        });
      } else {
        setQueryError({
          message:
            'Something went wrong processing this query, try again later',
          level: 'high',
        });
      }
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <section className="main-form">
      <h2>BUILD QUERY</h2>
      {showVisualization && (
        <QueryVisualization
          hypotheses={hypotheses}
          constraints={constraints}
          query={query}
          page={'search'}
          conditionsLengths={conditionsLengths}
          state={state}
        />
      )}
      <QueryModeToggle
        tempConstraints={tempConstraints}
        setTempConstraints={setTempConstraints}
      />
      <section className={`query-box-display`}>
        <ConditionList
          isGuideVisible={isGuideVisible}
          setIsGuideVisible={setIsGuideVisible}
          focusedInput={focusedInput}
          setFocusedInput={setFocusedInput}
          conditionType={ConditionType.hypotheses}
          conditions={hypotheses}
          setConditions={setHypotheses}
          conditionLengths={conditionsLengths}
        />
        {showConstraints && queryMode === 'simple' && (
          <div className="constraints-container">
            <hr></hr>
            <ConditionList
              isGuideVisible={isGuideVisible}
              setIsGuideVisible={setIsGuideVisible}
              focusedInput={focusedInput}
              setFocusedInput={setFocusedInput}
              conditions={constraints}
              setConditions={setConstraints}
              conditionType={ConditionType.constraints}
              conditionLengths={conditionsLengths}
            />
          </div>
        )}
        {queryMode === 'simple' && (
          <button
            className={`constraints-expand-button ${showConstraints ? 'up' : 'down'}`}
            onClick={() => setShowConstraints(!showConstraints)}
          >
            <span>
              {showConstraints ? 'Hide Constraints' : 'Show Constraints'}
            </span>
          </button>
        )}
      </section>
      <label style={{ float: 'right' }} htmlFor="candidates-number">
        Candidates to return: {candidatesNumber}
        <input
          type="range"
          min="1"
          max="20"
          step="1"
          style={{ marginLeft: '1rem' }}
          value={candidatesNumber}
          onChange={(e) => setCandidatesNumber(Number(e.target.value))}
        ></input>
      </label>
      <Stack
        direction="row"
        justifyContent={'flex-end'}
        sx={{ width: '100%', paddingTop: '2rem' }}
      >
        <LightTooltip
          title={`Reset query conditions and formula `}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <span>
            <Button
              onClick={clearQuery}
              variant="text"
              size="small"
              disabled={isResetSearchButtonDisabled}
              sx={{ color: 'black' }}
            >
              Reset
            </Button>
          </span>
        </LightTooltip>
        <LightTooltip
          title={`Search query formula `}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <span>
            <Button
              onClick={handleSearch}
              disabled={isSearchButtonDisabled}
              variant="contained"
              className="find-button"
              sx={{
                '&.Mui-disabled': {
                  opacity: '0.5',
                  color: 'white',
                },
                padding: '0.4rem 2rem',
                backgroundColor: 'var(--primary-pink)',
                fontWeight: 700,
              }}
            >
              {isSearching ? 'Searching...' : 'FIND'}
            </Button>
          </span>
        </LightTooltip>
      </Stack>
      <HistoryList />
      {queryError && <ErrorBox error={queryError} setError={setQueryError} />}
      {isSearching && <LoadingQuery />}
    </section>
  );
};

export default MainForm;
