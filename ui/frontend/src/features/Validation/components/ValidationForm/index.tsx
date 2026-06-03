/*Component gathering all validation form inputs and handling the validation*/

import './index.css';
import { Button } from '@mui/material';
import React, { SetStateAction, useEffect, useState } from 'react';
import { Concept } from '../../../../static/datasets';
import {
  filterOptionsOfDomains,
  filterOptionsOfRelations,
} from '../../../QueryForm/components/MainForm/utils';
import { EntityInputs, RelationInput } from '../ValidationInputs';
import { PredictionType } from '../../../../contexts/QueryContext/types';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import {
  buildTriplesFromSelection,
  isEmptySelection,
  mapTriplesToValidationFormat,
} from '../../utils';
import { validateTriples } from '../../../../services/validation';

interface ValidationFormProps {
  triples: ScoredTriple[];
  setTriples: React.Dispatch<SetStateAction<ScoredTriple[]>>;
}

export type SelectedOptions = {
  subjectDomain: string;
  objectDomain: string;
  relation: string;
  subjectConcept: Concept;
  objectConcept: Concept;
};

export const ValidationForm = ({ setTriples }: ValidationFormProps) => {
  const [error, setError] = useState('');
  const [validationError, setValidationError] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const [possibleRelations, setPossibleRelations] = useState(['']);
  const [possibleSubjectDomains, setPossibleSubjectDomains] = useState(['']);
  const [possibleObjectDomains, setPossibleObjectDomains] = useState(['']);

  const initialOptions = {
    subjectDomain: '',
    objectDomain: '',
    relation: '',
    subjectConcept: { id: '', name: '' },
    objectConcept: { id: '', name: '' },
  };

  const [selectedOptions, setSelectedOptions] = useState(initialOptions);

  const handleValidate = async (triplesToValidate: ScoredTriple[]) => {
    setValidationError('');
    setLoading(true);

    try {
      const payload: [string, string, string][] =
        mapTriplesToValidationFormat(triplesToValidate);

      const scores = await validateTriples(payload);

      const validated = triplesToValidate.map((triple, i) => ({
        ...triple,
        score: scores[i],
      }));

      setTriples((prev) => [...prev, ...validated]);
    } catch (err) {
      console.error(err);
      setValidationError('Something went wrong validating your triples');
    } finally {
      setLoading(false);
    }
  };

  const handleAddTriples = async () => {
    setError('');
    setValidationError('');

    if (isEmptySelection(selectedOptions)) {
      return setError('All fields have to be filled to add a new triple');
    }

    try {
      const newTriples = await buildTriplesFromSelection(selectedOptions);
      await handleValidate(newTriples);
      setSelectedOptions(initialOptions);
    } catch (err) {
      console.error(err);
      setValidationError('Something went wrong while adding the triple.');
    }
  };

  useEffect(() => {
    setError('');
    filterOptionsOfRelations(
      selectedOptions.subjectDomain,
      selectedOptions.objectDomain,
      PredictionType.SUBJECT,
    )
      .then(setPossibleRelations)
      .catch((err) => {
        console.error('Failed to fetch relations', err);
        setError('Failed to update relations');
      });
  }, [selectedOptions.subjectDomain, selectedOptions.objectDomain]);

  useEffect(() => {
    setError('');
    filterOptionsOfDomains(
      selectedOptions.objectDomain,
      selectedOptions.relation,
      true,
    )
      .then(setPossibleSubjectDomains)
      .catch((err) => {
        console.error('Failed to fetch subject domains', err);
        setError('Failed to update subject domains');
      });
  }, [selectedOptions.objectDomain, selectedOptions.relation]);

  useEffect(() => {
    setError('');
    filterOptionsOfDomains(
      selectedOptions.subjectDomain,
      selectedOptions.relation,
      false,
    )
      .then(setPossibleObjectDomains)
      .catch((err) => {
        console.error('Failed to fetch object domains', err);
        setError('Failed to update object domains');
      });
  }, [selectedOptions.subjectDomain, selectedOptions.relation]);

  return (
    <>
      <form className="validation-form">
        <EntityInputs
          selectedOptions={selectedOptions}
          setSelectedOptions={setSelectedOptions}
          possibleDomains={possibleSubjectDomains}
          position="subject"
          setError={setError}
        />

        <RelationInput
          setSelectedOptions={setSelectedOptions}
          possibleRelations={possibleRelations}
          selectedOptions={selectedOptions}
        />

        <EntityInputs
          selectedOptions={selectedOptions}
          setSelectedOptions={setSelectedOptions}
          possibleDomains={possibleObjectDomains}
          position="object"
          setError={setError}
        />

        <div className="button-box">
          <Button
            onClick={handleAddTriples}
            variant="contained"
            disabled={loading}
          >
            {loading ? 'VALIDATING...' : 'VALIDATE'}
          </Button>
          <Button onClick={() => setSelectedOptions(initialOptions)}>
            Clear
          </Button>
        </div>
      </form>
      {(error || validationError) && (
        <p className="error">{validationError ? validationError : error}</p>
      )}
    </>
  );
};
