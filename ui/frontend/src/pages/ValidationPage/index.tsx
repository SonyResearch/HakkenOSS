import { useState } from 'react';
import './index.css';
import { ValidationForm } from '../../features/Validation/components/ValidationForm';
import { Concept } from '../../static/datasets';
import { ValidationCandidatePage } from '../../features/CandidateDashboard/components/CandidatePage/Validation';
import { ExplanationProvider } from '../../contexts/ExplanationContext';
import ValidationTable from '../../features/Validation/components/ValidationTable';

export type ScoredTriple = {
  triple: [Concept, string, Concept];
  score: number | undefined;
};

const ValidationPage = () => {
  const [scoredTriples, setScoredTriples] = useState<ScoredTriple[]>([]);
  const [selectedTriple, setSelectedTriple] = useState<ScoredTriple>();

  return (
    <section className="validation-page">
      <h1>Validate Triples</h1>
      <p className="intro-text">
        This feature scores your triples. Provide the subject, predicate, and
        object, and our model will return a confidence score for that triple.
      </p>
      <ValidationForm triples={scoredTriples} setTriples={setScoredTriples} />
      {scoredTriples.length !== 0 && (
        <ValidationTable
          scoredTriples={scoredTriples}
          setScoredTriples={setScoredTriples}
          setSelectedTriple={setSelectedTriple}
        />
      )}
      {selectedTriple && (
        <div
          className="validation-results"
          onClick={() => setSelectedTriple(undefined)}
        >
          <ExplanationProvider>
            <div onClick={(e) => e.stopPropagation()}>
              <ValidationCandidatePage selectedTriple={selectedTriple} />
            </div>
          </ExplanationProvider>
        </div>
      )}
    </section>
  );
};

export default ValidationPage;
