import React, { SetStateAction, useState } from 'react';
import ExplanationSection from '../../../Explanation/components/ExplanationSection';
import { Filters, Sorting, ResultViews } from '../../types';
import TabNav from '../../../../shared/components/TabNav';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';

interface CandidatePageLayoutProps {
  explanationsRequestedKeys: string[];
  notKeys: string[];
  isInValidation: boolean;
  renderContext: (args: {
    setPageToShow: React.Dispatch<React.SetStateAction<ResultViews>>;
    currentHypothesisIndexes: number[];
    setCurrentHypothesisIndexes: React.Dispatch<React.SetStateAction<number[]>>;
    filters: Filters;
    sorting: Sorting;
    setFilters: React.Dispatch<React.SetStateAction<Filters>>;
    setSorting: React.Dispatch<React.SetStateAction<Sorting>>;
  }) => React.ReactNode;
  pageToShow: ResultViews;
  setPageToShow: React.Dispatch<SetStateAction<ResultViews>>;
}

export const CandidatePageLayout = ({
  explanationsRequestedKeys,
  notKeys,
  isInValidation,
  renderContext,
  pageToShow,
  setPageToShow,
}: CandidatePageLayoutProps) => {
  const [currentHypothesisIndexes, setCurrentHypothesisIndexes] = useState<
    number[]
  >([0]);
  const { explanations } = useExplanationContext();

  const [filters, setFilters] = useState<Filters>({
    author: '',
    title: '',
    abstract: '',
  });

  const [sorting, setSorting] = useState<Sorting>({
    category: 'score',
    order: 'descending',
  });

  const hasExplanation = explanationsRequestedKeys.some(
    (key) => explanations.get(key)?.status === 'ready',
  );

  const tabs: Record<ResultViews, boolean> = {
    context: true,
    explanation: hasExplanation,
  };

  return (
    <section
      className={`candidate-page-wrapper ${isInValidation ? 'validation' : ''}`}
    >
      <TabNav options={tabs} setView={setPageToShow} currentView={pageToShow} />

      {pageToShow === 'context' ? (
        renderContext({
          setPageToShow,
          currentHypothesisIndexes,
          setCurrentHypothesisIndexes,
          filters,
          sorting,
          setFilters,
          setSorting,
        })
      ) : (
        <ExplanationSection
          explanationsRequestedKeys={explanationsRequestedKeys}
          currentHypothesisIndexes={currentHypothesisIndexes}
          notKeys={notKeys}
          isInValidation={isInValidation}
        />
      )}
    </section>
  );
};
