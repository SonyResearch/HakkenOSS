/*Component that handles query visualizations, displaying, amplifying and switching in case of multiple ones*/

import './index.css';
import React, {
  useState,
  useEffect,
  useRef,
  RefObject,
  SetStateAction,
  useMemo,
} from 'react';

import { Condition, ConditionType } from '../../../contexts/QueryContext/types';
import maximizeIcon from '../../../assets/images/icons/maximize.png';
import { CandidateResultType } from '../../CandidateDashboard/types';
import { getConditionGroups } from './utils';
import { buildNodeGraph } from './buildVisualizationGraph';
import leftArrow from '../../../assets/images/icons/left-long-solid.svg';
import rightArrow from '../../../assets/images/icons/right-long-solid.svg';
import { SearchState } from '../../../contexts/QueryFormContext/types';

interface QueryVisualizationProps {
  hypotheses: Record<number, Condition>;
  constraints: Record<number, Condition>;
  query: string;
  page: 'search' | 'results';
  selectedCandidate?: CandidateResultType | null;
  conditionsLengths: Record<ConditionType, number>;
  state?: SearchState;
  selectedCandidateName?: string;
  setCurrentHypothesisIndexes?: React.Dispatch<SetStateAction<number[]>>;
  shortestPathLength?: number[];
}

export const QueryVisualization = ({
  hypotheses,
  constraints,
  selectedCandidate,
  conditionsLengths,
  page,
  state,
  selectedCandidateName,
  setCurrentHypothesisIndexes,
  shortestPathLength,
}: QueryVisualizationProps) => {
  const svgRefs = useRef<RefObject<SVGSVGElement>[]>([]);
  const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);
  const controllerRefs = [
    useRef<HTMLImageElement>(null),
    useRef<HTMLImageElement>(null),
  ];
  const containerRef = useRef<HTMLDivElement>(null);

  const [amplifyVisualization, setAmplifyVisualization] =
    useState<boolean>(false);
  const [currentVisualization, setCurrentVisualization] = useState<number>(0);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const clickedButtons = [...controllerRefs].some(
        (ref) => ref.current && ref.current.contains(event.target as Node),
      );
      const clickedVisualization =
        containerRef.current &&
        containerRef.current.contains(event.target as Node);
      if (!clickedButtons && !clickedVisualization) {
        setAmplifyVisualization(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const editingSomeCondition =
    state &&
    state[ConditionType.hypotheses].conditionToEdit &&
    state[ConditionType.constraints].conditionToEdit;

  const conditionGroups = useMemo(
    () => getConditionGroups(hypotheses, constraints),
    [hypotheses, constraints],
  );

  useEffect(() => {
    svgRefs.current = conditionGroups.map(
      (_, i) => svgRefs.current[i] ?? React.createRef<SVGSVGElement>(),
    );
  }, [conditionGroups]);

  useEffect(() => {
    if (editingSomeCondition && page === 'search') return; //don't display visualization if we're editing conditions
    const timeoutMs = 300;

    const timer = setTimeout(() => {
      buildNodeGraph({
        conditionGroups,
        svgRefs,
        selectedCandidate,
        selectedCandidateName,
        setCurrentHypothesisIndexes,
        shortestPathLength,
      });
    }, timeoutMs);

    return () => clearTimeout(timer);
  }, [
    conditionGroups,
    selectedCandidate,
    selectedCandidateName,
    shortestPathLength,
    editingSomeCondition,
    page,
  ]);

  const goToNext = () => {
    const nextVisualization = Math.min(
      currentVisualization + 1,
      svgRefs.current.length - 1,
    );
    setCurrentVisualization(nextVisualization);
    slideToVisualization(nextVisualization);
  };

  const goToPrevious = () => {
    const previousVisualization = Math.max(currentVisualization - 1, 0);
    setCurrentVisualization(previousVisualization);
    slideToVisualization(previousVisualization);
  };

  const handleAmplifyClick = (index: number) => {
    setAmplifyVisualization(true);
    setCurrentVisualization(index);
    setTimeout(() => {
      slideToVisualization(index);
    }, 0);
  };

  //keyboard controls
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowRight') goToNext();
    else if (event.key === 'ArrowLeft') goToPrevious();
  };

  useEffect(() => {
    const handler = () => slideToVisualization(currentVisualization);
    window.addEventListener('resize', handler);
    return () => {
      window.removeEventListener('resize', handler);
    };
  }, [currentVisualization]);

  const slideToVisualization = (index: number) => {
    const wrapperDiv = wrapperRefs.current[index];
    if (wrapperDiv?.scrollIntoView) {
      wrapperDiv.scrollIntoView({
        behavior: 'smooth',
        inline: 'center',
        block: 'nearest',
      });
    }
  };

  /*only show arrow controllers if we have more visualizations than they can possibly fit*/
  const showSliderControllers =
    (page === 'search'
      ? conditionGroups.length > 3
      : conditionGroups.length > 1) || amplifyVisualization;

  return (
    <div
      tabIndex={0}
      className={`${amplifyVisualization ? 'container-active' : 'container'} ${page === 'results' && 'results'}`}
      onKeyDown={handleKeyDown}
    >
      {showSliderControllers && currentVisualization !== 0 && (
        <img
          src={leftArrow}
          ref={controllerRefs[0]}
          className="slider-controllers left"
          onClick={goToPrevious}
        ></img>
      )}
      <div className="slider-wrapper" ref={containerRef}>
        {svgRefs.current.map((ref, index) => (
          <div key={index}>
            <div
              ref={(el) => (wrapperRefs.current[index] = el)}
              className="visualization"
              style={{
                transform:
                  currentVisualization === index &&
                  showSliderControllers &&
                  page === 'search'
                    ? 'scale(1)'
                    : 'scale(0.9)',
                zIndex:
                  currentVisualization === index && showSliderControllers
                    ? 1
                    : 0,
                transition: 'transform 0.3s ease',
              }}
            >
              <svg ref={ref}></svg>
              {amplifyVisualization && (
                <>
                  {' '}
                  <div
                    style={{
                      height: `clamp(50%, ${conditionsLengths[ConditionType.hypotheses] * 20}%, 80%`,
                    }}
                    className="condition-type-container hypotheses"
                  >
                    <p>
                      {conditionsLengths[ConditionType.hypotheses] > 1
                        ? 'Hypotheses'
                        : 'Hypothesis'}
                    </p>
                  </div>
                  {Object.values(constraints)[0] &&
                    Object.values(constraints)[0].condition.relation && (
                      <div
                        style={{
                          height: `clamp(50%, ${conditionsLengths[ConditionType.constraints] * 20}%, 80%`,
                        }}
                        className="condition-type-container constraints"
                      >
                        <p>
                          {conditionsLengths[ConditionType.constraints] > 1
                            ? 'Constraints'
                            : 'Constraint'}
                        </p>
                      </div>
                    )}
                </>
              )}
              {!amplifyVisualization && (
                <>
                  <img
                    className="maximize-icon"
                    src={maximizeIcon}
                    onClick={() => handleAmplifyClick(index)}
                    alt="maximize icon"
                  ></img>
                </>
              )}
            </div>
          </div>
        ))}
      </div>
      {showSliderControllers &&
        currentVisualization !== conditionGroups.length - 1 && (
          <img
            src={rightArrow}
            ref={controllerRefs[1]}
            className="slider-controllers right"
            onClick={goToNext}
          ></img>
        )}
    </div>
  );
};
