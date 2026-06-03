import './index.css';
import React, { SetStateAction, useEffect, useRef, useState } from 'react';
import { buildExplanationGraph } from './buildExplanationGraph';
import xIcon from '../../../assets/images/icons/window-close-regular.svg';
import amplifyIcon from '../../../assets/images/icons/maximize.png';
import { ParsedExplanation } from '../../../contexts/ExplanationContext/types';
import { getTriplesFromExplanationItem } from './utils';
import { getNodeNames } from '../../QueryForm/components/MainForm/utils';
import { useMobile } from '../../../hooks/useMobile';

interface ExplainabilityGraphProps {
  selectedExplanation: ParsedExplanation;
  nodeNameMap: Record<string, string>;
  setNodeNameMap: React.Dispatch<SetStateAction<Record<string, string>>>;
  isInValidation: boolean;
}

const ExplainabilityGraph = ({
  selectedExplanation,
  nodeNameMap,
  setNodeNameMap,
  isInValidation,
}: ExplainabilityGraphProps) => {
  const isMobile = useMobile();
  const [maxNumberOfExplanations, setMaxNumberOfExplanations] =
    useState<number>(3);
  const [numOfExplanations, setNumOfExplanations] = useState<number>(3);
  const [isAmplified, setIsAmplified] = useState<boolean>(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const shortestLength = Math.min(
    ...selectedExplanation.explanations.map(
      (explanation) => explanation.length,
    ),
  );

  useEffect(() => {
    async function fetchNodeNames() {
      const triples = getTriplesFromExplanationItem(
        selectedExplanation.explanations,
      );
      const nodeIds = new Set<string>();
      triples.forEach(({ head, tail }) => {
        nodeIds.add(head);
        nodeIds.add(tail);
      });

      try {
        const mappedNames = await getNodeNames(Array.from(nodeIds));
        setNodeNameMap(mappedNames);
      } catch (error) {
        console.error('Failed to fetch node names:', error);
      }
    }

    fetchNodeNames();
  }, [selectedExplanation.explanations]);

  useEffect(() => {
    const triples = getTriplesFromExplanationItem(
      selectedExplanation.explanations,
      numOfExplanations,
      setMaxNumberOfExplanations,
    );
    const timer = setTimeout(() => {
      buildExplanationGraph(
        selectedExplanation,
        svgRef,
        triples,
        nodeNameMap,
        isMobile,
      );
    }, 100);

    return () => clearTimeout(timer);
  }, [nodeNameMap, selectedExplanation, numOfExplanations]);

  return (
    <div
      tabIndex={0}
      className={`${isAmplified ? 'amplified' : ''} explanation-graph`}
    >
      <svg ref={svgRef}></svg>
      <div className="explanation-graph-parameters">
        <strong>Complexity: {shortestLength}</strong>
        <label>
          Number of explanations
          <input
            className="num-of-explanations"
            onChange={(e) => setNumOfExplanations(Number(e.target.value))}
            type="range"
            value={Math.min(numOfExplanations, maxNumberOfExplanations)}
            min={1}
            max={Math.min(maxNumberOfExplanations, 10)}
          ></input>
          {Math.min(numOfExplanations, maxNumberOfExplanations)}
        </label>
      </div>
      {!isInValidation ? (
        isAmplified ? (
          <img
            className="amplifying-icon close"
            onClick={() => setIsAmplified(false)}
            src={xIcon}
            alt="closing icon"
          ></img>
        ) : (
          <img
            className="amplifying-icon"
            onClick={() => setIsAmplified(true)}
            src={amplifyIcon}
            alt="amplify icon"
          ></img>
        )
      ) : null}
    </div>
  );
};

export { ExplainabilityGraph };
