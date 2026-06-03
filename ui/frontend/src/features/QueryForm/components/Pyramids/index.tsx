/*Pyramid indicators on top of form inputs*/

import { RefObject, SetStateAction, useEffect, useRef, useState } from 'react';
import './index.css';
import {
  ConditionType,
  InputType,
  PredictionType,
} from '../../../../contexts/QueryContext/types';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { useQueryContext } from '../../../../contexts/QueryContext';

interface PyramidGuideProps {
  highlight: InputType | null;
  predictionType: PredictionType;
  refs: Record<string, RefObject<HTMLDivElement>>;
}

interface PyramidToggleProps {
  highlight: InputType | null;
  isGuideVisible: boolean;
  setIsGuideVisible: React.Dispatch<SetStateAction<boolean>>;
}

const PyramidToggle = ({
  highlight,
  setIsGuideVisible,
  isGuideVisible,
}: PyramidToggleProps) => {
  const width = 35;
  const height = 25;
  const conceptWidth = 10;
  const variablePosition = {
    x: conceptWidth / 2,
    y: height - conceptWidth / 2,
  };
  const conceptPosition = {
    x: width - conceptWidth,
    y: height - conceptWidth,
  };
  return (
    <div
      className="mini-svg"
      onClick={() => setIsGuideVisible(!isGuideVisible)}
    >
      <svg width={width} height={height}>
        <circle
          opacity={highlight === InputType.VARIABLE ? 1 : 0.3}
          r={conceptWidth / 2}
          fill="var(--primary-pink)"
          cy={variablePosition.y}
          cx={variablePosition.x}
        ></circle>
        <path
          opacity={highlight === InputType.RELATION ? 1 : 0.7}
          d={`M ${variablePosition.x} ${variablePosition.y - conceptWidth / 2} C ${variablePosition.x + 5} ${variablePosition.y - height + 5}, ${conceptPosition.x} ${variablePosition.y - height + 5}, ${conceptPosition.x + conceptWidth / 2} ${conceptPosition.y}`}
          stroke="yellowgreen"
          fill="transparent"
        />
        <rect
          opacity={highlight === InputType.CONCEPT ? 1 : 0.3}
          height={conceptWidth}
          width={conceptWidth}
          fill="turquoise"
          rx={5}
          y={conceptPosition.y}
          x={conceptPosition.x}
        ></rect>
      </svg>
    </div>
  );
};

const PyramidGuide = ({
  highlight,
  predictionType,
  refs,
}: PyramidGuideProps) => {
  const { queryMode } = useQueryContext();
  const { state } = useQueryFormContext();
  const [trianglePoints, setTrianglePoints] = useState<string[]>();
  const svgRef = useRef<SVGSVGElement>(null);
  const width = '100%';
  const height = 150;
  const iconsY = 40;
  const conceptWidth = 20;
  const [positions, setPositions] = useState<Record<InputType, number>>({
    variable: 0,
    relation: 0,
    concept: 0,
    //filter: 0,
  });

  const updatePoints = () => {
    const newPoints: string[] = [];
    const newPositions: Record<InputType, number> = {
      variable: 0,
      relation: 0,
      concept: 0,
      //filter: 0,
    };
    Object.entries(refs).forEach(([key, ref]) => {
      if (!ref.current || !svgRef.current) return;
      const svgRect = svgRef.current.getBoundingClientRect();
      const rect = ref.current.getBoundingClientRect();
      const x1 = rect.left - svgRect.left;
      const x2 = rect.right - svgRect.left;
      const y1 = rect.top - svgRect.top;
      const midX = (x1 + x2) / 2;
      const pyramidY = key === 'relation' ? iconsY - 30 : iconsY;

      newPositions[key as InputType] = midX;
      newPoints.push(`${x1},${y1} ${midX},${pyramidY + 30} ${x2},${y1}`);
    });
    setPositions(newPositions);
    setTrianglePoints(newPoints);
  };

  useEffect(() => {
    updatePoints();
    window.addEventListener('resize', updatePoints);
    return () => {
      window.removeEventListener('resize', updatePoints);
    };
  }, [refs]);

  return (
    <div
      data-testid="svg-guide"
      className={`svg-guide-wrapper ${queryMode === 'simple' ? 'single' : ''}`}
    >
      <svg ref={svgRef} width={width} height={height}>
        {Object.values(positions).every((position) => position !== 0) && (
          <>
            <circle
              r={conceptWidth / 2}
              fill="var(--primary-pink)"
              cy={iconsY}
              cx={positions.variable}
            ></circle>
            <text
              x={positions.variable}
              y={iconsY + conceptWidth + 5}
              textAnchor="middle"
              fill="black"
            >
              {predictionType}
            </text>
            <path
              d={`M ${positions.variable} ${iconsY - conceptWidth / 2} C ${positions.variable + 50} ${iconsY - 50}, ${positions.concept - 50} ${iconsY - 50}, ${positions.concept + conceptWidth / 2} ${iconsY}`}
              stroke="yellowgreen"
              fill="transparent"
            />
            <text
              x={positions.relation}
              y={iconsY - conceptWidth}
              textAnchor="middle"
              fill="black"
            >
              relation
            </text>
            <rect
              height={conceptWidth}
              width={conceptWidth}
              fill="turquoise"
              rx={5}
              y={iconsY - conceptWidth / 2}
              x={positions.concept - conceptWidth / 2}
            ></rect>
            <text
              x={positions.concept}
              y={iconsY + conceptWidth + 5}
              textAnchor="middle"
              fill="black"
            >
              {predictionType === PredictionType.OBJECT
                ? PredictionType.SUBJECT
                : PredictionType.OBJECT}
            </text>
            {/*<image
              href={filterIcon}
              width={conceptWidth}
              filter="invert(54%) sepia(51%) saturate(3243%) hue-rotate(87deg) brightness(117%) contrast(93%)"
              y={iconsY}
              x={positions.filter - conceptWidth / 2}
            ></image>
            <svg
              r={conceptWidth / 2}
              fill="var(--primary-pink)"
              cy={iconsY}
              cx={positions.filter}
            ></svg>
            <text
              x={positions.filter}
              y={iconsY + conceptWidth + 5}
              textAnchor="middle"
              fill="black"
            >
              filter
            </text>*/}
          </>
        )}

        <defs>
          <linearGradient
            id="subject_gradient"
            x1="0%"
            y1="0%"
            x2="0%"
            y2="100%"
          >
            <stop offset="0%" stopColor="var(--primary-pink)" />
            <stop offset="50%" stopColor="rgba(255, 79, 205, 0.53)" />
            <stop offset="100%" stopColor="rgba(250, 237, 249, 0.12)" />
          </linearGradient>
          <linearGradient
            id="relation_gradient"
            x1="0%"
            y1="0%"
            x2="0%"
            y2="100%"
          >
            <stop offset="0%" stopColor="rgba(164, 252, 31, 0.76)" />
            <stop offset="50%" stopColor="rgba(164, 255, 79, 0.63)" />
            <stop offset="100%" stopColor="rgba(242, 250, 237, 0.12)" />
          </linearGradient>
          <linearGradient
            id="object_gradient"
            x1="0%"
            y1="0%"
            x2="0%"
            y2="100%"
          >
            <stop offset="0%" stopColor="rgba(9, 229, 207, 0.91)" />
            <stop offset="50%" stopColor="rgba(79, 255, 214, 0.63)" />
            <stop offset="100%" stopColor="rgba(237, 250, 248, 0.12)" />
          </linearGradient>
          {/*<linearGradient
            id="filter_gradient"
            x1="0%"
            y1="0%"
            x2="0%"
            y2="100%"
          >
            <stop offset="0%" stopColor="rgb(16, 223, 33)" />
            <stop offset="50%" stopColor="rgba(16, 207, 32, 0.59)" />
            <stop offset="100%" stopColor="rgba(13, 210, 29, 0)" />
          </linearGradient>*/}
        </defs>
        <polygon
          opacity={
            highlight === InputType.VARIABLE
              ? 1
              : state.selectedVariableDomain
                ? 0.3
                : 0
          }
          height={height}
          points={trianglePoints ? trianglePoints[0] : ''}
          fill={'url(#subject_gradient)'}
        ></polygon>
        <polygon
          opacity={
            highlight === InputType.RELATION
              ? 1
              : state[ConditionType.hypotheses].form.selectedRelation
                ? 0.3
                : 0
          }
          height={height}
          points={trianglePoints ? trianglePoints[1] : ''}
          fill={'url(#relation_gradient)'}
        />
        <polygon
          opacity={
            highlight === InputType.CONCEPT
              ? 1
              : state[ConditionType.hypotheses].form.selectedConceptDomain
                ? 0.3
                : 0
          }
          height={height}
          points={trianglePoints ? trianglePoints[2] : ''}
          fill={'url(#object_gradient)'}
        />
        {/*<polygon
          opacity={
            highlight === InputType.FILTER
              ? 1
              : state[ConditionType.hypotheses].form.selectedFilters.length > 1
                ? 0.3
                : 0
          }
          height={height}
          points={trianglePoints ? trianglePoints[3] : ''}
          fill={'url(#filter_gradient)'}
        />*/}
      </svg>
    </div>
  );
};

export { PyramidGuide, PyramidToggle };
