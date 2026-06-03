import * as d3 from 'd3';
import {
  constructNodesFromData,
  getLinksDataFromConditions,
  relationsArray,
} from './utils';
import { QueryLinkType, QueryNodeType } from './types';
import { ConditionType } from '../../../contexts/QueryContext/types';
import { Condition } from '../../../contexts/QueryContext/types';
import { CandidateResultType } from '../../CandidateDashboard/types';
import React, { MutableRefObject, RefObject, SetStateAction } from 'react';
import { addArrows, getRelationLabels } from '../Common/utils';
import { createSimulation } from './simulation';

const CONSTANTS = {
  WIDTH: 700,
  HEIGHT: 550,
  HEIGHT2: 800,
  MAX_CONCEPT_NAME_LENGTH: 18,
  VARIABLE_RADIUS: 35,
  CONCEPT_WIDTH: 40,
  RELATION_RADIUS: 8,
  CURVE: 70,
};

interface BuildNodeGraphProps {
  conditionGroups: Condition[][];
  svgRefs: MutableRefObject<RefObject<SVGSVGElement>[]>;
  selectedCandidate?: CandidateResultType | null;
  selectedCandidateName?: string;
  setCurrentHypothesisIndexes?: React.Dispatch<SetStateAction<number[]>>;
  shortestPathLength?: number[];
}

export const buildNodeGraph = ({
  conditionGroups,
  svgRefs,
  selectedCandidate,
  selectedCandidateName,
  setCurrentHypothesisIndexes,
  shortestPathLength,
}: BuildNodeGraphProps) => {
  let startingConditionIndex = 0;
  conditionGroups.map((conditionGroup, index) => {
    d3.select(svgRefs.current[index]?.current).selectAll('*').remove();

    /*format data for links and nodes from conditions*/
    const { linksData, nodeNames } = getLinksDataFromConditions(
      conditionGroup,
      startingConditionIndex,
    );
    const nodes = constructNodesFromData(nodeNames, linksData);

    const svg = d3
      .select(svgRefs.current[index]?.current)
      .attr('viewBox', [0, 0, CONSTANTS.WIDTH, CONSTANTS.HEIGHT])
      .attr('width', '100%')
      .attr('height', '100%');

    /*Create arrows pointing from links to nodes*/
    addArrows(svg, [
      { distanceFromNode: 5, id: 'leftGreyArrowHead', width: 8, color: 'gray' },
      {
        distanceFromNode: CONSTANTS.VARIABLE_RADIUS,
        id: 'rightGreyArrowHead',
        width: 8,
        color: 'gray',
      },
      {
        distanceFromNode: CONSTANTS.VARIABLE_RADIUS - 4,
        id: 'leftGreenArrowHead',
        width: 2.5,
        color: 'green',
      },
      {
        distanceFromNode: CONSTANTS.CONCEPT_WIDTH / 2 + 2,
        id: 'rightGreenArrowHead',
        width: 2.5,
        color: 'green',
      },
    ]);

    /*Create the  simulation with d3*/
    const simulation = createSimulation(nodes, linksData);

    const variableNode = nodes.find((node) => node.group === 'variable');
    if (variableNode) {
      //fix variable node position to middle
      variableNode.fx = CONSTANTS.WIDTH / 2;
      variableNode.fy = CONSTANTS.HEIGHT / 2;
    }

    /*Add the links*/
    const link = svg
      .append('g')
      .selectAll('.link')
      .data(linksData)
      .enter()
      .append('path')
      .attr('stroke', (d) =>
        d.type === ConditionType.hypotheses ? 'green' : 'grey',
      )
      .attr('stroke-dasharray', (d) =>
        d.type === ConditionType.hypotheses ? 5.5 : 0,
      )
      .attr('stroke-opacity', 1)
      .attr('fill', 'white')
      .attr('stroke-width', (d) =>
        d.type === ConditionType.hypotheses ? 6 : 2,
      )
      .attr('marker-end', (d) => {
        if (relationsArray.includes((d.target as QueryNodeType).id.trim())) {
          return '';
        }
        if (d.type === ConditionType.hypotheses) {
          if (
            d.source === variableNode ||
            (d.source as QueryNodeType).group === 'relation'
          )
            return 'url(#rightGreenArrowHead)';
          else return 'url(#leftGreenArrowHead)';
        } else {
          if (d.target === variableNode) return 'url(#rightGreyArrowHead)';
          else return 'url(#leftGreyArrowHead)';
        }
      });

    /*Labels indicating the relation name of each link*/
    const relationLabels = getRelationLabels(svg, linksData);

    /*Add complexity number to links in case we have it*/
    const shortestPathNumber = svg
      .append('g')
      .attr('class', 'shortest-path')
      .selectAll('g')
      .data(linksData)
      .enter()
      .append('g');

    if (selectedCandidate) {
      shortestPathNumber
        .append('circle')
        .attr('r', 10)
        .attr('fill', 'green')
        .attr('stroke-width', 1)
        .attr('stroke', 'black')
        .attr('opacity', (d) => (d.conditionIndex === null ? 0 : 1))
        .on('mouseover', function () {
          d3.select(this).attr('stroke-width', 2).attr('stroke', 'lightgreen');
        })
        .on('mouseout', function () {
          d3.select(this).attr('stroke-width', 1).attr('stroke', 'black');
        });

      shortestPathNumber
        .append('text')
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('y', 2)
        .attr('font-size', '0.5rem')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text((d) =>
          d.conditionIndex !== null &&
          shortestPathLength &&
          shortestPathLength[d.conditionIndex]
            ? Object.values(shortestPathLength)[d.conditionIndex].toString()
            : '?',
        );
    }

    /*Create nodes*/
    const node = svg
      .append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .call(
        d3
          .drag<SVGGElement, QueryNodeType>()
          .on('start', dragStarted)
          .on('drag', dragged)
          .on('end', dragended),
      );

    node //squares for concepts
      .filter((node) => node.group === 'concept')
      .append('rect')
      .attr('cursor', 'grab')
      .attr('x', -CONSTANTS.CONCEPT_WIDTH / 2)
      .attr('y', -CONSTANTS.CONCEPT_WIDTH / 2)
      .attr('width', CONSTANTS.CONCEPT_WIDTH)
      .attr('height', CONSTANTS.CONCEPT_WIDTH)
      .attr('rx', 10)
      .attr('ry', 10)
      .attr('fill', (node) =>
        node.type === ConditionType.hypotheses
          ? 'darkturquoise'
          : 'transparent',
      )
      .attr('brightness', '100%')
      .attr('stroke-width', (node) =>
        node.type === ConditionType.hypotheses ? '1' : '0',
      )
      .attr('stroke', 'black');

    node //circles for the relations and variable
      .filter((node) => node.group !== 'concept')
      .append('circle')
      .attr('cursor', 'grab')
      .attr('r', (node) =>
        node.group === 'variable'
          ? CONSTANTS.VARIABLE_RADIUS
          : CONSTANTS.RELATION_RADIUS,
      )
      .attr('fill', (node) =>
        node.group === 'variable' ? 'var(--primary-pink)' : 'yellowgreen',
      )
      .attr('stroke-width', '1')
      .attr('stroke', 'black');

    //text inside nodes
    node
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', function (node) {
        return node.group === 'concept' ? '0.2rem' : '6';
      })
      .attr('dx', function (node) {
        return node.group === 'concept' ? '0rem' : '0';
      })
      .attr('fill', function (node) {
        return node.group === 'concept'
          ? 'marineblue'
          : selectedCandidateName
            ? 'lightgray'
            : 'white';
      })
      .attr('font-weight', 'bold')
      .attr('font-size', function (node) {
        return node.group === 'variable' && !selectedCandidate
          ? '1.5rem'
          : node.group === 'concept'
            ? '0.7rem'
            : '1rem';
      })
      .text(function (node) {
        if (node.group === 'variable') {
          if (selectedCandidate) return '';
          return 'x';
        } else if (
          selectedCandidate &&
          node.group === 'concept' &&
          node.type === ConditionType.hypotheses &&
          !Number.isNaN(Number(node.conditionIndex))
        ) {
          if (
            !Number.isNaN(
              Math.floor(
                Object.values(selectedCandidate.conditionsScores)[
                  node.conditionIndex ?? 0
                ] * 100,
              ) / 100,
            )
          )
            return (
              Math.floor(
                Object.values(selectedCandidate.conditionsScores)[
                  node.conditionIndex ?? 0
                ] * 100, //specific to each concept score
              ) / 100
            );
          else return selectedCandidate.queryScore.toString().slice(0, 4);
        } else return '';
      });

    //add concept name on top of nodes
    node
      .append('text')
      .attr('dx', 0)
      .attr('dy', (node) =>
        node.group === 'variable'
          ? -(CONSTANTS.VARIABLE_RADIUS + 5)
          : node.group === 'concept'
            ? CONSTANTS.CONCEPT_WIDTH / 2 + 15
            : -(CONSTANTS.RELATION_RADIUS + 5),
      )
      .attr('text-anchor', 'middle')
      .text((node) => {
        const conceptName = node.id.slice(1, node.id.indexOf('#'));
        if (node.group === 'concept') {
          if (conceptName.length > CONSTANTS.MAX_CONCEPT_NAME_LENGTH)
            return (
              conceptName.slice(0, CONSTANTS.MAX_CONCEPT_NAME_LENGTH) + '...'
            );
          else return conceptName;
        } else {
          return node.id;
        }
      })
      .attr('fill', 'grey')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold');

    node
      .append('text')
      .attr('dx', 0)
      .attr('dy', CONSTANTS.CONCEPT_WIDTH / 2 + 27)
      .attr('text-anchor', 'middle')
      .text((node) => {
        if (node.group === 'concept') {
          const domainName = node.id
            .slice(node.id.indexOf('#') + 1)
            .replace(/_/g, ' ');
          return domainName;
        }
        return '';
      })
      .attr('fill', 'darkgrey')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold');

    //add the name of the predicted candidate
    if (selectedCandidateName) {
      svg
        .append('text')
        .attr('text-anchor', 'middle')
        .attr('dx', variableNode?.fx ?? 0)
        .attr('dy', variableNode?.fy ? variableNode?.fy - 55 : 0)
        .attr('fill', 'var(--primary-pink)')
        .attr('font-size', '20px')
        .attr('font-weight', 'bold')
        .text(selectedCandidateName ?? '');
    }

    //connect everything in place through the simulation (nodes, links, labels...)
    simulation.on('tick', () => {
      link.attr('d', curveLink).attr('fill', 'transparent');
      link.each(function (d) {
        const totalLength = this.getTotalLength();
        const midPoint = this.getPointAtLength(totalLength / 2);
        d.midX = midPoint.x;
        d.midY = midPoint.y;
      });

      shortestPathNumber
        .attr('transform', (d) => {
          if (
            d.type === ConditionType.constraints ||
            !selectedCandidate ||
            !shortestPathLength
          )
            return null;
          else return `translate(${d.midX}, ${d.midY})`;
        })
        .attr('cursor', 'pointer')
        .on('click', (_, d) => {
          if (setCurrentHypothesisIndexes && d.conditionIndex)
            setCurrentHypothesisIndexes([d.conditionIndex]);
        });

      relationLabels
        .attr(
          'x',
          (d) =>
            (((d.source as QueryNodeType).x ?? 0) +
              ((d.target as QueryNodeType).x ?? 0)) /
            2,
        )
        .attr(
          'y',
          (d) =>
            (((d.source as QueryNodeType).y ?? 0) +
              ((d.target as QueryNodeType)?.y ?? 0)) /
            2,
        )
        .attr('padding', '30')
        .attr('dy', '-5px')
        .attr('transform', (d) => {
          //to make the relation labels follow the link inclination
          const sourceX = (d.source as QueryNodeType).x ?? 0;
          const sourceY = (d.source as QueryNodeType).y ?? 0;
          const targetX = (d.target as QueryNodeType).x ?? 0;
          const targetY = (d.target as QueryNodeType).y ?? 0;
          const x = (sourceX + targetX) / 2;
          const y = (sourceY + targetY) / 2;
          let angle =
            (Math.atan2(targetY - sourceY, targetX - sourceX) * 180) / Math.PI;
          if (angle > 90 || angle < -90) {
            angle += 180;
          }
          return `rotate(${angle}, ${x}, ${y})`;
        });
      node.attr('transform', (link) => `translate(${link.x},${link.y})`);
    });

    function curveLink(d: QueryLinkType) {
      const sourceX = (d.source as QueryNodeType).x ?? 0;
      const sourceY = (d.source as QueryNodeType).y ?? 0;
      const targetX = (d.target as QueryNodeType).x ?? 0;
      const targetY = (d.target as QueryNodeType).y ?? 0;
      const curveOffset =
        d.type === ConditionType.hypotheses
          ? d.target === variableNode
            ? -CONSTANTS.CURVE
            : CONSTANTS.CURVE
          : 0; //curve for hypothesis, straight line for constraints

      const midpoint_x = (sourceX + targetX) / 2;
      const midpoint_y = (sourceY + targetY) / 2;

      const dx = targetX - sourceX;
      const dy = targetY - sourceY;

      const curve = Math.sqrt(dx * dx + dy * dy);

      const offSetX = midpoint_x + curveOffset * (dy / curve);
      const offSetY = midpoint_y - curveOffset * (dx / curve);

      return (
        // prettier-ignore
        'M' + sourceX + ',' + sourceY + 'S' + offSetX + ',' + offSetY + ' ' + targetX + ',' + targetY
      );
    }

    //Be able to move nodes around with the cursor
    function dragStarted(
      event: d3.D3DragEvent<SVGGElement, QueryNodeType, unknown>,
      node: QueryNodeType,
    ) {
      if (!event.active) simulation.alphaTarget(0.3).restart(); //gives movement to all nodes and update during drag
      if (node.group !== 'variable') {
        node.fx = node.x;
        node.fy = node.y;
      }
    }

    function dragged(
      event: d3.D3DragEvent<SVGGElement, QueryNodeType, unknown>,
      node: QueryNodeType,
    ) {
      if (node.group !== 'variable') {
        node.fx = event.x;
        node.fy = event.y;
      }
    }

    function dragended(
      event: d3.D3DragEvent<SVGGElement, QueryNodeType, unknown>,
      node: QueryNodeType,
    ) {
      if (!event.active) simulation.alphaTarget(0); //stops movement when event finishes
      if (node.group !== 'variable') {
        node.fx = null;
        node.fy = null;
      }
    }
    startingConditionIndex = conditionGroup.length;
  });
  return svgRefs;
};
