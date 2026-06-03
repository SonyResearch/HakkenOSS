/*d3 force graph behaviour and style*/

import './index.css';
import * as d3 from 'd3';
import {
  constructNodesFromData,
  distribute,
  getLinksDataFromTriples,
  normalizeName,
  positionByHop,
} from './utils';
import {
  ExplanationLinkType,
  ExplanationNodeType,
  ExplanationTriple,
} from './types';
import { RefObject } from 'react';
import { addArrows, getRelationLabels } from '../Common/utils';
import { LinkType } from '../Common/types';
import { ParsedExplanation } from '../../../contexts/ExplanationContext/types';

export const buildExplanationGraph = async (
  explanation: ParsedExplanation,
  svgRef: RefObject<SVGSVGElement>,
  triples: ExplanationTriple[],
  nodeNameMap: Record<string, string>,
  isMobile: boolean,
) => {
  /*format data for links and nodes from explanation triples*/
  const { linksData, nodeNames } = getLinksDataFromTriples(triples);
  const nodes = constructNodesFromData(
    nodeNames,
    explanation,
    nodeNameMap,
    triples,
  );

  /*graph params*/
  const width = isMobile ? 400 : 800;
  const height = isMobile ? 400 : 450;

  const headRadius = isMobile ? 20 : 25;
  const tailWidth = isMobile ? 35 : 40;
  const middleNodeRadius = isMobile ? 10 : 15;
  const relationRadius = 10;
  const margin = 40;
  const nodeNameGap = 5;

  let tooltipHovered = false;
  let linkHovered = false;
  const nodesByHop = d3.group(
    nodes.filter((n) => n.group === 'middle'),
    (d) => d.hop,
  );
  const maxHops = d3.max(nodes, (d) => d.hop) || 1;

  d3.select(svgRef.current).selectAll('*').remove();

  const svg = d3
    .select(svgRef.current)
    .attr('viewBox', [0, 0, width, height])
    .attr('width', '100%')
    .attr('height', '100%');

  const tooltip = d3
    .select('body')
    .append('div')
    .attr('class', 'explanation-tooltip');

  //the "simulation" grabs target and source from nodes and links them
  const simulation = d3
    .forceSimulation<ExplanationNodeType>(nodes)
    .force(
      'link',
      d3
        .forceLink<ExplanationNodeType, ExplanationLinkType>(linksData)
        .id((node) => node.id)
        .strength(0.2)
        .distance(200),
    )
    .force(
      'charge',
      d3.forceManyBody<ExplanationNodeType>().strength(-300).distanceMax(300),
    ) //repulsion between nodes
    .force('collide', d3.forceCollide().radius(15).strength(0.8))
    .force(
      'x',
      d3 //position along x axis, vertically on mobile and horizontally on desktop
        .forceX((d: ExplanationNodeType) => {
          if (!isMobile) {
            if (d.group === 'head' || d.group === 'tail') return width / 2;
            return positionByHop(d.hop, maxHops, width, margin);
          } else {
            const nodesAtHop = nodesByHop.get(d.hop) || [];
            const idx = nodesAtHop.findIndex((n) => n.id === d.id);
            return distribute(idx, nodesAtHop.length, width, margin);
          }
        })
        .strength(1),
    )
    .force(
      'y',
      d3 //position along x axis, vertically on mobile and horizontally on desktop
        .forceY((d: ExplanationNodeType) => {
          if (!isMobile) {
            const nodesAtHop = nodesByHop.get(d.hop) || [];
            const idx = nodesAtHop.findIndex((n) => n.id === d.id);
            return distribute(idx, nodesAtHop.length, height, margin);
          } else {
            return positionByHop(d.hop, maxHops, height, margin);
          }
        })
        .strength(1),
    )
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.5));

  /*position head and tail nodes fixed, vertically in case of mobile devices and horizontally on desktop*/
  const headNode = nodes.find((node) => node.group === 'head');
  if (headNode) {
    headNode.fx = isMobile ? width / 2 : margin;
    headNode.fy = isMobile ? 0 : height / 2;
  }

  const tailNode = nodes.find((node) => node.group === 'tail');
  if (tailNode) {
    tailNode.fx = isMobile ? width / 2 : width - margin;
    tailNode.fy = isMobile ? height : height / 2;
  }

  /*Create arrows pointing from links to nodes*/
  addArrows(svg, [
    {
      distanceFromNode: middleNodeRadius * 2 + 10,
      id: 'arrowMid',
      width: 10,
      color: 'gray',
    },
    {
      distanceFromNode: headRadius * 2 + 5,
      id: 'arrowTip',
      width: 10,
      color: 'gray',
    },
    {
      distanceFromNode: relationRadius * 2 + 10,
      id: 'arrowRel',
      width: 10,
      color: 'gray',
    },
  ]);

  /*create links*/
  const link = svg
    .append('g')
    .selectAll('.link')
    .data(linksData)
    .enter()
    .append('path')
    .attr('stroke', 'grey')
    .attr('stroke-opacity', 1)
    .attr('fill', 'white')
    .attr('stroke-width', '0.8')
    .attr('marker-end', (d: ExplanationLinkType) => {
      if (
        (d.target as ExplanationNodeType).id === headNode?.id ||
        (d.target as ExplanationNodeType).id === tailNode?.id
      )
        return 'url(#arrowTip)';
      else if ((d.target as ExplanationNodeType).group === 'relation')
        return 'url(#arrowRel)';
      else return 'url(#arrowMid)';
    });

  /*create a transparent link hover area which is wider than the links for better UX*/
  const linkHover = svg
    .append('g')
    .selectAll('.link-hover')
    .data(linksData)
    .enter()
    .append('path')
    .attr('class', 'link-hover')
    .attr('stroke', 'transparent')
    .attr('stroke-width', 35)
    .attr('data-paths', (d) => d.paths?.join(',') || '')
    .style('cursor', 'pointer');

  /*Avoid tooltip disappearing when hovering on it*/
  tooltip
    .on('mouseenter', function () {
      tooltipHovered = true;
      tooltip.classed('active', true);
    })
    .on('mouseleave', function () {
      tooltipHovered = false;
      tooltip.classed('active', false);
    });

  /*Make tooltip appear on link hover and highlight link*/
  linkHover
    .on('mouseover', function (event, d) {
      linkHovered = true;
      const currentPaths: number[] = d.paths || [];
      // Get all links that share at least one path
      const pathLinks = link.filter(
        (d) => d.paths?.some((p) => currentPaths.includes(p)) ?? false,
      );

      const pathLinksData: ExplanationLinkType[] = pathLinks.data();

      const tooltipText = pathLinksData
        .map((pathlink) => {
          const sourceName =
            nodeNameMap[(pathlink.source as ExplanationNodeType).id];
          const targetName =
            nodeNameMap[(pathlink.target as ExplanationNodeType).id];
          return `<li>${sourceName} &#8594; ${pathlink.relationName} &#8594; ${targetName}</li>`;
        })
        .join('');

      pathLinks.each(function () {
        d3.select(this)
          .raise()
          .transition()
          .duration(400)
          .attr('stroke', 'yellowgreen')
          .attr('stroke-width', 1.5);
      });

      tooltip
        .classed('active', true)
        .html(`<strong>Pathway</strong><ul>${tooltipText}</ul>`);
    })
    .on('mousemove', function (event) {
      tooltip
        .style('left', event.pageX + 10 + 'px')
        .style('top', event.pageY + 10 + 'px')
        .classed('active', true);
    })
    .on('mouseleave', function () {
      linkHovered = false;
      setTimeout(() => {
        if (!tooltipHovered && !linkHovered) {
          tooltip.classed('active', false);
        }
      }, 700);
      link
        .transition()
        .duration(500)
        .attr('stroke', 'grey')
        .attr('stroke-width', '0.8');
      linkHover
        .transition()
        .duration(500)
        .attr('stroke', 'transparent')
        .attr('stroke-width', 10);
    });

  /*get relation names*/
  const relationLabels = getRelationLabels(svg, linksData);

  /*create nodes*/
  const node = svg
    .append('g')
    .selectAll('g')
    .data(nodes)
    .enter()
    .append('g')
    .call(
      d3
        .drag<SVGGElement, ExplanationNodeType>()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragended),
    );

  node //squares for concepts
    .filter((node) => node.group === 'tail')
    .append('rect')
    .attr('cursor', 'grab')
    .attr('x', -tailWidth / 2)
    .attr('y', -tailWidth / 2)
    .attr('width', tailWidth)
    .attr('height', tailWidth)
    .attr('rx', 5)
    .attr('ry', 5)
    .attr('fill', 'darkturquoise')
    .attr('brightness', '100%')
    .attr('stroke-width', '0.3')
    .attr('stroke', 'black');

  node //circles for the relations and variable
    .filter((node) => node.group !== 'tail')
    .append('circle')
    .attr('cursor', 'grab')
    .attr('r', (node) =>
      node.group === 'head'
        ? headRadius
        : node.group === 'middle'
          ? middleNodeRadius
          : relationRadius,
    )
    .attr('fill', (node) =>
      node.group === 'head'
        ? 'var(--primary-pink)'
        : node.group === 'relation'
          ? 'yellowgreen'
          : 'azure',
    )
    .attr('stroke-width', '0.3')
    .attr('stroke', 'black');

  //text outside nodes
  node
    .append('text')
    .attr('dx', 0)
    .attr('dy', (node) =>
      node.group === 'middle'
        ? -(middleNodeRadius + nodeNameGap)
        : node.group === 'head'
          ? -(headRadius + nodeNameGap)
          : -(tailWidth / 2 + nodeNameGap),
    )
    .attr('text-anchor', 'middle')
    .text((node) => {
      const cleanName = normalizeName(node.name);
      return cleanName.length > 22 ? cleanName.slice(0, 22) + '...' : cleanName;
    })
    .attr('fill', 'grey')
    .attr('font-size', '11px')
    .attr('font-weight', 'bold');

  //link all
  simulation.on('tick', () => {
    link.attr('d', linkNodes);
    linkHover.attr('d', linkNodes);
    relationLabels
      .attr(
        'x',
        (d: LinkType) =>
          (((d.source as ExplanationNodeType).x ?? 0) +
            ((d.target as ExplanationNodeType).x ?? 0)) /
          2,
      )
      .attr(
        'y',
        (d: LinkType) =>
          (((d.source as ExplanationNodeType).y ?? 0) +
            ((d.target as ExplanationNodeType)?.y ?? 0)) /
          2,
      )
      .attr('padding', '30')
      .attr('dy', '-5px')
      .attr('transform', (d: LinkType) => {
        const sourceX = (d.source as ExplanationNodeType).x ?? 0;
        const sourceY = (d.source as ExplanationNodeType).y ?? 0;
        const targetX = (d.target as ExplanationNodeType).x ?? 0;
        const targetY = (d.target as ExplanationNodeType).y ?? 0;
        const x = (sourceX + targetX) / 2;
        const y = (sourceY + targetY) / 2;
        let angle =
          (Math.atan2(targetY - sourceY, targetX - sourceX) * 180) / Math.PI;
        if (angle > 90 || angle < -90) {
          angle += 180;
        }
        return `rotate(${angle}, ${x}, ${y})`;
      });
    node.attr('transform', (node) => `translate(${node.x},${node.y})`);
  });

  function linkNodes(d: ExplanationLinkType) {
    const sourceX = (d.source as ExplanationNodeType).x ?? 0;
    const sourceY = (d.source as ExplanationNodeType).y ?? 0;
    const targetX = (d.target as ExplanationNodeType).x ?? 0;
    const targetY = (d.target as ExplanationNodeType).y ?? 0;

    return 'M' + sourceX + ',' + sourceY + 'L' + targetX + ',' + targetY;
  }

  function dragStarted(
    event: d3.D3DragEvent<SVGGElement, ExplanationNodeType, unknown>,
    node: ExplanationNodeType,
  ) {
    if (!event.active) simulation.alphaTarget(0.3).restart(); //give movement to all nodes and update during drag
    node.fx = node.x;
    node.fy = node.y;
  }

  function dragged(
    event: d3.D3DragEvent<SVGGElement, ExplanationNodeType, unknown>,
    node: ExplanationNodeType,
  ) {
    node.fx = event.x;
    node.fy = event.y;
  }

  function dragended(
    event: d3.D3DragEvent<SVGGElement, ExplanationNodeType, unknown>,
    node: ExplanationNodeType,
  ) {
    if (!event.active) simulation.alphaTarget(0); //stop movement when event finishes
    node.fx = null;
    node.fy = null;
  }

  return svgRef;
};
