import { LinkType } from './types';

//will reuse on explainability visualization

type ArrowType = {
  distanceFromNode: number;
  id: string;
  color: string;
  width: number;
};

export const addArrows = (
  svg: d3.Selection<SVGSVGElement | null, unknown, null, undefined>,
  arrowType: ArrowType[],
) => {
  arrowType.forEach((arrow) =>
    svg
      .append('defs')
      .append('marker')
      .attr('id', arrow.id)
      .attr('viewBox', '0 -5 10 15')
      .attr('refX', arrow.distanceFromNode)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerHeight', 10)
      .attr('markerWidth', arrow.width)
      .append('path')
      .attr('d', 'M0, -5L10, 0L0, 5')
      .attr('fill', arrow.color)
      .attr('stroke', arrow.color)
      .attr('stroke-width', 1.5),
  );
};

export const getRelationLabels = (
  svg: d3.Selection<SVGSVGElement | null, unknown, null, undefined>,
  linksData: LinkType[],
) => {
  const relationLabels = svg
    .append('g')
    .attr('class', 'relation-labels')
    .selectAll('text')
    .data(linksData)
    .enter()
    .append('text')
    .attr('text-anchor', 'middle')
    .attr('fill', 'grey')
    .attr('font-size', '0.5rem')
    .attr('font-weight', 'bold')
    .text((d) => (d.relationName ? d.relationName : ''));

  return relationLabels;
};
