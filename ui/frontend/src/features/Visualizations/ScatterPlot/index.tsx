/*Scatter plot to visualize contextualization results*/

import './index.css';
import { ContextualizationResult } from '../../CandidateDashboard/types';
import * as d3 from 'd3';
import { forwardRef, useEffect, useRef } from 'react';
import { getPlotData } from './utils';
import { useNavigate } from 'react-router-dom';

type ScatterPlotProps = {
  contextualization: ContextualizationResult;
  handleReferenceClick: () => void;
};

const ContextualizationScatterPlot = forwardRef<
  HTMLDivElement,
  ScatterPlotProps
>(({ contextualization, handleReferenceClick }, ref) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const navigate = useNavigate();

  const data = getPlotData(contextualization.references);

  const margin = 100;
  const marginTop = 50;

  const svgWidth = 750;
  const svgHeight = 750;

  const width = svgWidth - margin * 2;
  const height = svgHeight - margin * 2;

  useEffect(() => {
    const svg = d3.select(svgRef.current);

    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([1960, 2025]).range([0, width]);

    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    const xAxis = d3.axisBottom(xScale).tickFormat(d3.format('d'));

    const yAxis = d3.axisLeft(yScale);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin}, ${marginTop})`);

    // X Axis
    g.append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(xAxis)
      .append('text')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Recency');

    // Y Axis
    g.append('g')
      .call(yAxis)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Score');

    // Scatter plot points
    const referenceCircles = g
      .selectAll('.candidate')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'candidate-circle')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', 10)
      .attr('id', (d) => d.id)
      .attr('stroke', 'gray')
      .attr('strokeWidth', 0.5)
      .style('fill-opacity', (d) => `${d.intensity * 2}%`);

    // Legend
    const legendHeight = 20;

    const legend = g
      .append('g')
      .attr('transform', `translate(${width - width / 4}, ${height + 70})`);

    const defs = svg.append('defs');

    const gradient = defs
      .append('linearGradient')
      .attr('id', 'recency-gradient');

    for (let i = 0; i <= 4; i++) {
      gradient
        .append('stop')
        .attr('offset', `${(i / 4) * 100}%`)
        .attr('stop-color', `rgba(25,118,210,${i / 4})`);
    }

    legend
      .append('rect')
      .attr('width', width / 4)
      .attr('height', legendHeight)
      .style('fill', 'url(#recency-gradient)')
      .style('stroke', '#333');

    legend
      .append('text')
      .attr('x', 0)
      .attr('y', legendHeight + 15)
      .text('Low')
      .style('font-size', '10px');

    legend
      .append('text')
      .attr('x', width / 8)
      .attr('y', legendHeight + 25)
      .text('Citations count')
      .style('font-size', '14px')
      .style('text-anchor', 'middle');

    legend
      .append('text')
      .attr('x', width / 4)
      .attr('y', legendHeight + 15)
      .text('High')
      .style('font-size', '10px')
      .style('text-anchor', 'end');

    // Tooltip displaying reference info
    d3.select('.scatter-plot-tooltip').remove();

    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'scatter-plot-tooltip');

    referenceCircles
      .on('mouseenter', function (event, reference) {
        d3.select(this).classed('active', true);

        tooltip
          .classed('active', true)
          .style('left', `${event.pageX + 20}px`)
          .style('top', `${event.pageY}px`)
          .html(
            `<h4>${reference.title}</h4>
             <p>Year: ${reference.x}</p>
             <p>Query Score: ${reference.y.toFixed(2)}</p>
             <span>${reference.citationsCount ? 'Citations count: ' + reference.citationsCount : ''}</span>`,
          );
      })
      .on('mouseleave', () => {
        d3.selectAll('.candidate-circle').classed('active', false);
        tooltip.classed('active', false);
      })
      .on('click', function (event, reference) {
        const id = `reference-${reference.id}`;

        navigate({ hash: id });

        document.getElementById(id)?.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });

        handleReferenceClick();
      });
  }, [data, navigate, handleReferenceClick]);

  return (
    <div ref={ref} className="scatter-plot-container">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        style={{
          maxWidth: '900px',
          height: 'auto',
          display: 'block',
        }}
      />
    </div>
  );
});
ContextualizationScatterPlot.displayName = 'ContextualizationScatterPlot';
export default ContextualizationScatterPlot;
