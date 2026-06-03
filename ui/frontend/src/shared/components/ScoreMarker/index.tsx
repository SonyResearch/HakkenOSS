const ScoreMarker = ({
  score,
  size,
}: {
  score: number;
  size: 'big' | 'small';
}) => {
  const width = size === 'big' ? 140 : 120;
  const height = size === 'big' ? 25 : 20;
  const strokeWidth = 3;
  return (
    <div className="score-marker-container">
      <span className="score-range">0</span>
      <svg width={width} height={height}>
        <rect
          height={height}
          width={width}
          fill="none"
          strokeWidth={strokeWidth}
          stroke="grey"
        ></rect>
        <rect
          height={height - strokeWidth}
          x={strokeWidth / 2}
          y={strokeWidth / 2}
          width={score * width - strokeWidth}
          fill={'var(--primary-pink)'}
          opacity={score}
        ></rect>
        {size === 'big' && (
          <text
            fill={score > 0.75 ? 'var(--light-gray)' : 'black'}
            fontWeight={600}
            textAnchor="middle"
            x={width / 2}
            y={height / 2 + 4}
            fontSize={'16px'}
          >
            {score.toString().slice(0, 7)}
          </text>
        )}
      </svg>
      <span className="score-range">1</span>
    </div>
  );
};

export default ScoreMarker;
