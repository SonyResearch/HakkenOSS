import './index.css';
import { Button, Collapse } from '@mui/material';
import { useState } from 'react';
import { ExplainabilityGraph } from '../../../Visualizations/ExplainabilityGraph';
import { ParsedExplanation } from '../../../../contexts/ExplanationContext/types';

export const ExplanationRow = ({
  explanation,
  collapsed,
  isNegated,
  isInValidation,
}: {
  explanation: ParsedExplanation | undefined;
  collapsed: boolean;
  isNegated: boolean;
  isInValidation: boolean;
}) => {
  const [nodeNameMap, setNodeNameMap] = useState<Record<string, string>>({});
  const [showTooltip, setShowTooltip] = useState<boolean>(false);
  if (!explanation) return null;
  const entities = explanation.queryId.split('-');
  const namedCondition = entities
    .map((entity) => ` ${nodeNameMap[entity] || entity}`)
    .join('-');
  const [isCollapsed, setIsCollapsed] = useState<boolean>(collapsed);
  return (
    <div className="explanation-row">
      <div className="explanation-title">
        <h4>
          {namedCondition}{' '}
          {isNegated && (
            <div
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              className="exclamation-icon"
            >
              !
            </div>
          )}
        </h4>
        {!isInValidation && (
          <Button
            variant="outlined"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? 'Show' : 'Hide'}
          </Button>
        )}
      </div>
      <Collapse in={!isCollapsed}>
        {explanation.explanations.length ? (
          <ExplainabilityGraph
            selectedExplanation={explanation}
            nodeNameMap={nodeNameMap}
            setNodeNameMap={setNodeNameMap}
            isInValidation={isInValidation}
          />
        ) : (
          <p>
            Sorry, our system could not find an explanation for this hypothesis
          </p>
        )}
      </Collapse>
      {showTooltip && (
        <div className="not-tooltip">
          At this time, our explanation engine cannot process negated
          relationships. We generated an explanation for the equivalent positive
          condition.
        </div>
      )}
    </div>
  );
};

export default ExplanationRow;
