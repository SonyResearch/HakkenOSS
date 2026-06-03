import { CandidateResultType, Reference } from '../../CandidateDashboard/types';

interface ExpandedCandidateResult extends CandidateResultType {
  recency: number;
  complexity: number;
}

export function getCandidateByOcid(
  candidatesData: ExpandedCandidateResult[],
  id: string,
) {
  return candidatesData.find(
    (candidate) => candidate.variableAssignments.x === id,
  );
}

export function getPlotData(references: Reference[]) {
  const data = references.map((reference) => {
    const citationsCount =
      reference.publication_info.citations_count === 'None'
        ? 0
        : reference.publication_info.citations_count;
    return {
      title: reference.publication_info.title,
      id: reference.publication_info.publication_id,
      x: reference.publication_info.year,
      y: reference.score,
      citationsCount: citationsCount,
      intensity: (citationsCount * 50) / 100,
    };
  });
  return data;
}
