import { ExplanationNodeType, ExplanationTriple } from './types';
import {
  ParsedExplanationItem,
  ParsedExplanation,
} from '../../../contexts/ExplanationContext/types';

export const getTriplesFromExplanationItem = (
  explanationItems: ParsedExplanationItem[],
  numExplanations?: number,
  setMaxNumberOfExplanations?: React.Dispatch<React.SetStateAction<number>>,
): ExplanationTriple[] => {
  const tripleMap = new Map<string, ExplanationTriple>();
  let acceptedGroups = 0;
  let totalGroupsWithNewTriples = 0;

  for (const item of explanationItems) {
    const triplesInGroup: ExplanationTriple[] = [];
    let hasNewTriples = false;

    item.data.forEach((triple, i) => {
      const key = [triple.head, triple.tail].sort().join('--');
      const hop = i + 1;

      if (!tripleMap.has(key)) {
        const newTriple = { ...triple, hop, paths: [] };
        tripleMap.set(key, newTriple);
        triplesInGroup.push(newTriple);
        hasNewTriples = true;
      } else {
        const existingTriple = tripleMap.get(key);
        if (existingTriple) {
          triplesInGroup.push(existingTriple);
        }
      }
    });

    if (hasNewTriples) {
      totalGroupsWithNewTriples++;

      if (numExplanations === undefined || acceptedGroups < numExplanations) {
        const pathIndex = acceptedGroups;

        triplesInGroup.forEach((t) => {
          if (!t.paths?.includes(pathIndex)) {
            t.paths?.push(pathIndex);
          }
        });

        acceptedGroups++;
      }
    }
  }

  if (setMaxNumberOfExplanations) {
    setMaxNumberOfExplanations(totalGroupsWithNewTriples);
  }

  const maxPath = numExplanations ?? totalGroupsWithNewTriples;

  return Array.from(tripleMap.values())
    .map((triple) => ({
      ...triple,
      paths: triple.paths?.filter((p) => p < maxPath),
    }))
    .filter((triple) => triple.paths?.length ?? 0 > 0);
};

export const getLinksDataFromTriples = (triples: ExplanationTriple[]) => {
  const linksData = triples.map((triple, index) => ({
    source: triple.head,
    target: triple.tail,
    relationName: triple.relation,
    paths: triple.paths,
    id: index,
    hop: triple.hop,
  }));

  const nodeNames = new Set<string>();
  linksData.forEach(({ source, target }) => {
    if (typeof source === 'string') nodeNames.add(source);
    if (typeof target === 'string') nodeNames.add(target);
  });

  return { linksData, nodeNames };
};

export const constructNodesFromData = (
  nodeNames: Set<string>,
  explanation: ParsedExplanation,
  nodeNameMap: Record<string, string> = {},
  triples: ExplanationTriple[], // we need this to compute hop per node
): ExplanationNodeType[] => {
  const nodeIds = Array.from(nodeNames);

  const nodeHopMap = new Map<string, number>();

  for (const { head, tail, hop } of triples) {
    const currentHeadHop = nodeHopMap.get(head);
    if (currentHeadHop === undefined || hop < currentHeadHop) {
      nodeHopMap.set(head, hop);
    }

    const currentTailHop = nodeHopMap.get(tail);
    if (currentTailHop === undefined || hop < currentTailHop) {
      nodeHopMap.set(tail, hop);
    }
  }

  return nodeIds.map((id) => ({
    id,
    name: nodeNameMap[id] || id,
    hop: nodeHopMap.get(id) ?? 0,
    group:
      id === explanation.predictedTriple.head
        ? 'head'
        : id === explanation.predictedTriple.tail
          ? 'tail'
          : 'middle',
  }));
};

export function normalizeName(name: string) {
  return name.replace(/\s+/g, ' ').trim(); // some names have lots of whitespaces
}

export const distribute = (
  index: number,
  count: number,
  size: number,
  margin: number,
) => {
  const space = size - 2 * margin;
  const spacing = space / (count + 1);
  return margin + spacing * (index + 1);
};

export const positionByHop = (
  hop: number,
  maxHop: number,
  size: number,
  margin: number,
) => {
  const space = size - 2 * margin;
  const spacing = space / (maxHop + 1);
  return margin + spacing * (hop + 1);
};
