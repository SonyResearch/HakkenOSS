import { describe, expect, vi, it } from 'vitest';
import { getPossibleTriplesByPair } from '../../../src/features/Validation/utils';
import { filterOptionsOfRelations } from '../../../src/features/QueryForm/components/MainForm/utils';
import { V } from 'vitest/dist/chunks/reporters.d.BFLkQcL6.js';
import { filter } from 'd3';

vi.mock('../../../src/features/QueryForm/components/MainForm/utils', () => ({
  filterOptionsOfRelations: vi.fn(),
}));

describe('getPossibleTriplesByPair', () => {
  const mockRelations = ['ASSOCIATE', 'TREAT', 'INHIBIT'];
  const entity1 = {
    id: '123',
    domain: 'DISEASE',
    name: 'Alzheimer',
  };
  const entity2 = {
    id: '234',
    domain: 'DISEASE',
    name: 'Parkinson',
  };

  it('Should return triples with all possible relations between two entities', async () => {
    vi.mocked(filterOptionsOfRelations).mockResolvedValue(mockRelations);
    const triples = await getPossibleTriplesByPair(entity1, entity2);
    expect(triples).toStrictEqual([
      { triple: [entity1, 'ASSOCIATE', entity2], score: undefined },
      { triple: [entity1, 'TREAT', entity2], score: undefined },
      { triple: [entity1, 'INHIBIT', entity2], score: undefined },
    ]);
  });
  it('SHould return no triples if filter relations response is empty', async () => {
    vi.mocked(filterOptionsOfRelations).mockResolvedValue([]);
    const triples = await getPossibleTriplesByPair(entity1, entity2);
    expect(triples).toStrictEqual([]);
  });
});
