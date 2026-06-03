import {
  formatTripleForConstraintRequest,
  getFieldsFromConstraintResponse,
  parseExplanationStringtoTriples,
  parsePredictedTripleStringToTriple,
  snakeToCamel,
} from '../../../src/services/converters';
import { expect, describe, it } from 'vitest';
import { ConstraintsFilteringResponse } from '../../../src/static/datasets/data-types';
import { PredictionType } from '../../../src/contexts/QueryContext/types';

describe('snakeToCamel', () => {
  it('Should return the object keys in camelCase in case of a single object', () => {
    const arr = [
      { some_variable: 1 },
      { some_other_variable: 2 },
      { constant: 3 },
    ];
    const parsed = snakeToCamel(arr);
    expect(parsed).toStrictEqual([
      { someVariable: 1 },
      { someOtherVariable: 2 },
      { constant: 3 },
    ]);
  });
  it('Should also return the object keys in camelCase in case of a nested object', () => {
    const arr = [
      [{ some_variable: 1 }, { some_other_variable: 2 }],
      [{ constant: 3 }],
    ];
    const parsed = snakeToCamel(arr);
    expect(parsed).toStrictEqual([
      [{ someVariable: 1 }, { someOtherVariable: 2 }],
      [{ constant: 3 }],
    ]);
  });
});

describe('parseExplanationStringToTriples', () => {
  it('should return an array with parsed triples if the explanation format is valid', () => {
    const correctExplanationString =
      '[Gene1-ASSOCIATE->Gene2] <> [Gene2-INHIBIT->Gene3]';
    const parsedTriples = parseExplanationStringtoTriples(
      correctExplanationString,
    );
    expect(parsedTriples).toStrictEqual([
      { head: 'Gene1', relation: 'ASSOCIATE', tail: 'Gene2' },
      { head: 'Gene2', relation: 'INHIBIT', tail: 'Gene3' },
    ]);
  });
  it('Should throw an error if the explanation is not well formatted', () => {
    const incorrectExplanationString =
      '[Gene1-ASSOCIATE->Gene2] incorrect [Gene2-INHIBIT->Gene3]';
    expect(() =>
      parseExplanationStringtoTriples(incorrectExplanationString),
    ).toThrowError(/Invalid/);
  });
});

describe('parsePredictedTripleStringToTriple', () => {
  it('should return the predicted triple if the string format is valid', () => {
    const correctPredictedTripleString = 'chemical1 - [TREAT] -> disease2';
    const parsedTriple = parsePredictedTripleStringToTriple(
      correctPredictedTripleString,
    );
    expect(parsedTriple).toStrictEqual({
      head: 'chemical1',
      relation: 'TREAT',
      tail: 'disease2',
    });
  });
  it('should throw an error if the string is invalid', () => {
    const incorrectPredictedTripleString = 'CHEMICAL1 - [TREAT] - DISEASE2';
    expect(() =>
      parsePredictedTripleStringToTriple(incorrectPredictedTripleString),
    ).toThrowError(/Invalid/);
  });
});

describe('getFieldsFromConstraintResponse', () => {
  let mockConstraintResponse: ConstraintsFilteringResponse[] = [
    {
      variable: 'R',
      type: 'relation',
      values: [
        { label: 'TREATS', identifier: '1231' },
        { label: 'ASSOCIATE', identifier: '234' },
      ],
    },
    {
      variable: 'X',
      type: 'concept',
      values: [
        { label: 'concept1', identifier: '345' },
        { label: 'concept2', identifier: '456' },
      ],
    },
  ];

  it('should return relation labels and concept names as string arrays stored in possibleRelations and possibleConceptNames variables respectively', () => {
    const { possibleRelations, possibleConceptNames } =
      getFieldsFromConstraintResponse(
        mockConstraintResponse,
        PredictionType.OBJECT,
      );
    expect(possibleRelations).toStrictEqual(['TREATS', 'ASSOCIATE']);
    expect(possibleConceptNames).toStrictEqual(['concept1', 'concept2']);
  });
  it('should return relation labels in an array and a empty concept array if the variable returned doesn`t match the prediction type', () => {
    const { possibleRelations, possibleConceptNames } =
      getFieldsFromConstraintResponse(
        mockConstraintResponse,
        PredictionType.SUBJECT,
      );
    expect(possibleRelations).toStrictEqual(['TREATS', 'ASSOCIATE']);
    expect(possibleConceptNames).toStrictEqual([]);
  });
  it('should return two empty strings if the constraint response is empty', () => {
    mockConstraintResponse = [];
    const { possibleRelations, possibleConceptNames } =
      getFieldsFromConstraintResponse(
        mockConstraintResponse,
        PredictionType.SUBJECT,
      );
    expect(possibleRelations).toStrictEqual([]);
    expect(possibleConceptNames).toStrictEqual([]);
  });
});

describe('formatTripleForConstraintRequest', () => {
  const subject = { id: 'Subject', domain: 'CHEMICAL' };
  let relation = 'INHIBIT';
  let object = { id: 'Object', domain: 'GENE' };
  it('Should return all triple information if none of subject, object or relation value is empty', () => {
    const parsedFullTriple = formatTripleForConstraintRequest(
      subject,
      relation,
      object,
    );
    const expectedTriple = {
      subject: { value: 'Subject', is_variable: false, domain: 'CHEMICAL' },
      relation: { value: 'INHIBIT', is_variable: false },
      object: { value: 'Object', is_variable: false, domain: 'GENE' },
    };
    expect(parsedFullTriple).toStrictEqual(expectedTriple);
  });
  it('Should format data with variable character as values for empty relation and entity labels', () => {
    relation = '';
    object = { ...object, id: '' };
    const parsedPartialTriple = formatTripleForConstraintRequest(
      subject,
      relation,
      object,
    );
    const expectedTriple = {
      subject: { value: 'Subject', is_variable: false, domain: 'CHEMICAL' },
      relation: { value: 'R', is_variable: true },
      object: { value: 'Y', is_variable: true, domain: 'GENE' },
    };
    expect(parsedPartialTriple).toStrictEqual(expectedTriple);
  });
  it('should not include the domain key if the domain is still undefined', () => {
    object = { id: '', domain: '' };
    const parsedTripleWithoutObjDomain = formatTripleForConstraintRequest(
      subject,
      relation,
      object,
    );
    const expectedTriple = {
      subject: { value: 'Subject', is_variable: false, domain: 'CHEMICAL' },
      relation: { value: 'R', is_variable: true },
      object: { value: 'Y', is_variable: true },
    };
    expect(parsedTripleWithoutObjDomain).toStrictEqual(expectedTriple);
  });
});
