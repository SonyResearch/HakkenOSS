import { describe, it, expect, vi } from 'vitest';
import { editAndShiftConditions } from '../../../src/features/QueryForm/utils';
import {
  AddValue,
  ConditionType,
} from '../../../src/contexts/QueryContext/types';

describe('editAndShiftConditions', () => {
  const mockCondition = (id: string) => ({
    condition: {
      head: {
        label: id,
        domain: 'disease',
        id: '1234',
        isVariable: true,
      },
      relation: 'ASSOCIATE',
      tail: {
        label: 'x',
        domain: 'protein',
        id: '1234',
        isVariable: false,
      },
    },
    conditionType: ConditionType.hypotheses,
    addValue: AddValue.AND,
  });

  describe('Editing with sequential indexes', () => {
    it('should return old conditions if newConditions is empty', () => {
      const conditions = {
        0: mockCondition('A'),
        1: mockCondition('B'),
      };

      const result = editAndShiftConditions(conditions, {});
      expect(result).toEqual(conditions);
    });

    it('should replace the last condition and add an extra one', () => {
      const conditions = {
        0: mockCondition('A'),
        1: mockCondition('B'),
      };
      const newConditions = {
        1: mockCondition('C'),
        2: mockCondition('D'),
      };
      const result = editAndShiftConditions(conditions, newConditions);

      expect(result).toEqual({
        0: mockCondition('A'),
        1: mockCondition('C'),
        2: mockCondition('D'),
      });
    });

    it('should replace the condition and add two extra ones', () => {
      const conditions = {
        0: mockCondition('A'),
      };
      const newConditions = {
        0: mockCondition('B'),
        1: mockCondition('C'),
      };
      const result = editAndShiftConditions(conditions, newConditions);

      expect(result).toEqual({
        0: mockCondition('B'),
        1: mockCondition('C'),
      });
    });
  });

  describe('Editing with index with gaps', () => {
    it('should replace the middle condition and add an extra one', () => {
      const conditions = {
        0: mockCondition('A'),
        '6': mockCondition('B'),
        '7': mockCondition('C'),
      };
      const newConditions = {
        '6': mockCondition('D'),
        '7': mockCondition('E'),
      };
      const result = editAndShiftConditions(conditions, newConditions);

      expect(result).toEqual({
        0: mockCondition('A'),
        1: mockCondition('D'),
        2: mockCondition('E'),
        3: mockCondition('C'),
      });
    });

    it('should replace the first condition and add two extra ones', () => {
      const conditions = {
        0: mockCondition('A'),
        '6': mockCondition('B'),
      };
      const newConditions = {
        0: mockCondition('D'),
        1: mockCondition('E'),
        2: mockCondition('F'),
      };
      const result = editAndShiftConditions(conditions, newConditions);

      expect(result).toEqual({
        0: mockCondition('D'),
        1: mockCondition('E'),
        2: mockCondition('F'),
        3: mockCondition('B'),
      });
    });
  });
});
