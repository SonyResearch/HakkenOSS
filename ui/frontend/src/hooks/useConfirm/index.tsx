/*Hook to create a custom confirm modal*/

import { useState, useCallback } from 'react';
import ConfirmModal from '../../shared/components/ConfirmModal';

export const useConfirm = () => {
  const [options, setOptions] = useState<{
    message: string[];
    resolve: (value: unknown) => void;
  } | null>(null);

  const confirm = useCallback((message: string[]) => {
    return new Promise((resolve) => {
      setOptions({
        message,
        resolve,
      });
    });
  }, []);

  const handleConfirm = () => {
    if (options) {
      options.resolve(true);
      setOptions(null);
    }
  };

  const handleCancel = () => {
    if (options) {
      options.resolve(false);
      setOptions(null);
    }
  };

  const ConfirmDialog = (
    <ConfirmModal
      open={!!options}
      message={options?.message || []}
      onConfirm={handleConfirm}
      onCancel={handleCancel}
    />
  );

  return { confirm, ConfirmDialog };
};
