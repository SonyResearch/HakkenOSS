import React, { SetStateAction } from 'react';
import './index.css';

export const ErrorBox = ({
  error,
  setError,
}: {
  error: { message: string; level: 'high' | 'low' };
  setError: React.Dispatch<
    SetStateAction<{ message: string; level: 'high' | 'low' } | null>
  >;
}) => {
  return (
    <div className={`error-modal`}>
      <div className={`error-box ${error.level}`}>
        {error.message}
        <button onClick={() => setError(null)}>Ok</button>
      </div>
      ;
    </div>
  );
};

export default ErrorBox;
