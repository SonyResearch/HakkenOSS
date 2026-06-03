import React, { useEffect, useState } from 'react';
import './index.css';

const ProgressBar = ({
  initialTime,
  remainingTime,
}: {
  initialTime: number;
  remainingTime: number;
}) => {
  const progress = ((initialTime - remainingTime) / initialTime) * 100;
  return (
    <div className="progress-bar">
      <div
        style={{ width: `${Math.min(progress, 99)}%` }}
        className="progress"
      ></div>
    </div>
  );
};

export const LoadingBanner = ({
  initialTime,
  remainingTime,
  texts,
}: {
  initialTime: number;
  remainingTime: number;
  texts: string[];
}) => {
  const [currentText, setCurrentText] = useState<number>(0);

  useEffect(() => {
    const interval = initialTime / texts.length;
    const timer = setInterval(() => {
      setCurrentText((prevText) =>
        prevText < texts.length - 1 ? prevText + 1 : prevText,
      );
    }, interval);

    return () => {
      clearInterval(timer);
    };
  }, [remainingTime]);

  return (
    <div className="loading-banner">
      <>
        <h4>Generating the explanation</h4>
        <p>{texts[currentText]}</p>
        <ProgressBar initialTime={initialTime} remainingTime={remainingTime} />
      </>
    </div>
  );
};
