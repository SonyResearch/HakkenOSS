import React, { SetStateAction } from 'react';
import './index.css';

interface TabNavProps<T extends string> {
  options: Record<T, boolean>;
  currentView: T;
  setView: React.Dispatch<SetStateAction<T>>;
}

export const TabNav = <T extends string>({
  options,
  setView,
  currentView,
}: TabNavProps<T>) => {
  return (
    <nav className="tab-nav">
      {(Object.entries(options) as [T, boolean][]).map(
        ([option, value]) =>
          value && (
            <button
              key={option}
              onClick={() => setView(option)}
              className={currentView === option ? 'active' : ''}
            >
              {option[0].toUpperCase() + option.slice(1)}
            </button>
          ),
      )}
    </nav>
  );
};
export default TabNav;
