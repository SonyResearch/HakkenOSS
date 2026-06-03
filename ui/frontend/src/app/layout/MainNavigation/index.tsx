import './index.css';
import { NavLink } from 'react-router-dom';
import { useQueryContext } from '../../../contexts/QueryContext';
import { HamburguerMenu } from '../../../shared/components/HamburguerMenu';
import { useState } from 'react';

export const MainNavigation = () => {
  const { candidatesResult } = useQueryContext();
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);

  return (
    <>
      <nav className="main-navigation">
        <NavLink
          className={({ isActive }) => (isActive ? 'active' : '')}
          to="/"
        >
          QUERY
        </NavLink>
        {candidatesResult.candidates.length > 0 && (
          <NavLink
            className={({ isActive }) => (isActive ? 'active' : '')}
            to="/query/results"
          >
            RESULTS
          </NavLink>
        )}
        <NavLink
          className={({ isActive }) => (isActive ? 'active' : '')}
          to="/validation"
        >
          {' '}
          VALIDATE{' '}
        </NavLink>
        <NavLink
          className={({ isActive }) => (isActive ? 'active' : '')}
          to="/user-guide"
        >
          USER GUIDE
        </NavLink>
        {/*<NavLink
        className={({ isActive }) => (isActive ? 'active' : '')}
        to="/faq"
      >
        FAQ
      </NavLink>*/}
      </nav>
      {isDropdownOpen && (
        <nav className="main-navigation-mobile">
          <NavLink
            className={({ isActive }) => (isActive ? 'active' : '')}
            to="/"
          >
            QUERY
          </NavLink>
          {candidatesResult.candidates.length > 0 && (
            <NavLink
              className={({ isActive }) => (isActive ? 'active' : '')}
              to="/query/results"
            >
              RESULTS
            </NavLink>
          )}
          <NavLink
            className={({ isActive }) => (isActive ? 'active' : '')}
            to="/validation"
          >
            {' '}
            VALIDATE{' '}
          </NavLink>
          <NavLink
            className={({ isActive }) => (isActive ? 'active' : '')}
            to="/user-guide"
          >
            USER GUIDE
          </NavLink>
          {/*<NavLink
        className={({ isActive }) => (isActive ? 'active' : '')}
        to="/faq"
      >
        FAQ
      </NavLink>*/}
        </nav>
      )}
      <HamburguerMenu
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        active={isDropdownOpen}
      />
    </>
  );
};
