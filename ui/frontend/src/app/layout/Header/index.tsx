import './index.css';

import homeIcon from '../../../assets/images/icons/house-solid.svg';
import { MainNavigation } from '../MainNavigation';

const Header = () => {
  return (
    <div className="header">
      <div>
        <a href="/">
          <img
            className="header-icon"
            src={homeIcon}
            alt="icon to go back to home"
          ></img>
        </a>
        <div className="logo">Hakken</div>
      </div>
      <MainNavigation />
    </div>
  );
};

export default Header;
