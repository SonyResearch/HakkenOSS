import './index.css';

import { GuideNavigation } from './GuideNavigation';
import { ResultGuide } from './GuideTopics/ResultGuide';
import { SearchGuide } from './GuideTopics/SearchGuide';
import { ValidationGuide } from './GuideTopics/ValidationGuide';

const UserGuide = () => {
  return (
    <section className="user-guide-container">
      <h1>User Guide</h1>
      <GuideNavigation />
      <div className="user-guide">
        <SearchGuide />
        <ValidationGuide />
        <ResultGuide />
      </div>
    </section>
  );
};

export default UserGuide;
