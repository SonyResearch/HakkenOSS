import './index.css';
import { searchContents } from '../GuideTopics/SearchGuide';
import { resultContents } from '../GuideTopics/ResultGuide';

import { useEffect, useState } from 'react';
import { validationContents } from '../GuideTopics/ValidationGuide';

type GuideSections = 'search' | 'validation' | 'result';

export const GuideNavigation = () => {
  const [activeSection, setActiveSection] = useState('search');
  const sections: GuideSections[] = ['search', 'validation', 'result'];
  const subsections = {
    search: Object.values(searchContents),
    result: Object.values(resultContents),
    validation: Object.values(validationContents),
  };
  const [activeSubsection, setActiveSubsection] = useState(
    subsections.search[0].id,
  );

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          console.log('entry', entry.target.id);
          if (entry.isIntersecting) {
            if (
              subsections.search
                .map((subsection) => subsection.id)
                .includes(entry.target.id)
            ) {
              setActiveSection('search');
              setActiveSubsection(entry.target.id);
            } else if (
              subsections.validation
                .map((subsection) => subsection.id)
                .includes(entry.target.id)
            ) {
              setActiveSection('validation');
              setActiveSubsection(entry.target.id);
            } else {
              setActiveSection('result');
              setActiveSubsection(entry.target.id);
            }
          }
        });
      },
      { threshold: 0.4 },
    );
    sections.forEach((sectionId) => {
      const sectionEl = document.getElementById(sectionId);
      if (sectionEl) {
        observer.observe(sectionEl);
      }
    });
    [
      ...subsections.search,
      ...subsections.validation,
      ...subsections.result,
    ].forEach((subsection) => {
      const subsectionEl = document.getElementById(subsection.id);
      if (subsectionEl) {
        observer.observe(subsectionEl);
      }
    });

    return () => observer.disconnect();
  }, []);

  return (
    <nav className="guide-navigation">
      {sections.map((section, index) => (
        <div key={index}>
          <a
            onClick={() => setActiveSection(`${section}`)}
            className={`${activeSection === `${section}` ? 'active' : ''} main-section`}
            href={`#${section}`}
          >
            {section.charAt(0).toUpperCase() + section.slice(1)}
          </a>
          <ul>
            {subsections[section] &&
              subsections[section].map((searchSubsection) => (
                <li key={searchSubsection.id}>
                  <a
                    onClick={() => setActiveSection(searchSubsection.id)}
                    href={`#${searchSubsection.id}`}
                    className={
                      activeSubsection === searchSubsection.id ? 'active' : ''
                    }
                  >
                    {searchSubsection.title}
                  </a>
                </li>
              ))}
          </ul>
        </div>
      ))}
    </nav>
  );
};
