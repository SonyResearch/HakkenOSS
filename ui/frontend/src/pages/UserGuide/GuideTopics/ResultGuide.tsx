import pageLayout from '../../../assets/images/help-guide/candidates-page-layout.png';
import explanationPage from '../../../assets/images/help-guide/explanations-page.png';
import explanationButton from '../../../assets/images/help-guide/explanations-button.png';
import explanationToggle from '../../../assets/images/help-guide/explanations-toggle.png';
import explanationHovered from '../../../assets/images/help-guide/explanation-hover.png';
import contextualization from '../../../assets/images/help-guide/contextualization.png';
import contextualizationFilters from '../../../assets/images/help-guide/contextualization-filters.png';
import scatterPlotToggle from '../../../assets/images/help-guide/scatter-plot-toggle.png';
import scatterPlotVideo from '../../../assets/images/help-guide/scatter-plot.mp4';
import TableOfContents from '../TableOfContents';
import { useMobile } from '../../../hooks/useMobile';

export const resultContents = {
  resultPageLayout: {
    title: 'Candidates Page Layout',
    id: 'page-layout',
  },
  explanation: {
    title: 'Explanation Pathways',
    id: 'explanation-pathways',
  },
  contextualization: {
    title: 'Contextualization',
    id: 'contextualization',
  },
  exploringReferences: {
    title: 'Exploring references',
    id: 'exploringReferences',
  },
};

export const ResultGuide = () => {
  const isMobile = useMobile();
  return (
    <article id="result">
      <h3>Candidates Page</h3>
      <TableOfContents contents={Object.values(resultContents)} />
      <ul>
        <li id={resultContents.resultPageLayout.id}>
          <h4>{resultContents.resultPageLayout.title}</h4>
          <p>
            Once you run your query, you will be brought to the candidates page,
            where you can see all the available information we have about each
            candidate the system returned.
          </p>
          <p></p>
          <img src={pageLayout} alt="Image of the candidates dashboard"></img>
          <ol>
            <li>
              <strong>Query Formula</strong>
              <p>
                The query formula build from your conditions to get the result.
                You can go back and edit the conditions by clicking on the edit
                icon.
              </p>
            </li>
            <li>
              <strong>Candidate List</strong>
              <p>
                You can click on each one of the candidates to see their
                available information, that will be loaded on the section on the
                right
              </p>
            </li>
            <li>
              <strong>Candidate Selected and it`&apos;s Query Score</strong>
              <p>
                The Query score is a value between 0.0 and 1.0, the higher the
                score is to 1, the greater the confidence our system has on the
                result relative to the query.
              </p>
            </li>
            <li>
              <strong>Contextualization</strong>
              <p>
                Here, you can see relevant pieces present in our literature that
                support the relation between our candidate and your query.
              </p>
            </li>
            <li>
              <strong>Explanation Button</strong>
              <p>
                If you are curious about one of the candidates returned, you can
                see it`&apos;s explanation by pressing on this button
              </p>
            </li>
            <li>
              <strong>Query Visualization</strong>
              <p>
                Just as in the search page, you can visualize the query made,
                this time filled with the selected candidate and it`&apos;s
                score or scores (if multiple hypotheses were used to build the
                query)
              </p>
            </li>
          </ol>
        </li>
        <li id={resultContents.explanation.id}>
          <h4>{resultContents.explanation.title}</h4>
          <p>
            If you are curious about the connection between a specific candidate
            and your hypothesis, you can see it`&apos;s explanation pathways by
            clicking on the `&apos;Get Explanation Pathways`&apos; button
          </p>
          <div className="image-container">
            <figure>
              <img
                src={explanationButton}
                alt="button to get explanations"
              ></img>
              <figcaption>Get explanation pathways button</figcaption>
            </figure>
          </div>
          <p>
            Once the explanation is loaded, you can view it either by clicking
            again on the same button, now saying{' '}
            <em>`&apos;View Explanation Pathways`&apos;</em> or by clicking on
            the <em>Explanation Pathways</em> tab on the top right corner of the
            candidate page
          </p>
          <div className="image-container">
            <figure>
              <img
                src={explanationToggle}
                alt="Explanation pathways toggle"
              ></img>
              <figcaption>Explanation pathways toggle</figcaption>
            </figure>
          </div>
          <p>
            In case of having more than one hypotheses to predict, first you
            will have to select the hypothesis you want to get the explanation
            pathwayss from by selecting it from the dropdown
          </p>
          <p>
            Once clicked, you can see the explanation pathways for a given
            hypothesis on the explanation
          </p>

          <figure>
            <img src={explanationPage} alt="Explanation page"></img>
            <figcaption>Explanation pathways section</figcaption>
          </figure>
          <p>
            You can increase the number of pathways by selecting a number on the
            bottom-right range, and highlight a pathway and see its details by
            hovering on it
          </p>
          <figure>
            <img
              src={explanationHovered}
              alt="Explanation pathway hovered"
            ></img>
            <figcaption>Hovered explanation pathway</figcaption>
          </figure>
        </li>
        <li id={resultContents.contextualization.id}>
          <h4>{resultContents.contextualization.title}</h4>
        </li>
        <p>
          On the references section inside the candidates page, you can see the
          contextualization of the selected candidate. This references include
          summaries and abstracts of papers which help contextualize the
          selected candidate according to our body of literature.
        </p>
        <p>
          {' '}
          You can expand on every reference by clicking on the `&apos;Show
          abstract`&apos; button, and even expand further by clicking on the
          corresponding identifier at the bottom of each reference.
        </p>
        <figure>
          <img src={contextualization} alt="contextualization image"></img>
          <figcaption>Contextualization references image</figcaption>
        </figure>
        <li id={resultContents.exploringReferences.id}>
          <h4>{resultContents.exploringReferences.title}</h4>
        </li>
        <p>
          You can organize the references by using the filter options at the top
          of the list. These allow you to sort by year, score, or citation
          count, and to filter results by entering an author’s name, title, or
          any keyword that might appear in the abstract.
        </p>
        <figure>
          <img
            src={contextualizationFilters}
            alt="contextualization filters"
          ></img>
          <figcaption>Contextualization filters image</figcaption>
        </figure>
        <p>
          For hypotheses with numerous references, visualizing the data can make
          analysis easier. Toggle the scatter plot view to compare references
          based on year, citation count, and score.
        </p>
        <figure>
          <img src={scatterPlotToggle} alt="scatter plot toggle button"></img>
          <figcaption>Toggle the scatter plot</figcaption>
        </figure>
        <p>
          Once the scatter plot is displayed, you can hover over each reference
          to view its details. Clicking on a point will take you to the
          corresponding reference in the list.
        </p>
        <figure>
          <video
            height={isMobile ? 250 : 350}
            src={scatterPlotVideo}
            autoPlay
            loop
            muted
            playsInline
          ></video>
          <figcaption>Exploring contextualization scatter plot</figcaption>
        </figure>
      </ul>
    </article>
  );
};
