import operatorsTwoImg from '../../../assets/images/help-guide/operators-2.png';
import historyImg from '../../../assets/images/help-guide/history.jpg';
import editIcon from '../../../assets/images/icons/edit-regular.svg';
import duplicateIcon from '../../../assets/images/icons/copy-regular.svg';
import deleteIcon from '../../../assets/images/icons/trash-can-regular.svg';
import tripleVisualization from '../../../assets/images/help-guide/triple-visualization.jpg';
import domainSelectionImg from '../../../assets/images/help-guide/domain-selection.png';
import domainSelectedImg from '../../../assets/images/help-guide/domain-selected.png';
import predictionSelectionImg from '../../../assets/images/help-guide/prediction-type-selection.png';
import relationSelection from '../../../assets/images/help-guide/relation-selection.png';
import conceptDomainSelection from '../../../assets/images/help-guide/concept-domain-selection.png';
import conceptSelection from '../../../assets/images/help-guide/concept-selection.png';
import addButtonImg from '../../../assets/images/help-guide/add-button.png';
import conditionImg from '../../../assets/images/help-guide/condition.png';
import multipleConditionsImg from '../../../assets/images/help-guide/multiple-conditions.png';
import variableSelectionImg from '../../../assets/images/help-guide/variable-selection.png';
import iconBoxImg from '../../../assets/images/help-guide/conditions-icon-box.png';
import resetButton from '../../../assets/images/help-guide/reset-button.png';
import queryModeToggle from '../../../assets/images/help-guide/query-mode-toggle.png';
import showConstraints from '../../../assets/images/help-guide/show-constraints.png';
import enteringQueryVideo from '../../../assets/images/help-guide/entering-query.mp4';
import searchingButton from '../../../assets/images/help-guide/searching-button.png';
import TableOfContents from '../TableOfContents';

export const searchContents = {
  querySyntax: { title: 'Understanding Query Syntax', id: 'query-syntax' },
  singleVsCombo: {
    title: 'Single Shot/Combo',
    id: 'single-vs-combo',
  },
  firstQuery: {
    title: 'Adding First Condition',
    id: 'first-query',
  },
  multipleQueries: {
    title: 'Adding Multiple Conditions',
    id: 'multiple-queries',
  },
  addConstraints: {
    title: 'Adding Constraints',
    id: 'add-constraints',
  },
  editCondition: {
    title: 'Edit Condition',
    id: 'edit-condition',
  },
  resetQuery: {
    title: 'Reset Query Entry',
    id: 'reset-query',
  },
  runSearch: {
    title: 'Run Your Search',
    id: 'run-search',
  },
  additionalSearch: {
    title: 'Other Search Methods',
    id: 'additional-search',
  },
};

export const SearchGuide = () => {
  return (
    <article id="search">
      <h3>Search</h3>
      <TableOfContents contents={Object.values(searchContents)} />
      <ul>
        <li id={searchContents.querySyntax.id}>
          <h4>{searchContents.querySyntax.title}</h4>
          <p>
            Our system analyzes the evolution of biomedical literature,
            understanding how entities and the relationships between them change
            over time. We use this information to predict connections between
            entities that haven’t been found yet. We call these ‘ hypotheses,’
            or ‘triplets‘ to predict. Through our GUI, you can use queries
            consisting of one or more conditions and see potential candidates
            that fulfill them. Each condition follows the following format:{' '}
            <strong>Subject – Relation - Object</strong> where the variable to
            predict can either appear as a subject or as an object:
          </p>
          <div className="image-container">
            <img
              src={tripleVisualization}
              alt="example of triple visualization"
            ></img>
          </div>
          <ol className="prediction-type-list">
            <li>
              <strong>Subject Prediction:</strong> You want to search for
              possible subjects that match a given relation and object.
            </li>
            <li>
              <strong>Object Prediction:</strong> You want to search for
              possible objects that match a given subject and relation.
            </li>
          </ol>
          <div className="example">
            <div>
              <p>
                <strong>Example 1: Subject Prediction</strong>
              </p>
              <p>
                If you want to search for proteins related to a disease, you can
                use query like:
              </p>
              <em>&quot;Protein X relates to lung cancer&quot;</em>
              <ul>
                <li>This is a Subject Prediction</li>
                <li>In the case: </li>
                <div className="query">
                  <p>Subject: Protein X (to be predicted)</p>
                  <p>Relation: Relates to</p>
                  <p>Object: Disease lung cancer</p>
                </div>
              </ul>
            </div>
            <div>
              <p>
                <strong>Example 2: Object Prediction</strong>
              </p>
              <p>
                If you want to search for a protein that is degraded by a
                polymer, you can use a query like:
              </p>
              <em>&quot;Polymer Lentinan Degrades Protein X&quot;</em>
              <ul>
                <li>This is an Object Prediction</li>
                <li>In the case: </li>
                <div className="query">
                  <p>Subject: Polymer Lentinan</p>
                  <p>Relation: Degrades</p>
                  <p>Object: Protein X (to be predicted)</p>
                </div>
              </ul>
            </div>
          </div>
        </li>
        <li id={searchContents.singleVsCombo.id}>
          <h4>{searchContents.singleVsCombo.title}</h4>
          <p>
            Our system allows two query modes depending on the type of query you
            are trying to perform. The default selected mode is{' '}
            <em>Single Shot</em>. You can switch between modes at any time by
            clicking at the corresponding tab at the top right of the form
          </p>
          <div className="image-container">
            <figure>
              <img
                src={queryModeToggle}
                alt="Image of the query mode toggle"
              ></img>
              <figcaption>Query mode toggle</figcaption>
            </figure>
          </div>
          <ul>
            <li>
              <strong>Simple query mode:</strong> Allows you to explore
              candidates for a single relationship. You can add constraints that
              all potential candidates must meet, based on previously proven
              knowledge. This mode is suited for focused searches where
              you`&apos;re exploring a single relationship.
            </li>
            <li>
              <strong>Complex query mode:</strong> Allows you to explore
              candidates that satisfy a combination of relationships. You can
              link several relationships, and the system will find candidates
              that fulfill all the conditions simultaneously. This is suited for
              more advanced searches with intersecting predictions.
            </li>
          </ul>
        </li>
        <li id={searchContents.firstQuery.id}>
          <h4>{searchContents.firstQuery.title}</h4>
          <p>On the query input area in the Home page, follow these steps:</p>
          <div className="image-container">
            <figure>
              <video
                src={enteringQueryVideo}
                autoPlay
                loop
                muted
                playsInline
              ></video>
              <figcaption>Entering a first hypothesis to predict</figcaption>
            </figure>
          </div>
          <ol>
            <li>
              Select the <strong>Domain</strong> that you are searching for from
              the dropdown menu. The <strong>X</strong> field represents the
              variable in the query formula. In the current version of the app,
              multiple variables are not supported.
              <div>
                <div className="image-container">
                  <figure>
                    <img
                      src={domainSelectionImg}
                      alt="domain selection input"
                    ></img>
                    <figcaption>Select a variable domain</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={domainSelectedImg}
                      alt="domain and variable inputs already selected"
                    ></img>
                    <figcaption>The variable defaults to (X)</figcaption>
                  </figure>
                </div>
              </div>
            </li>
            <li>
              Select a <strong>Direction</strong>(arrow) that represents the
              position of your variable. To build a subject prediction query
              (where the variable to predict is the subject), the arrow should
              point to the right <strong> &#8594;</strong>. Consequently, for
              object prediction, the arrow should point to the variable
              <strong> &#8592;</strong>. Then select a<strong> Relation</strong>{' '}
              from the dropdown menu.
              <div className="image-container">
                <figure>
                  <img
                    src={predictionSelectionImg}
                    alt="prediction selection input"
                  ></img>
                  <figcaption>Select a prediction type</figcaption>
                </figure>
                <figure>
                  <img
                    src={relationSelection}
                    alt="relation selection input"
                  ></img>
                  <figcaption>Select a relation</figcaption>
                </figure>
              </div>
            </li>

            <li>
              Select the <strong>Domain</strong> and the target
              <strong> Entity</strong> that relates to the variable you are
              searching for. The entity selection input allows you to type part
              of a term to filter the items in the dropdown menu.
              <div className="image-container">
                <figure>
                  <img
                    src={conceptDomainSelection}
                    alt="concept domain selection input"
                  ></img>
                  <figcaption>Select a concept domain</figcaption>
                </figure>
                <figure>
                  <img
                    src={conceptSelection}
                    alt="concept name selection input"
                  ></img>
                  <figcaption>Select a target domain</figcaption>
                </figure>
              </div>
            </li>
            <li>
              Once you are satisfied with your condition, click on the{' '}
              <strong>Add</strong> icon to add it to your condition list.
              <br></br>
              <figure>
                <img src={addButtonImg} alt="add button image"></img>
                <figcaption>Add icon</figcaption>
              </figure>
              <br></br>
              The condition will be displayed on the list. <br></br>
              <br></br>
              If a single condition is enough, , press `&apos;FIND`&apos; to
              <a href={`#${searchContents.runSearch.id}`}> run your search.</a>
              <figure>
                <img
                  src={conditionImg}
                  alt="image of a condition in the conditions list"
                ></img>
              </figure>
            </li>
          </ol>
        </li>
        <li id={searchContents.multipleQueries.id}>
          <h4>{searchContents.multipleQueries.title}</h4>
          <p>
            Both <a href={`#${searchContents.singleVsCombo.id}`}>Combo</a> query
            mode and the constraints area in{' '}
            <a href={`#${searchContents.singleVsCombo.id}`}>Single shot</a>{' '}
            support combining more than one condition, connecting them through
            `&apos;AND`&apos;, `&apos;OR`&apos;, and `&apos;NOT`&apos;
            operators.{' '}
          </p>
          <p>
            To input multiple conditions, click the <strong>[+]</strong> button
            on top of the condition list. A new set of inputs will appear at the
            top row, where you can start filling the additional condition.
            <br></br>
            <figure>
              <img
                src={multipleConditionsImg}
                alt="image of an additional condition in the conditions list"
              ></img>
            </figure>
          </p>
          <ol>
            <li>
              Select one of the boolean operators (AND, AND NOT, OR) to combine
              conditions in your query. <br></br>
              <figure>
                <img src={operatorsTwoImg} alt="operator selection input"></img>
                <figcaption>Select a boolean operator</figcaption>
              </figure>
            </li>
            <li>
              In the current version of the app, multiple variables are not
              supported. Therefore, you cannot change the variable domain (X) on
              the second condition.
              <br></br>
              <figure>
                <img
                  src={variableSelectionImg}
                  alt="variable selection input"
                ></img>
                <figcaption>Variable domain cannot be changed</figcaption>
              </figure>
            </li>
            <li>
              Select the <strong>Direction</strong> and{' '}
              <strong>Relation</strong> from the pulldown menu.
            </li>
            <li>
              Select the <strong>Domain</strong> and the{' '}
              <strong>Concept</strong> that relates to the concept you are
              searching for.
            </li>
            <li>
              Click on the <strong>ADD</strong> button. The condition will be
              displayed on the list. Once you finish adding conditions,{' '}
              <a href={`#${searchContents.runSearch.id}`}>run your search</a>
            </li>
          </ol>
        </li>
        <li id={searchContents.addConstraints.id}>
          <h4>{searchContents.addConstraints.title}</h4>
          <p>
            Constraints are supported only in <em>Single shot</em> mode. They
            provide an additional layer of filtering based on conditions that
            exist in our current body of literature. Consequently, the system
            retrieves only those candidates that our knowledge graph confirms
            satisfy the defined constraints.
          </p>
          <p>
            To add constraints to your search while in <em>Single shot</em>{' '}
            mode, click on the <em>Show constraints</em> arrow at the bottom of
            the form
          </p>
          <div className="image-container">
            <figure>
              <img src={showConstraints} alt="show constraints button"></img>
              <figcaption>Show constraints button</figcaption>
            </figure>
          </div>
          <p>
            After that, the form will expand and you will be able to add a new
            constraints row throught the <strong>[+]</strong> button
          </p>
          . You may notice that constraints have the same format as the rest of
          conditions, so you can follow the steps described in{' '}
          <a href={`#${searchContents.firstQuery.id}`}>
            {searchContents.firstQuery.title}
          </a>
        </li>
        <li id={searchContents.editCondition.id}>
          <h4>{searchContents.editCondition.title}</h4>
          <p>
            You can edit and delete a condition on the conditions list by
            clicking on the relevant icon next to it.<br></br>
            <figure>
              <img src={iconBoxImg} alt="condition icon box"></img>
              <figcaption>
                Use the icons on the right to modify queries
              </figcaption>
            </figure>
          </p>
          <p>
            <strong>Edit: </strong>
          </p>
          <ul>
            <li>
              Click on the Edit icon{' '}
              <img className="icon" src={editIcon} alt="edit icon"></img> to
              enable inline edit.
            </li>
          </ul>
          <p>
            <strong>Duplicate: </strong>
          </p>
          <ul>
            <li>
              Click on the Duplicate icon{' '}
              <img
                className="icon"
                src={duplicateIcon}
                alt="duplicate icon"
              ></img>{' '}
              to duplicate the condition
            </li>
            <li>
              The same condition will be added in the bottom of the list. Then
              you can click on the edit icon{' '}
              <img className="icon" src={editIcon} alt="edit icon"></img> to
              change the condition.
            </li>
          </ul>
          <p>
            <strong>Delete: </strong>
          </p>
          <ul>
            <li>
              Click on the Delete icon
              <img className="icon" src={deleteIcon} alt="delete icon"></img>
              to delete the condition from the list
            </li>
          </ul>
        </li>
        <li id={searchContents.resetQuery.id}>
          <h4>{searchContents.resetQuery.title}</h4>
          <p>
            Click on the RESET button to clear all conditions on the list.
            <br></br>
            <figure>
              <img src={resetButton} alt="reset button image"></img>
              <figcaption>Click on reset to reset the query</figcaption>
            </figure>
          </p>
          <p>
            Note that all conditions will be deleted from your list and not
            sotred on your history unless you have performed a search before
          </p>
        </li>
        <li id={searchContents.runSearch.id}>
          <h4>{searchContents.runSearch.title}</h4>
          <p>
            Once you have finished setting your query, click on the{' '}
            <strong>FIND</strong> button under the Conditions list.
          </p>

          <figure>
            <img src={searchingButton} alt="searching button"></img>
            <figcaption>Find button while loading the search</figcaption>
          </figure>
          <p>
            Our system will start searching for candidates for your query and
            once the query is processed you will be taken to the Candidates
            Page.
          </p>
          <p>
            Note that some searches may take longer than others depending on the
            complexity of the query
          </p>
        </li>
        <li id={searchContents.additionalSearch.id}>
          <h4>{searchContents.additionalSearch.title}</h4>
          <p>
            <u>History</u>
          </p>
          <figure>
            <img
              src={historyImg}
              alt="Image of the search history feature"
            ></img>
            <figcaption>Click on a list item to use it</figcaption>
          </figure>

          <p>
            You can select a previously searched query from your search history
            if you want to try it again or edit it.
          </p>
        </li>
      </ul>
    </article>
  );
};
