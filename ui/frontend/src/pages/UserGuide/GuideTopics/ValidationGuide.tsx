import TableOfContents from '../TableOfContents';
import validationForm from '../../../assets/images/help-guide/validation-form.png';
import validationAny from '../../../assets/images/help-guide/validation-any.png';
import validationButton from '../../../assets/images/help-guide/validation-button.png';
import validationTable from '../../../assets/images/help-guide/validation-table.png';
import { AssignmentOutlined } from '@mui/icons-material';

export const validationContents = {
  addingATriple: {
    title: 'Adding a triple',
    id: 'adding-a-triple',
  },
  validationTable: {
    title: 'Validation table',
    id: 'validation-table',
  },
};

export const ValidationGuide = () => {
  return (
    <article id="validation">
      <h3>Validation</h3>
      <TableOfContents contents={Object.values(validationContents)} />
      <ul>
        <li>
          <p>
            <p>
              In validation mode, you can enter a{' '}
              <strong>complete triple</strong>
              (Subject – Relation – Object) instead of leaving one component
              unspecified. Rather than generating new hypotheses, the system
              evaluates the provided triple and assigns it a{' '}
              <strong>confidence score</strong>.
            </p>
            <p>
              As in Hakken’s hypothesis generation feature, the confidence score
              quantifies the degree of support the model assigns to a given
              relationship based on its training over our body of biomedical
              literature. This metric can be used to evaluate the strength and
              plausibility of both established and hypothesized relationships.
            </p>
          </p>
        </li>
        <li id={validationContents.addingATriple.id}>
          <h4>{validationContents.addingATriple.title}</h4>
          <p>
            To add a triple for scoring, fill in the <strong>Subject</strong>,
            <strong> Relation</strong>, and <strong>Object</strong> fields with
            the appropriate values.
          </p>
          <div className="image-container">
            <img src={validationForm} alt="image of the validation form"></img>
          </div>
          <p>
            If you do not care about the specific relationship between two
            entities and would like to receive scores for all possible
            relationships between them, select <strong>ANY</strong> as the
            relation.
          </p>
          <div className="image-container">
            <figure>
              <img src={validationAny} alt="any relationship field"></img>
              <figcaption>
                Select <strong>ANY</strong> on the relationship field to score
                all relations
              </figcaption>
            </figure>
          </div>
          <p>
            Once you are satisfied with your triple, click on the
            `&APOS;`VALIDATE`&APOS;` button to add the triple to the table
          </p>
          <figure>
            <img src={validationButton} alt="validation button"></img>
            <figcaption>Click on the button to see the scores</figcaption>
          </figure>
        </li>
        <li id={validationContents.validationTable.id}>
          <h4>{validationContents.validationTable.title}</h4>
          <p>
            Once you click the <strong>VALIDATE</strong> button, the triples
            will be added to the table.
          </p>
          <div className="image-container">
            <img src={validationTable} alt="validation table"></img>
          </div>
          <p>
            You can view the contextualization and explanation for each triple
            by clicking the <AssignmentOutlined /> icon and following the
            instructions described on{' '}
            <a href={`#page-layout`}>candidates page layout</a>.
          </p>
        </li>
      </ul>
    </article>
  );
};
