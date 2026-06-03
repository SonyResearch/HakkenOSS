export const helpInputQuery = (
  <div>
    <h3>How do I search ?</h3>
    <ol>
      <li>
        Have in mind the domain of the concept you want to search. Such as drug,
        biomarker, so on.
      </li>
      <li>
        Query Condition is triple of Subject-Relation-Object. Select
        &apos;Prediction type&apos; in the Query Syntax box. Select
        &apos;Subject Prediction&apos; if the concept you are searching for acts
        as the subject in the condition sentense. e.g. You are searching X:
        &apos;Drug related concept X - relates to - disease Alzheimer&apos;
        Select &apos;Object Prediction&apos; if the concept you are searching
        for acts as the object in the condition sentence. e.g. You are searching
        X: &apos;Drug related concept aspirin - treats - disease X&apos;
      </li>
      <li>
        Set query parameters in the Query Condition box.
        <ol>
          <li>
            Select &apos;Boolean Operation&apos;. The app applies Boolean
            operators (AND, OR, NOT) between concepts. If you enter the first
            condition, select the default &apos;AND&apos;. Currently the app
            processes searches in a right to left sequence. e.g. &apos;A AND B
            OR C &apos; is parsed as &apos; (A AND (B OR C))&apos;. The
            &apos;nest&apos; function using parentheses is not supported.
          </li>
          <li>
            Select search &apos;Type&apos;. Currently you can select
            &apos;PROBABILITY&apos; type.
          </li>
          <li>Select &apos;Relation&apos; from the dropdown menu.</li>
          <li>
            Select &apos;Domain&apos; and the &apos;concept&apos; in the
            &apos;Object&apos; section. If the prediction type is Subject
            Prediction, select the domain and the concept that relates to the
            concept you are searching for. The concept selection box allows you
            to type part of a term to filter the items in the dropdown menu. If
            the prediction type is Object Prediction, select the domain of the
            concept you are searching for from the dropdown menu. Then select
            &apos;X&apos; as variable in the query formula.
          </li>
        </ol>
      </li>
      <li>
        Press the &apos;ADD&apos; key to add the query condition.<br></br>If you
        input more condition, Press &apos;+&apos; key to add another condition.
        You can edit and delete the condition in the list by pressing the icon
        in the right. When you add similar condition, press the
        &apos;Duplicate&apos; icon. The same condition will be add in the bottom
        of the list. Then edit a part of the condition.
      </li>
      <li>
        Once you have completed your query input, press the &apos;SEARCH&apos;
        key.
      </li>
      <li>
        Search result will be displayed in the Search Result page. The most
        relevant candidates will be listed. The query score is the predicted
        score based on all conditions. It is a value between 0 and 1. The closer
        to 1 the more relevant the query is. You can also check the score of
        each condition by selecting the GRAPH mode in the dropdown list of
        Display Option .
      </li>
    </ol>
    <div>
      <strong>Note:</strong>
      <ul>
        <li>
          You can only set one domain for the target concept you are searching
          for. If you want to change the domain, press &apos;RESET&apos; under
          the query box to clear the conditions you are setting.
        </li>
        <li>
          If the subject-relation-object triple combination is not available as
          a search condition, a warning will be displayed. Try changing it to a
          different combination. In the next step, it will be improved so that
          you can select only triples that can be searched.
        </li>
      </ul>
    </div>
  </div>
);
