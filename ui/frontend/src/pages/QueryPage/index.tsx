import MainForm from '../../features/QueryForm/components/MainForm';
import UseCases from '../../features/UseCases';
import { QueryFormProvider } from '../../contexts/QueryFormContext';
import IntroductionText from '../../shared/components/Introduction';

const QueryPage = () => {
  return (
    <div>
      <IntroductionText />
      <QueryFormProvider>
        <UseCases />
        <MainForm />
      </QueryFormProvider>
    </div>
  );
};

export default QueryPage;
