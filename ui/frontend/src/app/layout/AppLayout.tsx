import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Footer from './Footer';
import Header from './Header';
import ResultsPage from '../../pages/ResultsPage';
import { QueryProvider } from '../../contexts/QueryContext';
import UserGuide from '../../pages/UserGuide';
import ScrollToTop from '../../shared/components/ScrollToTop';
import ValidationPage from '../../pages/ValidationPage';
import QueryPage from '../../pages/QueryPage';

function AppLayout() {
  return (
    <BrowserRouter>
      <QueryProvider>
        <Header />
        <div className="page-content">
          <ScrollToTop />
          <Routes>
            <Route path="/" element={<QueryPage />} />
            <Route path="/user-guide" element={<UserGuide />} />
            {/*<Route path="/faq" element={<FAQPage />} />*/}
            <Route path="/query/results" element={<ResultsPage />} />
            <Route path="/validation" element={<ValidationPage />} />
          </Routes>
        </div>
        <Footer />
      </QueryProvider>
    </BrowserRouter>
  );
}

export default AppLayout;
