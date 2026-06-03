import './index.css';

export const LoadingQuery = () => {
  return (
    <div className="loading-modal">
      <div>
        <h2>
          <strong>Hang tight</strong>
        </h2>
        <p>Our system is analyzing results to identify the best candidates.</p>
        <div className="loader"></div>
      </div>
    </div>
  );
};

export default LoadingQuery;
