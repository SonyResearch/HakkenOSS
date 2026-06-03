import './index.css';
const IntroductionText = () => {
  return (
    <>
      <h1>Welcome to Hakken</h1>
      <p className="intro-text">
        Hakken brings a new data source to biomedical research and drug
        discovery. Its aim is to massively augment the impact of scientific
        knowledge, using vast collections of peer-reviewed literature to predict
        millions of new connections between entities. It then scores these
        predictions, and explains them.
      </p>
      <p className="intro-text">
        You can explore Hakken’s predictions individually or en masse. This
        platform opens up Hakken’s predicted data for deep exploration.
      </p>
    </>
  );
};

export default IntroductionText;
