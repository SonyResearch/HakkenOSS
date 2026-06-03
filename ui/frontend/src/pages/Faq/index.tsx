import { useState } from 'react';
import './index.css';

interface FaqItemProps {
  question: string;
  answer: string;
}

const FAQPage = () => {
  const faqs = [
    {
      question: 'I forgot my password, what do I do?',
      answer: 'To be discussed',
    },
    {
      question: "I don't remember my username, what can I do?",
      answer: 'To be discussed',
    },
    {
      question: "What does '' error mean",
      answer: 'To be discussed',
    },
    {
      question: 'Why do the dropdown menus change content?',
      answer: 'To be discussed',
    },
    {
      question: 'How do I get more candidates?',
      answer: 'To be discussed',
    },
    {
      question: 'Can I change the query formula after the search?',
      answer:
        'Yes, once you have submitted your query and you are in the results page, you will see an icon next to your query. Clicking on that icon will bring you to the search form again, where you will be able to edit your conditions',
    },
  ];

  const FaqItem = ({ question, answer }: FaqItemProps) => {
    const [showAnswer, setShowAnswer] = useState(false);
    return (
      <div className={`faq-item-container ${showAnswer ? 'open' : ''}`}>
        <div
          className="question-container"
          onClick={() => setShowAnswer(!showAnswer)}
        >
          <span>+</span>
          <p>{question}</p>
        </div>
        <div className="answer-container">
          <p>{answer}</p>
        </div>
      </div>
    );
  };

  return (
    <section className="faq-page">
      <h1>Frequently Asked Questions</h1>
      {faqs.map((faq, index) => (
        <FaqItem key={index} question={faq.question} answer={faq.answer} />
      ))}
    </section>
  );
};

export default FAQPage;
