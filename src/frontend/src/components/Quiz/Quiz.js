import React from 'react';
import { useSelector } from 'react-redux';
import Question from './Question';
import Settings from './Settings';
import FinalScreen from './FinalScreen';
import './Quiz.css'; // Import the CSS for the Quiz component

function Quiz() {
  const questions = useSelector((state) => state.questions);
  const questionIndex = useSelector((state) => state.index);

  let component;

  if (questions.length && questionIndex < questions.length) {
    component = <Question />;
  } else if (!questions.length) {
    component = <Settings />;
  } else {
    component = <FinalScreen />;
  }
  console.log('Questions:', questions); // Debugging line

  return (
    <div className="quiz-container">
        <h2 className="chatbot-title"> Radiology Quiz</h2>

      {component}
    </div>
  );
}

export default Quiz;