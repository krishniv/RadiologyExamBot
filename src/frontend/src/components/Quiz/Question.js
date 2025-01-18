import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import './Quiz.css'; // Import the CSS for the Quiz component

function Question() {
  const [answerSelected, setAnswerSelected] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);

  const score = useSelector((state) => state.score);
  const questions = useSelector((state) => state.questions);
  const questionIndex = useSelector((state) => state.index);

  const dispatch = useDispatch();
  const question = questions[questionIndex];
  const answer = question && question.correct;
  const options = [...question.options];

  const handleListItemClick = (event) => {
    if (answerSelected) return;

    const clickedAnswer = event.target.textContent;
    setAnswerSelected(true);
    setSelectedAnswer(clickedAnswer);

    if (clickedAnswer === answer) {
      dispatch({
        type: 'SET_SCORE',
        score: score + 1,
      });
    }

    // Add delay before moving to next question
    setTimeout(() => {
      setAnswerSelected(false);
      setSelectedAnswer(null);
      dispatch({
        type: 'SET_INDEX',
        index: questionIndex + 1,
      });
    }, 1500);
  };

  const getClass = (option) => {
    if (!answerSelected) return 'option';
    if (option === answer) return 'option correct';
    if (option === selectedAnswer) return 'option selected';
    return 'option';
  };

  if (!question) {
    return <div>Loading</div>;
  }


  return (
      <div className="quiz-content">
        <div className="image-section">
          <img
            src={`/medical_images/${question.image}`}
            alt="Medical Scan"
            className="quiz-image"
          />
        </div>

        <div className="question-section">
          <div className="question-card">
            <div className="question-header">
              <h4 className="question-number">Question {questionIndex + 1}</h4>
              <h2 className="question-title">Identify the Scan</h2>
            </div>

            <ul className="options-list">
              {options.map((option, i) => (
                <li
                  key={i}
                  onClick={handleListItemClick}
                  className={getClass(option)}
                >
                  {option}
                </li>
              ))}
            </ul>

            <div className="score-display">
              Score: {score} / {questions.length}
            </div>
          </div>
        </div>
      </div>
  );
}

export default Question;