import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';

const decodeHTML = function (html) {
  const txt = document.createElement('textarea');
  txt.innerHTML = html;
  return txt.value;
};

function Question() {
  const [answerSelected, setAnswerSelected] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);

  const score = useSelector((state) => state.score);
  const questions = useSelector((state) => state.questions);
  const questionIndex = useSelector((state) => state.index);

  const dispatch = useDispatch();

  // Get the current question based on the index
  const question = questions[questionIndex];
  const answer = question && question.correct;


  const options =[...question.options]

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

    if (questionIndex + 1 <= questions.length) {
      setTimeout(() => {
        setAnswerSelected(false);
        setSelectedAnswer(null);
        dispatch({
          type: 'SET_INDEX',
          index: questionIndex + 1,
        });
      }, 2500);
    }
    else {
      dispatch({
        type: 'SET_INDEX',
        index: questionIndex + 1,
      });
    }
  };

  const getClass = (option) => {
    if (!answerSelected) return '';
    if (option === answer) return 'correct';
    if (option === selectedAnswer) return 'selected';
    return '';
  };

  if (!question) {
    return <div>Loading</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'row'}}>
  <div style={{ marginRight: '20px' }}>
    <img
      src={`/medical_images/${question.image}`}
      alt="Medical Scan"
      style={{ width: '800px', height: '300px', maxWidth: '100%',paddingTop:"100px",paddingRight:"60px"}}
    />
  </div>
  <div>
    <p>Question {questionIndex + 1}</p>
    <h3>Identify the Scan</h3>
    <ul>
      {options.map((option, i) => (
        <li key={i} onClick={handleListItemClick} className={getClass(option)}>
          {option}
        </li>
      ))}
    </ul>
    <div>
      Score: {score} / {questions.length}
    </div>
  </div>
</div>
  );
}

export default Question;
