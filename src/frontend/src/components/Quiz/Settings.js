import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import FetchButton from './FetchButton';
import './Quiz.css'; // Import the CSS for the Quiz component

function Settings() {
  const questionAmount = useSelector((state) => state.options.amount_of_questions);
  const loading = useSelector((state) => state.options.loading);
  const dispatch = useDispatch();

  const handleAmountChange = (event) => {
    dispatch({
      type: 'CHANGE_AMOUNT',
      amount_of_questions: event.target.value,
    });
  }

  if (!loading) {
    return (
        <div className="settings-card">
          <h2 className="settings-header">Number of Questions:</h2>
          <input
            type="number"
            value={questionAmount}
            onChange={handleAmountChange}
            min="1"
            className="settings-input"
          />
        <FetchButton text="Get Started!" />
      </div>
    );
  }

  return <p className="loading-text">LOADING...</p>;
}

export default Settings;