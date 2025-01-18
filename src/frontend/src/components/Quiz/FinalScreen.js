import React from 'react'
import { useSelector, useDispatch } from 'react-redux'
import './Quiz.css'; // Import the CSS for the Quiz component


function FinalScreen() {
  const score = useSelector((state) => state.score)
  const questions = useSelector((state) => state.questions)
  const dispatch = useDispatch()

  const percentage = Math.round((score / questions.length) * 100)

  const replay = () => {
    dispatch({
      type: 'SET_INDEX',
      index: 0,
    })

    dispatch({
      type: 'SET_SCORE',
      score: 0,
    })
  }

  const settings = () => {
    dispatch({
      type: 'SET_QUESTIONS',
      questions: [],
    })

    dispatch({
      type: 'SET_SCORE',
      score: 0,
    })
  }

  const getFeedbackMessage = () => {
    if (percentage >= 80) return "Excellent! You're a radiology expert!"
    if (percentage >= 60) return "Good job! Keep practicing!"
    return "Keep learning! You'll improve!"
  }

  return (
    <div className="final-screen">
      <div className="final-card">
        <h2 className="final-header">Quiz Complete!</h2>
        
        <div className="score-section">
          <div className="score-circle">
            <span className="score-percentage">{percentage}%</span>
            <span className="score-text">Score</span>
          </div>
          <div className="score-details">
            <p className="correct-answers">
              Correct Answers: {score} / {questions.length}
            </p>
            <p className="feedback-message">{getFeedbackMessage()}</p>
          </div>
        </div>

        <div className="action-buttons">
          <button className="replay-button" onClick={replay}>
            Try Again
          </button>
          <button className="new-quiz-button" onClick={settings}>
            New Quiz
          </button>
        </div>
      </div>
    </div>
  )
}

export default FinalScreen