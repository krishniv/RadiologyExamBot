import React from 'react'
import { useSelector, useDispatch } from 'react-redux'
import './Quiz.css'; // Import the CSS for the Quiz component

function FetchButton(props) {
  
  const questionAmount = useSelector(
    (state) => state.options.amount_of_questions
  )
  const questionIndex = useSelector((state) => state.index)

  const dispatch = useDispatch()

  const setLoading = (value) => {
    dispatch({
      type: 'CHANGE_LOADING',
      loading: value,
    })
  }

  const setQuestions = (value) => {
    dispatch({
      type: 'SET_QUESTIONS',
      questions: value,
    })
  }

  const handleQuery = async () => {
    const apiUrl = `${process.env.REACT_APP_API_URL}/generate/${questionAmount}`;

    setLoading(true)

    await fetch(apiUrl)
      .then((res) => res.json())
      .then((response) => {
        setQuestions(response.questions)
        setLoading(false)
      })

    if (questionIndex > 0) {
      dispatch({
        type: 'SET_INDEX',
        index: 0,
      })

      dispatch({
        type: 'SET_SCORE',
        score: 0,
      })
    }
  }

  return <button 
  className="new-quiz-button"
  onClick={handleQuery}>{props.text}</button>
}
export default FetchButton