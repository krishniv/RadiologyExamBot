import React from 'react'
import { useSelector, useDispatch } from 'react-redux'
import FetchButton from './FetchButton'

function Settings() {
  const questionAmount = useSelector((state) => state.options.amount_of_questions)
  const loading = useSelector((state) => state.options.loading)
  const dispatch = useDispatch()

  const handleAmountChange = (event) => {
    dispatch({
      type: 'CHANGE_AMOUNT',
      amount_of_questions: event.target.value,
    })
  }


  if (!loading) {
    return (
      <div>
        <h1>Quiz App</h1>
        <div>
          <h2>Amount of Questions:</h2>
          <input
            type="number"
            value={questionAmount}
            onChange={handleAmountChange}
            min="1"
          />
        </div>
        <FetchButton text="Get Started!"/>
      </div>
    )
  }

  return <p>LOADING...</p>
}

export default Settings
