import React from 'react'
import ReactDOM from 'react-dom'
import './index.css'
import App from './App'
import Quiz from './components/Quiz/Quiz'
import Reducer from './Reducer'
import { createStore } from 'redux'
import { Provider } from 'react-redux'
import { BrowserRouter as Router } from 'react-router-dom'
const store = createStore(Reducer)

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <Router>
        <App />
      </Router>
    </Provider>
  </React.StrictMode>,
  document.getElementById('root')
)
