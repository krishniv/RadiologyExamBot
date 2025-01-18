import { useSelector } from 'react-redux';
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Quiz from './components/Quiz/Quiz'; // Import the new Quiz component
import Sidebar from './components/Sidebar/Sidebar';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import Chatbot from './components/chatbot/Chatbot'; // Import the Chatbot component
import './App.css';

function App() {

  const state = useSelector((state) => state);
  console.log('Full Redux State:', state);
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <div className="content-area">
          <div className="content-header">
            <h1 className="content-title">Radiology Assistant</h1>
            <p className="content-subtitle">One App!</p>
          </div>
          <div className="settings-chat-container">
            <div className="chatbot-container">
              <Quiz /> {/* Use the Quiz component here */}
            </div>
            <div className="chatbot-container">
              <Chatbot /> {/* Add the Chatbot component here */}
            </div>
            <div className="chatbot-container">
              <Chatbot /> {/* Add the Chatbot component here */}
            </div>
          </div>
        </div>
        <Footer />
      </div>
    </div>
  );
}

export default App;