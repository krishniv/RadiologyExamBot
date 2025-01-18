import React, { useState } from 'react';
import './Chatbot.css'; // Optional: Create a CSS file for styling

function Chatbot() {
  const [userMessage, setUserMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleSendMessage = async () => {
    if (userMessage.trim()) {
      // Add user message to chat history
      setChatHistory([...chatHistory, { sender: 'user', text: userMessage }]);
      
      // Simulate a response from the chatbot (replace with actual API call)
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_message: userMessage }),
      });
      const data = await response.json();
      
      // Add chatbot response to chat history
      setChatHistory([...chatHistory, { sender: 'user', text: userMessage }, { sender: 'bot', text: data.response }]);
      setUserMessage(''); // Clear input field
    }
  };

  return (
    <div className="chatbot-container">
      <h2 className="chatbot-title">Chatbot</h2>
      <div className="chat-history">
        {chatHistory.map((msg, index) => (
          <div key={index} className={msg.sender === 'user' ? 'user-message' : 'bot-message'}>
            {msg.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={userMessage}
        onChange={(e) => setUserMessage(e.target.value)}
        placeholder="Type your message..."
        className="chat-input"
      />
      <button onClick={handleSendMessage} className="send-button">Send</button>
    </div>
  );
}

export default Chatbot;