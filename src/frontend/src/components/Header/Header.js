import React from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate for navigation
import './Header.css';
import ConnectWallet from './ConnectWallet'; // Import the ConnectWallet component

function Header() {
  const navigate = useNavigate(); // Initialize useNavigate

  const handleProfileClick = async () => {
    try {
      const response = await fetch('/api/auth/login'); // Fetch login endpoint
      if (!response.ok) {
        throw new Error('Failed to log in');
      }
      const data = await response.json();
      console.log('Login Response:', data); // Handle the login response as needed
      // Optionally navigate to the profile page after successful login
      navigate('/profile'); // Navigate to the profile page
    } catch (error) {
      console.error('Error logging in:', error);
    }
  };

  return (
    <header className="header">
      <h2 className="header-title">Welcome to Medical Rad Agent</h2>
      <div className="header-buttons">
        <ConnectWallet /> {/* Use the ConnectWallet component */}
        <button className="header-button" onClick={handleProfileClick}>
          Profile
        </button>
      </div>
    </header>
  );
}

export default Header; 