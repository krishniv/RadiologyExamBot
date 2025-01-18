import React from 'react'

const Footer = () => {
  return (
    <footer style={{
      backgroundColor: '#1a1a2e',
      padding: '2rem',
      marginTop: 'auto',
      borderTop: '1px solid #363648',
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '2rem'
      }}>
        <div style={{ color: '#8890b5' }}>
          © 2024 Radiology Quiz App
        </div>
        
        <div style={{
          display: 'flex',
          gap: '1.5rem'
        }}>
          <a
            href="https://github.com/yourusername/your-repo"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: '#8890b5',
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <i className="fab fa-github"></i>
            GitHub
          </a>
          <a
            href="https://linkedin.com/in/your-profile"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: '#8890b5',
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <i className="fab fa-linkedin"></i>
            LinkedIn
          </a>
        </div>
      </div>
    </footer>
  )
}

export default Footer 