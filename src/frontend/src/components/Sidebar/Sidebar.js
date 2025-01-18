import './Sidebar.css'

const Sidebar = () => {
  return (
    <aside style={{
      width: '250px',
      height: '100vh',
      backgroundColor: '#1a1a2e',
      borderRight: '1px solid #363648',
      position: 'fixed',
      left: 0,
      top: 0
    }}>
      <div className="sidebar-header">
        <h1>Quiz Master</h1>
      </div>
      <nav className="sidebar-nav">
        <ul>
          <li className="nav-item active">
            <span className="nav-icon">📊</span>
            Dashboard
          </li>
          <li className="nav-item">
            <span className="nav-icon">🏆</span>
            Leaderboard
          </li>
          <li className="nav-item">
            <span className="nav-icon">📝</span>
            My Quizzes
          </li>
          <li className="nav-item">
            <span className="nav-icon">⚙️</span>
            Settings
          </li>
        </ul>
      </nav>
    </aside>
  )
}

export default Sidebar 