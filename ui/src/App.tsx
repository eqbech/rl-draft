import './App.css'
import { QTable } from './components/qTable'

function App() {

  return (
    <div>
      <h1>Reinforcement Learning</h1>
      <div
      style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <QTable />
      </div>
    </div>
  )
}

export default App
