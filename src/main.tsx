import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { applyAppearance, loadAppearance } from './appearance'
import './styles.css'

applyAppearance(loadAppearance())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
