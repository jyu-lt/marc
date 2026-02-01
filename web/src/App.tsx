import { useState, useRef, useEffect } from 'react'
import './index.css'

interface ReasoningResult {
  answer: string;
  frameworks_used: string[];
  beliefs_used: string[];
  analogies_used: string[];
  reasoning_steps: string[];
  caveats: string[];
  confidence: number;
}

function App() {
  const [query, setQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<ReasoningResult | null>(null)
  const [lastQuery, setLastQuery] = useState('')
  const resultRef = useRef<HTMLDivElement>(null)

  const handleAsk = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    setResult(null)
    setLastQuery(query)

    try {
      const response = await fetch('http://localhost:3001/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      if (!response.ok) throw new Error('Failed to fetch')
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error:', error)
      alert('Error connecting to the reasoning server. Make sure the backend is running.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (result) {
      resultRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [result])

  return (
    <div className="container">
      <header>
        <h1>Marc</h1>
        <p className="subtitle">Advanced Reasoning & Knowledge Synthesis</p>
      </header>

      <main className="chat-container">
        <form className="input-group" onSubmit={handleAsk}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a deep question..."
            disabled={isLoading}
            autoFocus
          />
          <button type="submit" disabled={isLoading || !query.trim()}>
            {isLoading ? (
              <div className="loading-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            ) : 'Analyze'}
          </button>
        </form>

        <div className="result-area">
          {result && (
            <div className="message assistant" ref={resultRef}>
              <div className="query-context" style={{ marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-color)' }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Query: </span>
                <span style={{ fontWeight: 500 }}>{lastQuery}</span>
              </div>

              <div className="synthesis">
                <p style={{ fontSize: '1.2rem', lineHeight: '1.6', marginBottom: '1.5rem' }}>
                  {result.answer}
                </p>
              </div>

              <div className="reasoning-grid">
                {result.reasoning_steps?.length > 0 && (
                  <div className="card" style={{ gridColumn: '1 / -1' }}>
                    <h3>🪜 Reasoning Process</h3>
                    <div className="card-content">
                      <ol style={{ paddingLeft: '1.2rem' }}>
                        {result.reasoning_steps.map((step, i) => (
                          <li key={i} style={{ marginBottom: '0.5rem' }}>{step}</li>
                        ))}
                      </ol>
                    </div>
                  </div>
                )}

                {result.frameworks_used?.length > 0 && (
                  <div className="card">
                    <h3>🔍 Frameworks Applied</h3>
                    <div className="card-content">
                      {result.frameworks_used.map((name, i) => (
                        <span key={i} className="framework-tag">{name}</span>
                      ))}
                    </div>
                  </div>
                )}

                {result.beliefs_used?.length > 0 && (
                  <div className="card">
                    <h3>💡 Key Beliefs</h3>
                    <div className="card-content">
                      <ul style={{ paddingLeft: '1rem' }}>
                        {result.beliefs_used.map((statement, i) => (
                          <li key={i} style={{ marginBottom: '0.5rem' }}>{statement}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {result.analogies_used?.length > 0 && (
                  <div className="card">
                    <h3>🔗 Analogies</h3>
                    <div className="card-content">
                      {result.analogies_used.map((source, i) => (
                        <div key={i} style={{ marginBottom: '0.5rem' }}>
                          <span style={{ color: 'var(--accent-color)' }}>Source: </span> {source}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="card">
                  <h3>📊 Confidence</h3>
                  <div className="card-content">
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-color)' }}>
                      {(result.confidence * 100).toFixed(0)}%
                    </div>
                    {result.caveats?.length > 0 && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
                        <strong>Caveats:</strong>
                        <ul style={{ paddingLeft: '1rem', marginTop: '0.25rem' }}>
                          {result.caveats.map((c, i) => <li key={i}>{c}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              <details style={{ marginTop: '2rem' }}>
                <summary style={{ fontSize: '0.8rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                  View Raw JSON Protocol
                </summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
