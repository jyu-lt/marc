import { useState, useRef, useEffect } from 'react'
import './index.css'

interface ReasoningResult {
  query: string;
  frameworks_used: Array<{
    id: string;
    name: string;
    description: string;
    rationale: string;
  }>;
  relevant_beliefs: Array<{
    id: string;
    statement: string;
    confidence: number;
    relevance_rationale: string;
  }>;
  analogies: Array<{
    analogy_id: string;
    lesson: string;
    relevance: string;
  }>;
  synthesis: string;
}

function App() {
  const [query, setQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<ReasoningResult | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)

  const handleAsk = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    setResult(null)

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
              <div className="synthesis">
                <p style={{ fontSize: '1.2rem', lineHeight: '1.6', marginBottom: '1.5rem' }}>
                  {result.synthesis}
                </p>
              </div>

              <div className="reasoning-grid">
                {result.frameworks_used.length > 0 && (
                  <div className="card">
                    <h3>🔍 Frameworks Applied</h3>
                    <div className="card-content">
                      {result.frameworks_used.map(f => (
                        <div key={f.id} style={{ marginBottom: '0.75rem' }}>
                          <span className="framework-tag">{f.name}</span>
                          <p style={{ fontSize: '0.8rem', marginTop: '0.25rem' }}>{f.rationale}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.relevant_beliefs.length > 0 && (
                  <div className="card">
                    <h3>💡 Relevant Beliefs</h3>
                    <div className="card-content">
                      <ul style={{ paddingLeft: '1rem' }}>
                        {result.relevant_beliefs.map((b, i) => (
                          <li key={i} style={{ marginBottom: '0.5rem' }}>
                            {b.statement}
                            <div style={{ fontSize: '0.75rem', color: 'var(--accent-color)' }}>
                              Confidence: {(b.confidence * 100).toFixed(0)}%
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {result.analogies.length > 0 && (
                  <div className="card">
                    <h3>🔗 Analogical Mapping</h3>
                    <div className="card-content">
                      {result.analogies.map((a, i) => (
                        <div key={i} style={{ marginBottom: '0.75rem' }}>
                          <p>"{a.lesson}"</p>
                          <p style={{ fontSize: '0.8rem', fontStyle: 'italic', marginTop: '0.25rem' }}>
                            Relevance: {a.relevance}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              <details style={{ marginTop: '1.5rem' }}>
                <summary style={{ fontSize: '0.8rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                  View Raw JSON Result
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
