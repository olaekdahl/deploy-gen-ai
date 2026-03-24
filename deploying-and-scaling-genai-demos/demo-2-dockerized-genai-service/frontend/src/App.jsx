import { useState, useEffect } from 'react';

const styles = {
  body: {
    fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
    margin: 0,
    padding: 0,
    minHeight: '100vh',
    background: '#0f172a',
    color: '#e2e8f0',
  },
  container: {
    maxWidth: 720,
    margin: '0 auto',
    padding: '2rem 1.5rem',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem',
  },
  title: {
    fontSize: '1.75rem',
    fontWeight: 700,
    color: '#f8fafc',
    margin: 0,
  },
  subtitle: {
    fontSize: '0.875rem',
    color: '#94a3b8',
    marginTop: '0.5rem',
  },
  statusBadge: (healthy) => ({
    display: 'inline-block',
    padding: '0.25rem 0.75rem',
    borderRadius: '9999px',
    fontSize: '0.75rem',
    fontWeight: 600,
    background: healthy ? '#065f4620' : '#7f1d1d20',
    color: healthy ? '#4ade80' : '#f87171',
    border: `1px solid ${healthy ? '#4ade8040' : '#f8717140'}`,
    marginTop: '0.75rem',
  }),
  card: {
    background: '#1e293b',
    borderRadius: 12,
    padding: '1.5rem',
    marginBottom: '1.5rem',
    border: '1px solid #334155',
  },
  label: {
    display: 'block',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: '#94a3b8',
    marginBottom: '0.4rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  textarea: {
    width: '100%',
    minHeight: 100,
    padding: '0.75rem',
    borderRadius: 8,
    border: '1px solid #334155',
    background: '#0f172a',
    color: '#e2e8f0',
    fontSize: '0.95rem',
    fontFamily: 'inherit',
    resize: 'vertical',
    outline: 'none',
    boxSizing: 'border-box',
  },
  sliderRow: {
    display: 'flex',
    gap: '1.5rem',
    marginTop: '1rem',
  },
  sliderGroup: {
    flex: 1,
  },
  slider: {
    width: '100%',
    accentColor: '#6366f1',
  },
  sliderValue: {
    fontSize: '0.8rem',
    color: '#6366f1',
    fontWeight: 600,
  },
  button: (disabled) => ({
    width: '100%',
    padding: '0.75rem',
    borderRadius: 8,
    border: 'none',
    background: disabled ? '#334155' : '#6366f1',
    color: disabled ? '#64748b' : '#fff',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    marginTop: '1rem',
    transition: 'background 0.2s',
  }),
  resultBox: {
    background: '#0f172a',
    borderRadius: 8,
    padding: '1rem',
    fontFamily: "'Fira Code', 'Consolas', monospace",
    fontSize: '0.9rem',
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    border: '1px solid #334155',
  },
  highlight: {
    color: '#818cf8',
    fontWeight: 600,
  },
  meta: {
    display: 'flex',
    gap: '1.5rem',
    marginTop: '0.75rem',
    fontSize: '0.8rem',
    color: '#64748b',
  },
  error: {
    background: '#7f1d1d20',
    border: '1px solid #f8717140',
    borderRadius: 8,
    padding: '1rem',
    color: '#f87171',
    fontSize: '0.875rem',
  },
};

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(50);
  const [temperature, setTemperature] = useState(0.7);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  // Check service health on mount
  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const start = performance.now();
    try {
      const resp = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          max_tokens: maxTokens,
          temperature,
        }),
      });

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      data._clientLatencyMs = Math.round(performance.now() - start);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleGenerate();
    }
  };

  const isHealthy = health?.model_loaded === true;

  return (
    <div style={styles.body}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>GenAI Text Generation</h1>
          <p style={styles.subtitle}>
            Demo 2 -- Dockerized Service with React UI
          </p>
          <div style={styles.statusBadge(isHealthy)}>
            {isHealthy
              ? `Model loaded: ${health.model_name}`
              : 'Service unavailable'}
          </div>
        </div>

        {/* Input card */}
        <div style={styles.card}>
          <label style={styles.label}>Prompt</label>
          <textarea
            style={styles.textarea}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter your prompt here... (Ctrl+Enter to generate)"
          />

          <div style={styles.sliderRow}>
            <div style={styles.sliderGroup}>
              <label style={styles.label}>
                Max Tokens{' '}
                <span style={styles.sliderValue}>{maxTokens}</span>
              </label>
              <input
                type="range"
                min={1}
                max={200}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                style={styles.slider}
              />
            </div>
            <div style={styles.sliderGroup}>
              <label style={styles.label}>
                Temperature{' '}
                <span style={styles.sliderValue}>{temperature.toFixed(1)}</span>
              </label>
              <input
                type="range"
                min={0}
                max={20}
                value={Math.round(temperature * 10)}
                onChange={(e) => setTemperature(Number(e.target.value) / 10)}
                style={styles.slider}
              />
            </div>
          </div>

          <button
            style={styles.button(loading || !prompt.trim())}
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
          >
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={styles.error}>Error: {error}</div>
        )}

        {/* Result card */}
        {result && (
          <div style={styles.card}>
            <label style={styles.label}>Generated Output</label>
            <div style={styles.resultBox}>
              <span style={styles.highlight}>{result.prompt}</span>
              {result.generated_text.slice(result.prompt.length)}
            </div>
            <div style={styles.meta}>
              <span>Tokens: {result.tokens_generated}</span>
              <span>Model: {result.model_name}</span>
              <span>Latency: {result._clientLatencyMs}ms</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
