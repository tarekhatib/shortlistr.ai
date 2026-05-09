import { useRef, useState } from 'react'
import './App.css'

const FEATURE_META = [
  { key: 'years_experience',   label: 'Experience',      fmt: v => `${v} yrs` },
  { key: 'skills_match_score', label: 'Skills Match',    fmt: v => `${Number(v).toFixed(1)}%` },
  { key: 'education_level',    label: 'Education',       fmt: v => v },
  { key: 'project_count',      label: 'Projects',        fmt: v => v },
  { key: 'resume_length',      label: 'Resume Length',   fmt: v => `${v} words` },
]

export default function App() {
  const [pdfFile, setPdfFile]   = useState(null)
  const [jdText, setJdText]     = useState('')
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)
  const fileInputRef = useRef()

  function handleFile(file) {
    if (file?.type === 'application/pdf') {
      setPdfFile(file)
      setResult(null)
      setError(null)
    }
  }

  async function handleAnalyze() {
    if (!pdfFile || !jdText.trim()) return
    setLoading(true)
    setResult(null)
    setError(null)
    const form = new FormData()
    form.append('cv', pdfFile)
    form.append('jd', jdText)
    try {
      const res  = await fetch('/analyze', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Analysis failed.')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const canAnalyze = pdfFile && jdText.trim() && !loading

  return (
    <div className="page">
      <nav className="nav">
        <span className="nav-logo"><strong>Shortlistr</strong>.ai</span>
        <span className="badge-beta">BETA</span>
      </nav>

      <main className="main">
        <div className="grid">

          {/* ── Input column ── */}
          <div className="col-input">
            <h1 className="page-title">Does this CV make the cut?</h1>
            <p className="page-desc">Drop a CV and paste a job description. Two models predict whether the candidate gets shortlisted.</p>

            <div
              className={`dropzone ${dragging ? 'over' : ''} ${pdfFile ? 'filled' : ''}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
            >
              <input ref={fileInputRef} type="file" accept=".pdf" hidden onChange={e => handleFile(e.target.files[0])} />
              {pdfFile ? (
                <div className="file-row">
                  <PdfIcon />
                  <span className="file-name">{pdfFile.name}</span>
                  <button className="clear-btn" onClick={e => { e.stopPropagation(); setPdfFile(null); setResult(null) }}>Remove</button>
                </div>
              ) : (
                <div className="drop-inner">
                  <UploadIcon />
                  <p className="drop-primary">Drop your PDF here</p>
                  <p className="drop-secondary">or click to browse</p>
                </div>
              )}
            </div>

            <p className="field-label">Job description</p>
            <textarea
              className="jd-field"
              placeholder="Paste the job posting…"
              value={jdText}
              onChange={e => setJdText(e.target.value)}
              rows={9}
            />

            <button className="submit-btn" onClick={handleAnalyze} disabled={!canAnalyze}>
              {loading ? <LoadingDots /> : 'Analyze Resume'}
            </button>

            {error && <p className="err">{error}</p>}
          </div>

          {/* ── Results column ── */}
          <div className="col-result">
            {!result && !loading && (
              <div className="placeholder">
                <PlaceholderIcon />
                <p className="placeholder-title">Results will appear here</p>
                <p className="placeholder-sub">Upload a CV and paste a job description to get started</p>
              </div>
            )}
            {loading && (
              <div className="placeholder">
                <Spinner />
                <p className="placeholder-title">Analyzing your CV…</p>
                <p className="placeholder-sub">This may take a few seconds</p>
              </div>
            )}
            {result && <Results data={result} />}
          </div>

        </div>
      </main>
    </div>
  )
}

function Results({ data }) {
  const { features, predictions, feedback } = data
  const rf  = predictions.random_forest
  const nn  = predictions.neural_network
  const yes = rf.verdict === 'shortlisted'

  return (
    <div className="results">

      <div className={`verdict-card ${yes ? 'v-yes' : 'v-no'}`}>
        <div className="verdict-icon">{yes ? <CheckIcon /> : <XIcon />}</div>
        <div className="verdict-body">
          <p className="verdict-word">{yes ? 'Shortlisted' : 'Not Shortlisted'}</p>
          <p className="verdict-note">Consensus from both models</p>
        </div>
        <div className="model-row">
          <div className="model-stat">
            <span className="stat-num">{rf.confidence}%</span>
            <span className="stat-name">Random Forest</span>
          </div>
          <div className="divider-v" />
          <div className="model-stat">
            <span className="stat-num">{nn.confidence}%</span>
            <span className="stat-name">Neural Network</span>
          </div>
        </div>
      </div>

      <p className="block-title">Extracted Profile</p>
      <div className="feat-grid">
        {FEATURE_META.map(({ key, label, fmt }) => (
          <div className="feat-cell" key={key}>
            <span className="feat-val">{fmt(features[key])}</span>
            <span className="feat-key">{label}</span>
          </div>
        ))}
      </div>

      {!yes && feedback?.length === 0 && (
        <div className="no-improve-note">
          <p className="no-improve-title">No specific gaps found for this role</p>
          <p className="no-improve-body">This candidate meets the job's stated requirements. The model's decision is likely driven by patterns in its training data rather than a specific deficiency — most notably, limited formal work experience compared to candidates in the dataset.</p>
        </div>
      )}

      {feedback?.length > 0 && (
        <>
          <p className="block-title" style={{ marginBottom: 12 }}>What to improve</p>
          <div className="improve-list">
            {feedback.map((item, i) => (
              <div className="improve-card" key={i}>
                <div className="improve-left">
                  <span className="improve-label">{item.label}</span>
                  <span className={`improve-source ${item.source === 'job description' ? 'src-jd' : 'src-median'}`}>
                    {item.source === 'job description' ? 'from job description' : 'shortlisted median'}
                  </span>
                </div>
                <div className="improve-values">
                  <span className="improve-current">{item.current}</span>
                  <span className="improve-arrow">→</span>
                  <span className="improve-target">{item.benchmark}</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function LoadingDots() {
  return <span className="dots"><span /><span /><span /></span>
}
function Spinner() {
  return <span className="spinner" />
}
function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}
function PdfIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}
function XIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}
function PlaceholderIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#ccc' }}>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  )
}
