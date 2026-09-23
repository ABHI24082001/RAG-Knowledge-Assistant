import { useEffect, useRef, useState } from 'react'
import { api } from './services/api'
import { buildChatPayload, NO_ANSWER } from './utils/chat'
import './App.css'
import './integrations.css'

const Icons = ({ name, size = 20 }) => {
  const shapes = {
    brain: <><path d="M8.5 4.5a3.5 3.5 0 0 0-3.4 4.3A3.7 3.7 0 0 0 6 16a3.6 3.6 0 0 0 6-2.7A3.6 3.6 0 0 0 16 9.7a3.5 3.5 0 0 0-4.8-5.2"/><path d="M12 4v16M8 8h4M8 13h4"/></>,
    upload: <><path d="M12 16V4M8 8l4-4 4 4"/><path d="M5 14v5h14v-5"/></>, file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></>,
    trash: <><path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3"/></>, send: <path d="m22 2-7 20-4-9-9-4Z M11 13l5-5"/>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></>, moon: <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z"/>,
    menu: <path d="M4 7h16M4 12h16M4 17h16"/>, close: <path d="m6 6 12 12M18 6 6 18"/>, database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></>,
    chevron: <path d="m6 9 6 6 6-6"/>, check: <path d="m5 12 4 4L19 6"/>, info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>, search: <><circle cx="11" cy="11" r="6"/><path d="m20 20-4.2-4.2"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{shapes[name]}</svg>
}

function SourceList({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return <div className="sources"><button className="sources-toggle" onClick={() => setOpen(!open)}>Sources ({sources.length}) <span className={open ? 'rotated' : ''}><Icons name="chevron" size={16}/></span></button>{open && <div className="source-grid">{sources.map((source, index) => <div className="source-card" key={`${source.file_name}-${source.chunk_id}-${index}`}><div className="source-file"><Icons name="file" size={15}/>{source.file_name}</div><div className="source-meta"><span>Page {source.page}</span><span>Chunk {source.chunk_id}</span><span>{Math.round(source.score * 100)}% relevant</span></div></div>)}</div>}</div>
}

function App() {
  const [documents, setDocuments] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [deletingId, setDeletingId] = useState('')
  const [connected, setConnected] = useState(false)
  const [qdrantStatus, setQdrantStatus] = useState('Checking connection...')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [dark, setDark] = useState(() => localStorage.getItem('rag-theme') === 'dark')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [chatting, setChatting] = useState(false)
  const [toasts, setToasts] = useState([])
  const [inspection, setInspection] = useState(null)
  const [inspecting, setInspecting] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState([])
  const fileInput = useRef(null)
  const chatEnd = useRef(null)
  const selected = documents.find((doc) => doc.document_id === selectedId)

  const toast = (message, type = 'success') => {
    const id = Date.now() + Math.random()
    setToasts((items) => [...items, { id, message, type }])
    setTimeout(() => setToasts((items) => items.filter((item) => item.id !== id)), 3800)
  }

  const loadDocuments = async () => {
    setLoadingDocs(true)
    try {
      const [data, health, qdrant] = await Promise.all([api.listDocuments(), api.health(), api.qdrantHealth()])
      const nextDocuments = data.documents || []
      setDocuments(nextDocuments)
      setSelectedId((currentId) => (
        currentId && !nextDocuments.some((doc) => doc.document_id === currentId)
          ? ''
          : currentId
      ))
      setConnected(health.status === 'healthy')
      setQdrantStatus(qdrant.status === 'connected' ? 'Vector DB connected' : 'Vector DB unavailable')
    } catch (error) {
      setConnected(false)
      setQdrantStatus('Qdrant offline')
      toast(error.message || 'Backend is currently unavailable. Please try again.', 'error')
    } finally { setLoadingDocs(false) }
  }

  useEffect(() => {
    const requestId = window.setTimeout(() => { void loadDocuments() }, 0)
    return () => window.clearTimeout(requestId)
  }, [])
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light'; localStorage.setItem('rag-theme', dark ? 'dark' : 'light') }, [dark])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, chatting])

  const chooseFile = (candidate) => {
    if (!candidate) return
    if (candidate.type !== 'application/pdf' && !candidate.name.toLowerCase().endsWith('.pdf')) return toast('Only PDF files are supported.', 'error')
    if (candidate.size > 20 * 1024 * 1024) return toast('PDF size must be 20 MB or less.', 'error')
    setInspection(null)
    setFile(candidate)
  }

  const upload = async () => {
    if (!file || uploading) return
    setUploading(true)
    try {
      const result = await api.indexDocument(file)
      toast(`Indexed ${result.file_name}: ${result.chunk_count} chunks across ${result.page_count} pages.`)
      setFile(null)
      await loadDocuments()
      setSelectedId(result.document_id)
    } catch (error) { toast(error.message || 'Failed to index document.', 'error') } finally { setUploading(false) }
  }

  const inspectPdf = async (action, label) => {
    if (!file || inspecting) return
    setInspecting(action)
    try {
      const result = action === 'extract' ? await api.extractDocument(file) : await api.previewChunks(file)
      setInspection({ label, result })
      toast(`${label} completed.`)
    } catch (error) { toast(error.message || `${label} failed.`, 'error') } finally { setInspecting('') }
  }

  const removeDocument = async (event, doc) => {
    event.stopPropagation()
    if (!window.confirm(`Delete “${doc.file_name}” and all of its indexed chunks?`)) return
    setDeletingId(doc.document_id)
    try {
      await api.deleteDocument(doc.document_id)
      setDocuments((items) => items.filter((item) => item.document_id !== doc.document_id))
      if (selectedId === doc.document_id) setSelectedId('')
      toast('Document deleted.')
    } catch (error) { toast(error.message || 'Failed to delete document.', 'error') } finally { setDeletingId('') }
  }

  const send = async () => {
    const text = question.trim()
    if (!text || chatting) return
    setMessages((items) => [...items, { role: 'user', text }])
    setQuestion('')
    setChatting(true)
    try {
      const payload = buildChatPayload(text, selectedId)
      const response = await api.chat(payload)
      setMessages((items) => [...items, {
        role: 'assistant',
        text: response.answer,
        sources: response.sources,
        notFound: response.answer === NO_ANSWER,
      }])
    } catch (error) {
      console.error('Chat request failed', {
        status: error.status || 0,
        responseError: error.responseError || error.message,
      })
      toast(error.message || 'Chat request failed.', 'error')
    } finally { setChatting(false) }
  }

  const search = async (event) => {
    event.preventDefault()
    const text = searchQuery.trim()
    if (!text || searching) return
    setSearching(true)
    try {
      const result = await api.searchVectors(text, 3)
      setSearchResults(result.results || [])
      if (!(result.results || []).length) toast('No matching document sections found.', 'error')
    } catch (error) { toast(error.message || 'Search failed.', 'error') } finally { setSearching(false) }
  }

  const examples = ['Summarize this document', 'What is the main methodology?', 'What dataset was used?', 'What are the key findings?']
  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Icons name="brain" size={23}/></div><div><h1>RAG Knowledge Assistant</h1><p>Retrieval-Augmented Generation</p></div></div><div className="header-actions"><div className={`api-status ${connected ? 'connected' : ''}`}><span/>{connected ? 'API Connected' : 'API Offline'}</div><button className="icon-button theme-toggle" onClick={() => setDark(!dark)} aria-label="Toggle theme"><Icons name={dark ? 'sun' : 'moon'}/></button><button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open documents"><Icons name="menu"/></button></div></header>
    <div className="dashboard">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="mobile-sidebar-header"><strong>Knowledge Base</strong><button className="icon-button" onClick={() => setSidebarOpen(false)}><Icons name="close"/></button></div>
        <section className="knowledge-heading"><div><p className="eyebrow">YOUR DOCUMENTS</p><h2>Knowledge Base</h2><p>Upload and manage your documents</p></div><span className="document-count">{documents.length}</span></section>
        <div className={`upload-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]) }} onClick={() => fileInput.current?.click()}><input ref={fileInput} type="file" accept="application/pdf,.pdf" onChange={(event) => chooseFile(event.target.files[0])}/><div className="upload-icon"><Icons name={file ? 'file' : 'upload'} size={22}/></div>{file ? <><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB ready to index</span></> : <><strong>Drop your PDF here</strong><span>or <b>browse files</b></span><small>PDF up to 20 MB</small></>}</div>
        <div className="file-actions"><button disabled={!file || inspecting} onClick={() => inspectPdf('extract', 'Text extraction')}>{inspecting === 'extract' ? 'Extracting...' : 'Extract text'}</button><button disabled={!file || inspecting} onClick={() => inspectPdf('preview', 'Chunk preview')}>{inspecting === 'preview' ? 'Preparing...' : 'Preview chunks'}</button></div>
        {inspection && <div className="inspection"><div><strong>{inspection.label}</strong><button onClick={() => setInspection(null)}><Icons name="close" size={13}/></button></div><span>{inspection.result.page_count} pages {inspection.result.chunk_count ? `• ${inspection.result.chunk_count} chunks` : `• ${inspection.result.total_characters} characters`}</span></div>}
        <button className="index-button" disabled={!file || uploading} onClick={upload}>{uploading ? <><i className="spinner"/> Indexing document...</> : <><Icons name="upload" size={17}/> Upload &amp; Index</>}</button>
        <section className="document-section"><div className="section-label"><span>INDEXED DOCUMENTS</span><button onClick={loadDocuments} disabled={loadingDocs}>Refresh</button></div>{loadingDocs ? <div className="skeleton-list"><span/><span/><span/></div> : documents.length ? <div className="document-list">{documents.map((doc) => <button className={`document-card ${selectedId === doc.document_id ? 'selected' : ''}`} key={doc.document_id} onClick={() => { setSelectedId(doc.document_id); setSidebarOpen(false) }}><div className="document-icon"><Icons name="file" size={19}/></div><div className="document-info"><strong title={doc.file_name}>{doc.file_name}</strong><span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'} <i>•</i> {doc.chunk_count} chunks</span></div><span className="delete-button" role="button" tabIndex="0" onClick={(event) => removeDocument(event, doc)}>{deletingId === doc.document_id ? <i className="spinner small"/> : <Icons name="trash" size={17}/>}</span></button>)}</div> : <div className="empty-documents"><div><Icons name="database" size={26}/></div><strong>No documents indexed yet</strong><p>Upload a PDF to start building your knowledge base.</p></div>}</section>
        <div className="system-status"><span className={connected ? 'online-dot' : ''}/><div><strong>{connected ? 'Backend healthy' : 'Backend offline'}</strong><small>{qdrantStatus}</small></div></div>
      </aside><div className={`backdrop ${sidebarOpen ? 'show' : ''}`} onClick={() => setSidebarOpen(false)}/>
      <main className="chat-panel">
        <div className="chat-header"><div><p className="eyebrow">GROUND YOUR ANSWERS</p><h2>Ask Your Knowledge Base</h2><p>Get grounded answers from your indexed documents</p></div><div className="scope-select"><label>Scope</label><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}><option value="">All documents</option>{documents.map((doc) => <option value={doc.document_id} key={doc.document_id}>{doc.file_name}</option>)}</select></div></div>
        {selected && <div className="active-document"><Icons name="file" size={16}/> Chatting with: <strong>{selected.file_name}</strong><button onClick={() => setSelectedId('')}><Icons name="close" size={14}/></button></div>}
        <form className="vector-search" onSubmit={search}><Icons name="search" size={17}/><input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search indexed document sections" aria-label="Search indexed document sections"/><button disabled={!searchQuery.trim() || searching}>{searching ? 'Searching...' : 'Search'}</button></form>
        {searchResults.length > 0 && <div className="search-results">{searchResults.map((result) => <article key={result.id} className="search-result"><div><strong>{result.file_name || 'Stored text'}</strong><span>{result.page ? `Page ${result.page} · ` : ''}{result.chunk_id ? `Chunk ${result.chunk_id} · ` : ''}{Math.round(result.score * 100)}% match</span></div><p>{result.text}</p></article>)}</div>}
        <div className="conversation">{messages.length === 0 ? <div className="welcome"><div className="welcome-icon"><Icons name="brain" size={29}/></div><h3>Ask anything about your documents</h3><p>Your answers are generated using retrieved context from your indexed PDFs.</p><div className="question-chips">{examples.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}</button>)}</div></div> : messages.map((message, index) => <div className={`message ${message.role}`} key={index}>{message.role === 'assistant' && <div className="avatar"><Icons name="brain" size={17}/></div>}<div className="message-content"><div className="bubble">{message.text}</div>{message.notFound && <div className="not-found-note"><Icons name="info" size={15}/> No sufficiently relevant context was found.</div>}{message.role === 'assistant' && <SourceList sources={message.sources}/>}</div></div>)}{chatting && <div className="message assistant"><div className="avatar"><Icons name="brain" size={17}/></div><div className="typing"><div>Searching knowledge base...</div><p><i/><i/><i/> Generating grounded response</p></div></div>}<div ref={chatEnd}/></div>
        <div className="chat-composer"><div className="composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send() } }} placeholder="Ask a question about your documents..." rows="1" disabled={chatting}/><button className="send-button" disabled={!question.trim() || chatting} onClick={send}><Icons name="send" size={18}/></button></div><div className="composer-hint"><span>Answers are grounded in your indexed knowledge base</span><kbd>Enter</kbd> to send <b>•</b> <kbd>Shift + Enter</kbd> for new line</div></div>
      </main>
    </div><div className="toasts">{toasts.map((item) => <div className={`toast ${item.type}`} key={item.id}><span>{item.type === 'success' ? <Icons name="check" size={17}/> : <Icons name="info" size={17}/>}</span>{item.message}</div>)}</div>
  </div>
}

export default App
