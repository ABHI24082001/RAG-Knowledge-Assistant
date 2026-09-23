const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const DEFAULT_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function friendlyMessage(status, detail) {
  if (status === 413) return 'The PDF is too large. Please choose a file under 20 MB.'
  if (status === 400 || status === 422) return detail || 'Please check the information and try again.'
  if (status === 404) return 'The requested resource was not found.'
  if (status >= 500) return 'The server could not complete the request. Please try again.'
  return detail || 'Backend is currently unavailable. Please try again.'
}

async function request(path, options = {}) {
  if (!API_URL) throw new ApiError('API URL is not configured. Set VITE_API_URL and restart Vite.')

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeout ?? DEFAULT_TIMEOUT_MS)
  try {
    const response = await fetch(`${API_URL}${path}`, { ...options, signal: controller.signal })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) throw new ApiError(friendlyMessage(response.status, data.detail), response.status)
    return data
  } catch (error) {
    if (error.name === 'AbortError') throw new ApiError('The request timed out. Please try again.')
    if (error instanceof ApiError) throw error
    console.error('API request failed:', error)
    throw new ApiError('Backend is currently unavailable. Please try again.')
  } finally {
    clearTimeout(timeout)
  }
}

const pdfBody = (file) => {
  const body = new FormData()
  body.append('file', file)
  return body
}

export const api = {
  root: () => request('/'),
  health: () => request('/health'),
  qdrantHealth: () => request('/qdrant/health'),
  createCollection: () => request('/qdrant/collection', { method: 'POST' }),
  embeddingTest: () => request('/embedding/test'),
  addVector: (payload) => request('/vectors/add', jsonOptions('POST', payload)),
  searchVectors: (query, limit = 3) => request('/vectors/search', jsonOptions('POST', { query, limit })),
  extractDocument: (file) => request('/documents/extract', { method: 'POST', body: pdfBody(file), timeout: 60_000 }),
  previewChunks: (file) => request('/documents/chunk-preview', { method: 'POST', body: pdfBody(file), timeout: 60_000 }),
  indexDocument: (file) => request('/documents/index', { method: 'POST', body: pdfBody(file), timeout: 180_000 }),
  listDocuments: () => request('/documents'),
  deleteDocument: (documentId) => request(`/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' }),
  chat: (payload) => request('/chat', jsonOptions('POST', payload, 180_000)),
}

function jsonOptions(method, body, timeout) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    ...(timeout ? { timeout } : {}),
  }
}
