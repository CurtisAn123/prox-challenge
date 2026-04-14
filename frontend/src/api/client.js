/**
 * api/client.js
 * ─────────────
 * Thin API client for the FastAPI backend.
 *
 * All fetch calls go here so the rest of the app never touches raw URLs.
 * The Vite dev proxy forwards these to http://localhost:8000.
 */

const BASE = import.meta.env.VITE_API_URL ?? ''

/**
 * Send a user query to the agent router.
 *
 * @param {string} query - The user's natural language question.
 * @returns {Promise<{intent: string, type: string, sources: Array, content: object}>}
 */
export async function sendQuery(query) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  return res.json()
}

/**
 * Stream a query through the agent router, receiving progress events
 * as the agent works, then the final response.
 *
 * @param {string} query - The user's question.
 * @param {(event: {type: string, [key: string]: any}) => void} onEvent
 *   Called for each intermediate event (planning, thinking, tool_call).
 * @returns {Promise<{intent: string, type: string, sources: Array, content: object}>}
 *   Resolves with the final QueryResponse when the agent finishes.
 */
export async function streamQuery(query, onEvent) {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    // SSE lines are separated by \n\n; split on single \n and look for "data: " prefix
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep any incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const event = JSON.parse(line.slice(6))
        if (event.type === 'done') {
          finalResult = event.result
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Agent error')
        } else {
          onEvent(event)
        }
      } catch (err) {
        if (err.message !== 'Agent error' && !err.message?.startsWith('HTTP')) {
          // JSON parse error — skip malformed line
        } else {
          throw err
        }
      }
    }
  }

  if (!finalResult) throw new Error('Stream ended without a response')
  return finalResult
}

/**
 * Fetch system health / readiness status.
 *
 * @returns {Promise<{status: string, vector_store: object, graph: object}>}
 */
export async function getHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
