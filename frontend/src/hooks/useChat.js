/**
 * hooks/useChat.js
 * ──────────────────
 * State management for the chat thread.
 *
 * Message shape:
 *   User:      { id, role: 'user', text: string }
 *   Assistant: { id, role: 'assistant', response: BackendResponse, error?: string }
 *   Loading:   { id, role: 'assistant', loading: true, statusText: string }
 */

import { useState, useCallback } from 'react'
import { streamQuery } from '../api/client'

let _id = 0
const uid = () => `msg-${++_id}`

// Maps backend tool names to human-readable status labels
const TOOL_LABELS = {
  search_text:                'Searching manual...',
  search_kg_entity:           'Checking knowledge graph...',
  find_kg_path:               'Tracing relationships...',
  retrieve_image:             'Searching for diagrams...',
  analyze_image_with_context: 'Analyzing diagram...',
  calculate_duty_cycle:       'Calculating duty cycle...',
  generate_diagram:           'Drawing diagram...',
  return_widget:              'Building configurator...',
  finish:                     'Composing answer...',
}

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const append = useCallback((msg) => {
    setMessages(prev => [...prev, msg])
  }, [])

  const updateLast = useCallback((updater) => {
    setMessages(prev => {
      const next = [...prev]
      next[next.length - 1] = updater(next[next.length - 1])
      return next
    })
  }, [])

  /**
   * Send a query through the agent router via the SSE streaming endpoint.
   * Progress events update the loading message's statusText in real time.
   * The loading message is replaced in-place by the final response.
   */
  const submit = useCallback(async (text) => {
    if (!text.trim() || isLoading) return

    // Append user message immediately
    append({ id: uid(), role: 'user', text: text.trim() })

    // Append a loading placeholder with initial status
    const loadingId = uid()
    append({ id: loadingId, role: 'assistant', loading: true, statusText: 'Thinking...' })
    setIsLoading(true)

    try {
      const response = await streamQuery(text.trim(), (event) => {
        if (event.type === 'planning') {
          updateLast(msg => ({ ...msg, statusText: 'Planning...' }))
        } else if (event.type === 'thinking') {
          updateLast(msg => ({ ...msg, statusText: 'Thinking...' }))
        } else if (event.type === 'tool_call') {
          const label = TOOL_LABELS[event.tool] ?? 'Working...'
          updateLast(msg => ({ ...msg, statusText: label }))
        }
      })

      // Replace loading placeholder with the real response
      updateLast(() => ({ id: loadingId, role: 'assistant', response }))
    } catch (err) {
      updateLast(() => ({
        id: loadingId,
        role: 'assistant',
        error: err.message || 'Something went wrong. Is the backend running?',
      }))
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, append, updateLast])

  const clear = useCallback(() => setMessages([]), [])

  return { messages, isLoading, submit, clear }
}
