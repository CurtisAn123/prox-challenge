/**
 * ChatInput.jsx
 * ──────────────
 * Input bar at the bottom of the chat.
 *
 * Features:
 *  - Auto-expanding textarea (grows up to 5 lines, then scrolls)
 *  - Enter to submit, Shift+Enter for newline
 *  - Disabled while a response is loading
 *  - Character count indicator for long queries
 */

import { useRef, useState, useEffect, useCallback } from 'react'

const MAX_CHARS = 1000

export function ChatInput({ onSubmit, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  // Resize the textarea to fit its content (up to a max height)
  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`  // max 128px ≈ 5 lines
  }, [])

  useEffect(() => { resize() }, [text, resize])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSubmit(trimmed)
    setText('')
  }

  const remaining = MAX_CHARS - text.length
  const nearLimit = remaining < 100

  return (
    <div className="border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-3">
      <div className="max-w-3xl mx-auto">
        <div className="
          flex items-end gap-2
          bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-2xl
          px-4 py-2.5 shadow-sm
          focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100 dark:focus-within:ring-brand-900
          transition
        ">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => setText(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Ask about your Vulcan OmniPro 220…"
            rows={1}
            className="
              flex-1 resize-none bg-transparent
              text-sm text-slate-800 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-600
              focus:outline-none disabled:opacity-50
              leading-relaxed py-0.5
            "
            style={{ minHeight: '24px' }}
            aria-label="Chat input"
          />

          {/* Character count (only shows when approaching limit) */}
          {nearLimit && (
            <span className={`text-xs shrink-0 mb-0.5 ${remaining < 20 ? 'text-red-400' : 'text-slate-400'}`}>
              {remaining}
            </span>
          )}

          {/* Submit button */}
          <button
            onClick={handleSubmit}
            disabled={!text.trim() || disabled}
            aria-label="Send message"
            className="
              shrink-0 w-8 h-8 rounded-xl mb-0.5
              bg-brand-600 text-white
              hover:bg-brand-700 active:scale-95
              disabled:opacity-40 disabled:cursor-not-allowed
              transition flex items-center justify-center
            "
          >
            {disabled ? (
              // Spinner while loading
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
            ) : (
              // Arrow icon
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>

        <p className="text-center text-xs text-slate-400 dark:text-slate-600 mt-2">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
