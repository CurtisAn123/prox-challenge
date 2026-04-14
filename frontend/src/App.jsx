/**
 * App.jsx
 * ────────
 * Root component. Owns layout, dark mode, and manual filter state.
 *
 *   ┌─────────────────────────────────┐
 *   │  Header (title + controls)      │
 *   ├─────────────────────────────────┤
 *   │                                 │
 *   │  MessageList (scrollable)       │
 *   │                                 │
 *   ├─────────────────────────────────┤
 *   │  ManualFilter (selected manual) │
 *   ├─────────────────────────────────┤
 *   │  ChatInput (fixed bottom)       │
 *   └─────────────────────────────────┘
 */

import { useState, useEffect } from 'react'
import { MessageList } from './components/MessageList'
import { ChatInput } from './components/ChatInput'
import { useChat } from './hooks/useChat'
import { getHealth } from './api/client'

const MANUALS = [
  { id: 'vulcan-220', label: 'Vulcan OmniPro 220', icon: '🔥' },
]

export default function App() {
  const { messages, isLoading, submit, clear } = useChat()
  const [health, setHealth] = useState(null)
  const [selectedManual] = useState('vulcan-220')

  // ── Dark mode — persisted to localStorage, defaults to dark ──────────────
  const [isDark, setIsDark] = useState(() => {
    const stored = localStorage.getItem('theme')
    if (stored) return stored === 'dark'
    return true  // dark by default
  })

  useEffect(() => {
    const root = document.documentElement
    if (isDark) {
      root.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [isDark])

  // ── Poll backend health on mount ───────────────────────────────────────────
  useEffect(() => {
    let cancelled = false

    async function checkHealth() {
      try {
        const data = await getHealth()
        if (!cancelled) setHealth(data)
      } catch {
        if (!cancelled) setHealth({ status: 'unreachable' })
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 10_000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  return (
    <div className="h-full flex flex-col bg-surface-50 dark:bg-slate-900">

      {/* ── Top bar ────────────────────────────────────────────────────── */}
      <header className="shrink-0 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-4 py-2.5">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">🔥</span>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 dark:text-slate-100 leading-none">
                OmniPro 220 Assistant
              </h1>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                Vulcan OmniPro 220 · Expert AI
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <StatusBadge health={health} />

            {/* Dark mode toggle */}
            <button
              onClick={() => setIsDark(d => !d)}
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              className="w-7 h-7 rounded-lg flex items-center justify-center text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition"
            >
              {isDark ? <SunIcon /> : <MoonIcon />}
            </button>

            {messages.length > 0 && (
              <button
                onClick={clear}
                className="text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition px-2 py-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Message list ───────────────────────────────────────────────── */}
      <MessageList messages={messages} />

      {/* ── Manual filter ──────────────────────────────────────────────── */}
      <ManualFilter selected={selectedManual} />

      {/* ── Input ──────────────────────────────────────────────────────── */}
      <ChatInput onSubmit={submit} disabled={isLoading} />
    </div>
  )
}

// ── Manual filter strip ───────────────────────────────────────────────────────

function ManualFilter({ selected }) {
  return (
    <div className="shrink-0 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 px-4 py-2">
      <div className="max-w-3xl mx-auto flex items-center gap-2.5">
        <span className="text-xs text-slate-400 dark:text-slate-500 shrink-0 font-medium">
          Manual
        </span>
        <div className="flex gap-1.5 flex-wrap">
          {MANUALS.map(m => (
            <span
              key={m.id}
              className={`
                inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
                text-xs font-medium border transition-colors
                ${m.id === selected
                  ? 'bg-brand-50 dark:bg-blue-950 text-brand-600 dark:text-blue-400 border-brand-200 dark:border-blue-800'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-600'
                }
              `}
            >
              {m.id === selected && (
                <svg className="w-3 h-3 shrink-0" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
                  <path d="M10 3L5 8.5 2 5.5l1-1 2 2 4-4 1 1z" />
                </svg>
              )}
              <span>{m.icon}</span>
              <span>{m.label}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ health }) {
  if (!health) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-slate-400">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse" />
        Connecting…
      </span>
    )
  }

  if (health.status === 'unreachable') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-red-500">
        <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
        Backend offline
      </span>
    )
  }

  if (health.status === 'ready') {
    const chunks = health.vector_store?.text_chunks ?? 0
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-500">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
        Ready · {chunks} chunks
      </span>
    )
  }

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-amber-600">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
      Not ready
    </span>
  )
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function MoonIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
    </svg>
  )
}

function SunIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="5" />
      <path strokeLinecap="round"
        d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  )
}
