/**
 * MessageList.jsx
 * ─────────────────
 * Scrollable list of chat messages. Auto-scrolls to the bottom whenever
 * a new message is added.
 */

import { useEffect, useRef } from 'react'
import { UserMessage, AssistantMessage } from './Message'

export function MessageList({ messages, onSuggest }) {
  const bottomRef = useRef(null)

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-slate-900">
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">

        {/* Empty state: welcoming hero + suggested queries */}
        {messages.length === 0 && (
          <EmptyState onSuggest={onSuggest} />
        )}

        {/* Messages */}
        {messages.map(msg =>
          msg.role === 'user' ? (
            <UserMessage key={msg.id} text={msg.text} />
          ) : (
            <AssistantMessage
              key={msg.id}
              response={msg.response}
              error={msg.error}
              loading={msg.loading}
              statusText={msg.statusText}
            />
          )
        )}

        {/* Invisible anchor for auto-scroll */}
        <div ref={bottomRef} className="h-1" />
      </div>
    </div>
  )
}

function EmptyState({ onSuggest }) {
  return (
    <div className="py-8 text-center space-y-8 animate-fade-in">
      {/* Hero */}
      <div className="space-y-2">
        <div className="text-4xl">🔥</div>
        <h1 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
          Vulcan OmniPro 220 Assistant
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
          Ask anything about your welder. I can answer questions, draw diagrams,
          show manual images, and launch interactive calculators.
        </p>
      </div>

    </div>
  )
}
