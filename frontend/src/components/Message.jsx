/**
 * Message.jsx
 * ─────────────
 * Renders a single message in the chat thread.
 *
 * Dispatch logic (the frontend's counterpart to the backend router):
 *
 *   response.type === "text"    → TextRenderer
 *   response.type === "mermaid" → artifact card + MermaidRenderer
 *   response.type === "image"   → artifact card + ImageRenderer
 *   response.type === "widget"  → WidgetRenderer (manages its own card)
 *
 * Plus: loading state, error state, source citations.
 */

import { MermaidRenderer } from './renderers/MermaidRenderer'
import { ImageRenderer } from './renderers/ImageRenderer'
import { TextRenderer } from './renderers/TextRenderer'
import { WidgetRenderer } from './renderers/WidgetRenderer'
import { SourceCitations } from './SourceCitations'

// ── Intent → icon label map ──────────────────────────────────────────────────
const INTENT_META = {
  text_qa:  { icon: '💬', label: 'Answer' },
  diagram:  { icon: '📐', label: 'Diagram' },
  image:    { icon: '🖼',  label: 'Manual Image' },
  widget:   { icon: '🧮', label: 'Interactive Tool' },
  rich:     { icon: '📋', label: 'Full Breakdown' },
}

// ── User bubble ───────────────────────────────────────────────────────────────

export function UserMessage({ text }) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="
        max-w-[80%] lg:max-w-[65%] px-4 py-2.5
        bg-brand-600 text-white rounded-2xl rounded-tr-md
        text-sm leading-relaxed shadow-sm
      ">
        {text}
      </div>
    </div>
  )
}

// ── Assistant bubble ──────────────────────────────────────────────────────────

export function AssistantMessage({ response, error, loading, statusText }) {
  // ── Loading state ──────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex items-start gap-3 animate-fade-in">
        <AgentAvatar />
        <div className="px-4 py-3 bg-white dark:bg-slate-800 rounded-2xl rounded-tl-md border border-slate-200 dark:border-slate-700 shadow-sm">
          <div className="flex items-center gap-2 h-5">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
            {statusText && (
              <span
                key={statusText}
                className="text-xs text-slate-400 dark:text-slate-500 ml-1 animate-fade-in"
              >
                {statusText}
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  // ── Error state ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="flex items-start gap-3 animate-slide-up">
        <AgentAvatar />
        <div className="
          px-4 py-3 bg-red-50 border border-red-200
          rounded-2xl rounded-tl-md text-sm text-red-600
          max-w-[80%] lg:max-w-[65%]
        ">
          <span className="font-semibold">Error: </span>{error}
        </div>
      </div>
    )
  }

  if (!response) return null

  const { type, intent, sources, content } = response
  const meta = INTENT_META[intent] ?? INTENT_META.text_qa

  return (
    <div className="flex items-start gap-3 animate-slide-up">
      <AgentAvatar />

      <div className="flex-1 min-w-0 max-w-[85%] lg:max-w-[75%] space-y-2">
        {/* Response body — type-switched */}
        <ResponseBody type={type} content={content} meta={meta} />

        {/* Source citations (collapsed by default) */}
        {sources?.length > 0 && <SourceCitations sources={sources} />}
      </div>
    </div>
  )
}

// ── Response body switcher ────────────────────────────────────────────────────

function ResponseBody({ type, content, meta }) {
  switch (type) {
    case 'text':
      return (
        <div className="
          px-4 py-3 bg-white dark:bg-slate-800 rounded-2xl rounded-tl-md
          border border-slate-200 dark:border-slate-700 shadow-sm
        ">
          <TextRenderer text={content.answer} />
        </div>
      )

    case 'mermaid':
      return (
        <div className="artifact-card">
          <div className="artifact-header">
            <span>{meta.icon}</span>
            <span>{content.title || meta.label}</span>
          </div>
          <div className="artifact-body">
            <MermaidRenderer syntax={content.syntax} title={content.title} />
          </div>
        </div>
      )

    case 'image':
      return (
        <div className="
          px-4 py-3 bg-white dark:bg-slate-800 rounded-2xl rounded-tl-md
          border border-slate-200 dark:border-slate-700 shadow-sm
        ">
          {content.answer && <TextRenderer text={content.answer} />}
          {content.images?.length > 0 && (
            <div className="mt-3">
              <ImageRenderer images={content.images} />
            </div>
          )}
        </div>
      )

    case 'widget':
      return <WidgetRenderer content={content} />

    case 'rich':
      // Composite response: text + manual images in one unified box; Mermaid diagram below.
      return (
        <div className="space-y-3">
          <div className="
            px-4 py-3 bg-white dark:bg-slate-800 rounded-2xl rounded-tl-md
            border border-slate-200 dark:border-slate-700 shadow-sm
          ">
            <TextRenderer text={content.answer} />
            {content.images?.length > 0 && (
              <div className="mt-3">
                <ImageRenderer images={content.images} />
              </div>
            )}
          </div>

          {content.mermaid?.syntax && (
            <div className="artifact-card">
              <div className="artifact-header">
                <span>📐</span>
                <span>{content.mermaid.title || 'Wiring Diagram'}</span>
              </div>
              <div className="artifact-body">
                <MermaidRenderer syntax={content.mermaid.syntax} title={content.mermaid.title} />
              </div>
            </div>
          )}
        </div>
      )

    default:
      // Catch-all: render raw JSON for debugging unknown types
      return (
        <div className="artifact-card">
          <div className="artifact-header">
            <span>📦</span><span>Response</span>
          </div>
          <div className="artifact-body">
            <pre className="text-xs font-mono overflow-x-auto">
              {JSON.stringify(content, null, 2)}
            </pre>
          </div>
        </div>
      )
  }
}

// ── Avatar ─────────────────────────────────────────────────────────────────

function AgentAvatar() {
  return (
    <div className="
      shrink-0 w-8 h-8 rounded-full
      bg-gradient-to-br from-brand-500 to-blue-700
      flex items-center justify-center
      text-white text-xs font-bold shadow-sm mt-0.5
    ">
      V
    </div>
  )
}
