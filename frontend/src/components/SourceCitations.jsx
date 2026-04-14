/**
 * SourceCitations.jsx
 * ─────────────────────
 * Collapsible source citation list shown below assistant responses.
 *
 * Keeps the chat clean by default (collapsed) but lets power users
 * verify exactly which manual pages the answer came from.
 */

import { useState } from 'react'

export function SourceCitations({ sources }) {
  const [open, setOpen] = useState(false)

  // Deduplicate sources (same page/file may appear multiple times from chunking)
  const unique = deduplicateSources(sources)

  return (
    <div className="pl-0.5">
      <button
        onClick={() => setOpen(o => !o)}
        className="
          text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-400
          flex items-center gap-1 transition-colors
        "
        aria-expanded={open}
      >
        <span className="text-[10px]">{open ? '▾' : '▸'}</span>
        {unique.length} source{unique.length !== 1 ? 's' : ''}
      </button>

      {open && (
        <ul className="mt-1.5 space-y-0.5 animate-fade-in">
          {unique.map((s, i) => (
            <li key={i} className="flex items-center gap-1.5 text-xs text-slate-400 dark:text-slate-500">
              <span className="w-1 h-1 rounded-full bg-slate-300 dark:bg-slate-600 shrink-0" />
              <span>
                {s.source}
                {s.page > 0 && (
                  <span className="ml-1 font-mono text-slate-300 dark:text-slate-600">p.{s.page}</span>
                )}
                {s.section && (
                  <span className="ml-1 italic">— {s.section}</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function deduplicateSources(sources) {
  const seen = new Set()
  return sources.filter(s => {
    const key = `${s.source}:${s.page}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
