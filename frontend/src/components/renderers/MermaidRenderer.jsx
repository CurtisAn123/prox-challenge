/**
 * MermaidRenderer.jsx
 * ─────────────────────
 * Renders Mermaid.js diagram syntax into an SVG diagram.
 *
 * Key design decisions:
 *  1. mermaid.initialize() is called ONCE at module load, not on every render.
 *  2. Each diagram gets a unique DOM id via a module-level counter so Mermaid
 *     never confuses two diagrams rendered in the same session.
 *  3. On render failure, we fall back to displaying the raw syntax in a
 *     <pre> block rather than crashing or showing nothing.
 *  4. Re-renders (syntax prop changes) cancel any in-flight render and
 *     restart via the useEffect dependency array.
 *  5. An expand button opens a full-screen modal (via React portal) with
 *     zoom/pan support: scroll-wheel to zoom, drag to pan, double-click to reset.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import mermaid from 'mermaid'

// Initialise once at module load — before any component mounts
mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    primaryColor: '#dbeafe',
    primaryTextColor: '#1e3a5f',
    primaryBorderColor: '#93c5fd',
    lineColor: '#64748b',
    secondaryColor: '#f0fdf4',
    tertiaryColor: '#fef3c7',
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: '16px',
  },
  flowchart: { curve: 'basis', padding: 20 },
  securityLevel: 'loose',    // required to render in React
})

// Module-level counter so each diagram instance gets a unique id
let _diagramCounter = 0

// ── Shared render helper ──────────────────────────────────────────────────────

async function renderInto(containerEl, syntax) {
  // Always clear first and generate a fresh id — Mermaid v11 throws if it finds
  // an existing DOM element with the same id from a previous render.
  containerEl.innerHTML = ''
  const id = `mermaid-${++_diagramCounter}`
  const { svg } = await mermaid.render(id, syntax.trim())
  containerEl.innerHTML = svg
  const svgEl = containerEl.querySelector('svg')
  if (svgEl) {
    svgEl.removeAttribute('height')
    svgEl.style.maxWidth = '100%'
  }
}

// ── Zoom/pan hook ─────────────────────────────────────────────────────────────

function useZoomPan() {
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragRef = useRef(null) // { startX, startY, baseX, baseY }

  const reset = useCallback(() => {
    setScale(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  const handleWheel = useCallback((e) => {
    e.preventDefault()
    setScale(s => Math.min(5, Math.max(0.5, s + e.deltaY * -0.001)))
  }, [])

  const handleMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    setIsDragging(true)
    const startX = e.clientX
    const startY = e.clientY
    // Capture current offset at drag start via functional update
    setOffset(curr => {
      dragRef.current = { startX, startY, baseX: curr.x, baseY: curr.y }
      return curr
    })
  }, [])

  const handleMouseMove = useCallback((e) => {
    if (!dragRef.current) return
    setOffset({
      x: dragRef.current.baseX + (e.clientX - dragRef.current.startX),
      y: dragRef.current.baseY + (e.clientY - dragRef.current.startY),
    })
  }, [])

  const handleMouseUp = useCallback(() => {
    dragRef.current = null
    setIsDragging(false)
  }, [])

  return {
    scale, setScale, offset, isDragging,
    reset, handleWheel, handleMouseDown, handleMouseMove, handleMouseUp,
  }
}

// ── Expand icon (simple SVG) ──────────────────────────────────────────────────

function ExpandIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  )
}

// ── Fullscreen modal with zoom/pan ────────────────────────────────────────────

function DiagramModal({ syntax, title, onClose }) {
  const containerRef = useRef(null)
  const outerRef = useRef(null)
  const {
    scale, setScale, offset, isDragging,
    reset, handleWheel, handleMouseDown, handleMouseMove, handleMouseUp,
  } = useZoomPan()

  // Render the diagram inside the modal
  useEffect(() => {
    if (!containerRef.current || !syntax?.trim()) return
    let cancelled = false
    renderInto(containerRef.current, syntax).catch(() => {
      if (!cancelled && containerRef.current) {
        containerRef.current.innerHTML = `<pre style="font-size:12px;padding:1rem">${syntax}</pre>`
      }
    })
    return () => { cancelled = true }
  }, [syntax])

  // Attach non-passive wheel listener so preventDefault works (React's onWheel is passive)
  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="
          bg-white dark:bg-slate-800 rounded-2xl shadow-2xl
          max-w-[92vw] w-[min(960px,92vw)] h-[85vh]
          p-6 flex flex-col gap-4
        "
        onClick={e => e.stopPropagation()}
      >
        {/* Header: title + zoom controls + close */}
        <div className="flex items-center gap-2 shrink-0">
          {title && (
            <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-300 truncate flex-1 min-w-0">
              {title}
            </h3>
          )}

          {/* Zoom controls */}
          <div className="flex items-center gap-1 ml-auto shrink-0">
            <button
              onClick={() => setScale(s => Math.max(0.5, +(s - 0.25).toFixed(2)))}
              className="w-7 h-7 flex items-center justify-center rounded text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 text-lg font-bold leading-none select-none"
              title="Zoom out (or scroll)"
            >−</button>
            <button
              onClick={reset}
              className="px-2 h-7 flex items-center justify-center rounded text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 text-xs font-mono min-w-[3.5rem] select-none"
              title="Reset zoom and position"
            >{Math.round(scale * 100)}%</button>
            <button
              onClick={() => setScale(s => Math.min(5, +(s + 0.25).toFixed(2)))}
              className="w-7 h-7 flex items-center justify-center rounded text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 text-lg font-bold leading-none select-none"
              title="Zoom in (or scroll)"
            >+</button>
          </div>

          {/* Close */}
          <button
            onClick={onClose}
            className="
              shrink-0 w-7 h-7 flex items-center justify-center
              rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-100
              dark:hover:text-slate-200 dark:hover:bg-slate-700
              transition-colors text-base leading-none
            "
            aria-label="Close diagram"
          >
            ✕
          </button>
        </div>

        {/* Diagram canvas — overflow hidden, zoom/pan applied via transform */}
        <div
          ref={outerRef}
          className="flex-1 min-h-0 overflow-hidden rounded-lg bg-slate-50 dark:bg-slate-900"
          style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={reset}
          title="Scroll to zoom · Drag to pan · Double-click to reset"
        >
          <div
            ref={containerRef}
            className="w-full h-full flex justify-center items-start p-4"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: 'center top',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out',
              userSelect: 'none',
              willChange: 'transform',
            }}
            aria-label={title ?? 'Diagram (expanded)'}
          />
        </div>
      </div>
    </div>,
    document.body,
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function MermaidRenderer({ syntax, title }) {
  const containerRef = useRef(null)
  const [renderError, setRenderError] = useState(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!containerRef.current || !syntax?.trim()) return

    let cancelled = false

    async function render() {
      try {
        await renderInto(containerRef.current, syntax)
        if (!cancelled) setRenderError(null)
      } catch (err) {
        if (cancelled) return
        console.error('[MermaidRenderer] render failed:', err)
        setRenderError(err.message ?? 'Diagram render failed')
      }
    }

    render()

    return () => { cancelled = true }
  }, [syntax])

  // Expand button — always visible, shown in both success and error states
  const expandButton = (
    <button
      onClick={() => setExpanded(true)}
      title="Expand diagram"
      className="
        absolute top-1 right-1
        w-7 h-7 flex items-center justify-center
        rounded-md bg-white/80 dark:bg-slate-700/80
        text-slate-500 dark:text-slate-300
        hover:bg-white dark:hover:bg-slate-600
        border border-slate-200 dark:border-slate-600
        shadow-sm transition-colors
      "
      aria-label="Expand diagram"
    >
      <ExpandIcon />
    </button>
  )

  if (renderError) {
    return (
      <div className="space-y-2">
        <div className="relative">
          <p className="text-xs text-amber-600 font-medium">
            ⚠ Diagram render failed — showing raw syntax:
          </p>
          {expandButton}
        </div>
        <pre className="text-xs bg-slate-50 border border-slate-200 rounded-lg p-3 overflow-x-auto font-mono leading-relaxed">
          {syntax}
        </pre>
        {expanded && <DiagramModal syntax={syntax} title={title} onClose={() => setExpanded(false)} />}
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {title && (
        <p className="text-xs text-slate-500 font-medium text-center">{title}</p>
      )}

      {/* Diagram + expand button wrapper */}
      <div className="relative">
        <div
          ref={containerRef}
          className="mermaid-output flex justify-center overflow-x-auto"
          aria-label={title ?? 'Diagram'}
        />
        {expandButton}
      </div>

      {/* Fullscreen modal */}
      {expanded && (
        <DiagramModal
          syntax={syntax}
          title={title}
          onClose={() => setExpanded(false)}
        />
      )}
    </div>
  )
}
