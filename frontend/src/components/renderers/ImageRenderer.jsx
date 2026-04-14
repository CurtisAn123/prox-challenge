/**
 * ImageRenderer.jsx
 * ──────────────────
 * Displays one or more images returned by the IMAGE route.
 *
 * Backend shape (content.images[]):
 *   { base64_data: string, caption: string, page: number, source: string,
 *     width: number, height: number, score: number }
 *
 * Clicking a thumbnail opens a full-screen modal lightbox with zoom/pan:
 *   - Scroll wheel to zoom in/out
 *   - Drag to pan
 *   - Double-click to reset zoom and position
 *   - +/− buttons and a % label for keyboard-friendly zoom control
 *   - Click backdrop or press Escape to close
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

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

// ── ImageRenderer (list of cards) ─────────────────────────────────────────────

export function ImageRenderer({ images }) {
  if (!images?.length) return null

  return (
    <div className="space-y-4">
      {images.map((img, idx) => (
        <ImageCard key={idx} img={img} />
      ))}
    </div>
  )
}

// ── ImageCard ─────────────────────────────────────────────────────────────────

function ImageCard({ img }) {
  const [expanded, setExpanded] = useState(false)
  const src = img.base64_data
    ? `data:image/png;base64,${img.base64_data}`
    : img.file_path  // fallback to server path

  return (
    <div className="space-y-2">
      {/* Thumbnail — click to open modal */}
      <button
        onClick={() => setExpanded(true)}
        className="w-full text-left focus:outline-none focus:ring-2 focus:ring-brand-500 rounded-lg"
        title="Click to expand"
        aria-haspopup="dialog"
      >
        <img
          src={src}
          alt={img.caption || 'Manual diagram'}
          className="w-full max-h-72 rounded-lg border border-slate-200 object-contain bg-white cursor-zoom-in"
          style={{ display: 'block' }}
        />
      </button>

      {/* Page reference */}
      {img.page > 0 && (
        <div className="flex justify-end px-1">
          <span className="text-xs text-slate-400 whitespace-nowrap font-mono">
            p.{img.page}
          </span>
        </div>
      )}

      {/* Lightbox modal */}
      {expanded && (
        <ImageModal
          src={src}
          alt={img.caption || 'Manual diagram'}
          page={img.page}
          onClose={() => setExpanded(false)}
        />
      )}
    </div>
  )
}

// ── Lightbox modal with zoom/pan ──────────────────────────────────────────────

function ImageModal({ src, alt, page, onClose }) {
  const outerRef = useRef(null)
  const {
    scale, setScale, offset, isDragging,
    reset, handleWheel, handleMouseDown, handleMouseMove, handleMouseUp,
  } = useZoomPan()

  // Attach non-passive wheel listener so preventDefault works
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
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="
          bg-white dark:bg-slate-800 rounded-2xl shadow-2xl
          max-w-[92vw] w-[min(960px,92vw)] h-[85vh]
          p-4 flex flex-col gap-3
        "
        onClick={e => e.stopPropagation()}
      >
        {/* Header: zoom controls + close */}
        <div className="flex items-center gap-2 shrink-0">
          {page > 0 && (
            <span className="text-xs text-slate-400 font-mono">p.{page}</span>
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
            aria-label="Close image"
          >
            ✕
          </button>
        </div>

        {/* Image canvas — overflow hidden, zoom/pan via transform */}
        <div
          ref={outerRef}
          className="flex-1 min-h-0 overflow-hidden rounded-lg bg-slate-100 dark:bg-slate-900 flex items-center justify-center"
          style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={reset}
          title="Scroll to zoom · Drag to pan · Double-click to reset"
        >
          <img
            src={src}
            alt={alt}
            draggable={false}
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: 'center center',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out',
              userSelect: 'none',
              willChange: 'transform',
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
            }}
          />
        </div>
      </div>
    </div>,
    document.body,
  )
}
