/**
 * TextRenderer.jsx
 * ─────────────────
 * Renders the assistant's text answer as formatted HTML.
 *
 * We do a lightweight markdown-to-HTML conversion ourselves rather than
 * pulling in a full markdown library.  The manual answers are predictably
 * formatted (bold, lists, numbered steps) so this handles 95%+ of cases.
 *
 * If you want full CommonMark support, swap this out for react-markdown.
 */

/**
 * Very small markdown → HTML transformer.
 * Handles: bold, inline code, ordered lists, unordered lists, headings,
 * blockquotes, paragraph breaks.
 */
function simpleMarkdown(text) {
  if (!text) return ''

  // Escape HTML entities first (security: prevent XSS from server content)
  const esc = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  const lines = text.split('\n')
  const output = []
  let inOl = false
  let inUl = false

  const closeLists = () => {
    if (inOl) { output.push('</ol>'); inOl = false }
    if (inUl) { output.push('</ul>'); inUl = false }
  }

  const inlineFormat = (line) => {
    return esc(line)
      // Bold: **text**
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic: *text* (single asterisk not inside a word)
      .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
      // Inline code: `code`
      .replace(/`([^`]+)`/g, '<code>$1</code>')
  }

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    const trimmed = raw.trim()

    // Heading
    if (/^#{1,3}\s/.test(trimmed)) {
      closeLists()
      const level = trimmed.match(/^(#+)/)[1].length
      const content = trimmed.replace(/^#+\s+/, '')
      output.push(`<h${level}>${inlineFormat(content)}</h${level}>`)
      continue
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      closeLists()
      output.push(`<blockquote>${inlineFormat(trimmed.slice(2))}</blockquote>`)
      continue
    }

    // Ordered list item
    if (/^\d+\.\s/.test(trimmed)) {
      if (inUl) { output.push('</ul>'); inUl = false }
      if (!inOl) { output.push('<ol>'); inOl = true }
      output.push(`<li>${inlineFormat(trimmed.replace(/^\d+\.\s+/, ''))}</li>`)
      continue
    }

    // Unordered list item
    if (/^[-*•]\s/.test(trimmed)) {
      if (inOl) { output.push('</ol>'); inOl = false }
      if (!inUl) { output.push('<ul>'); inUl = true }
      output.push(`<li>${inlineFormat(trimmed.replace(/^[-*•]\s+/, ''))}</li>`)
      continue
    }

    // Empty line → paragraph break
    if (trimmed === '') {
      closeLists()
      output.push('<br>')
      continue
    }

    // Plain paragraph line
    closeLists()
    output.push(`<p>${inlineFormat(trimmed)}</p>`)
  }

  closeLists()
  return output.join('\n')
}

export function TextRenderer({ text }) {
  if (!text) return null

  return (
    <div
      className="prose text-sm text-slate-700 leading-relaxed"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: simpleMarkdown(text) }}
    />
  )
}
