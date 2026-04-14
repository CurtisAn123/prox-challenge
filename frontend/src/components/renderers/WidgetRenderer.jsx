/**
 * WidgetRenderer.jsx
 * ──────────────────
 * Renders an interactive widget from the structured JSON schema returned
 * by the WIDGET route.
 *
 * Backend shape (content):
 *   {
 *     type: "widget",
 *     component: "WireSpeedConfigurator" | "DutyCycleCalculator" | ...,
 *     summary: string,              // brief text answer
 *     schema: {
 *       title, description, fields, output_fields, pre_populated, notes
 *     }
 *   }
 *
 * This component is a GENERIC renderer — it reads the schema and produces
 * a fully interactive form without knowing the specific widget type.
 * Computed outputs are requested via a follow-up query to the backend so
 * we never need to hard-code the manual's data tables in the frontend.
 */

import { useState } from 'react'
import { sendQuery } from '../../api/client'
import { MermaidRenderer } from './MermaidRenderer'
import { TextRenderer } from './TextRenderer'

export function WidgetRenderer({ content }) {
  const { summary, schema } = content
  if (!schema) return <TextRenderer text={summary} />

  return (
    <div className="space-y-4">
      {/* Brief text summary shown immediately above the widget */}
      {summary && (
        <TextRenderer text={summary} />
      )}

      {/* The interactive widget */}
      <DynamicWidget schema={schema} />
    </div>
  )
}

/**
 * DynamicWidget
 * ─────────────
 * Generic form rendered entirely from the JSON schema.
 * Works for any widget type without type-specific code.
 */
function DynamicWidget({ schema }) {
  // Initialise form values with pre-populated defaults
  const initial = {}
  for (const field of schema.fields ?? []) {
    initial[field.name] = schema.pre_populated?.[field.name]
      ?? field.default
      ?? (field.type === 'select' ? (field.options?.[0] ?? '') : '')
  }

  const [values, setValues] = useState(initial)
  const [result, setResult] = useState(null)
  const [computing, setComputing] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (name, value) => {
    setValues(prev => ({ ...prev, [name]: value }))
    setResult(null)  // clear previous result when inputs change
  }

  /**
   * "Calculate" sends the form values back to the agent as a structured
   * natural-language query. Claude resolves it against the manual data.
   *
   * Example query: "Wire speed settings for MIG welding 1/4 inch mild steel"
   */
  const handleSubmit = async (e) => {
    e.preventDefault()
    setComputing(true)
    setError(null)

    // Build a natural-language query from the form values
    const valueDesc = Object.entries(values)
      .filter(([, v]) => v)
      .map(([k, v]) => {
        const fieldDef = schema.fields.find(f => f.name === k)
        return `${fieldDef?.label ?? k}: ${v}`
      })
      .join(', ')

    const query = `${schema.title}: ${valueDesc}`

    try {
      const response = await sendQuery(query)
      setResult(response)
    } catch (err) {
      setError(err.message)
    } finally {
      setComputing(false)
    }
  }

  return (
    <div className="artifact-card">
      {/* Widget header */}
      <div className="artifact-header">
        <span className="text-base">⚙</span>
        <span>{schema.title}</span>
      </div>

      <div className="artifact-body space-y-4">
        {/* Description */}
        {schema.description && (
          <p className="text-xs text-slate-500 leading-relaxed">{schema.description}</p>
        )}

        {/* Dynamic form */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(schema.fields ?? []).map(field => (
              <FormField
                key={field.name}
                field={field}
                value={values[field.name] ?? ''}
                onChange={v => handleChange(field.name, v)}
              />
            ))}
          </div>

          <button
            type="submit"
            disabled={computing}
            className="
              w-full sm:w-auto px-5 py-2 rounded-lg text-sm font-semibold
              bg-brand-600 text-white
              hover:bg-brand-700 active:bg-brand-700
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors
            "
          >
            {computing ? (
              <span className="flex items-center gap-2">
                <span className="loading-dot" />
                <span className="loading-dot" />
                <span className="loading-dot" />
                Calculating…
              </span>
            ) : 'Calculate'}
          </button>
        </form>

        {/* Error state */}
        {error && (
          <p className="text-xs text-red-500 bg-red-50 rounded-lg p-2">{error}</p>
        )}

        {/* Computed result */}
        {result && (
          <div className="border-t border-slate-100 pt-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Result
            </p>
            {result.type === 'mermaid'
              ? <MermaidRenderer syntax={result.content?.syntax} title={result.content?.title} />
              : <TextRenderer text={
                  result.content?.answer
                    ?? result.content?.summary
                    ?? JSON.stringify(result.content, null, 2)
                } />
            }
          </div>
        )}

        {/* Source note */}
        {schema.notes && (
          <p className="text-xs text-slate-400 italic border-t border-slate-100 pt-2">
            {schema.notes}
          </p>
        )}
      </div>
    </div>
  )
}

/**
 * FormField
 * ─────────
 * Renders a single form field based on its type definition.
 */
function FormField({ field, value, onChange }) {
  const labelClass = 'block text-xs font-medium text-slate-600 mb-1'
  const inputClass = `
    w-full rounded-lg border border-slate-200 bg-white
    px-3 py-2 text-sm text-slate-800
    focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent
    transition
  `

  return (
    <div>
      <label htmlFor={field.name} className={labelClass}>
        {field.label}
        {field.required && <span className="text-red-400 ml-0.5">*</span>}
      </label>

      {field.type === 'select' ? (
        <select
          id={field.name}
          value={value}
          onChange={e => onChange(e.target.value)}
          className={inputClass}
          required={field.required}
        >
          {(field.options ?? []).map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      ) : field.type === 'number' ? (
        <input
          id={field.name}
          type="number"
          value={value}
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          onChange={e => onChange(e.target.value)}
          className={inputClass}
          required={field.required}
          placeholder={`${field.min ?? ''}–${field.max ?? ''}`}
        />
      ) : (
        <input
          id={field.name}
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          className={inputClass}
          required={field.required}
        />
      )}
    </div>
  )
}
