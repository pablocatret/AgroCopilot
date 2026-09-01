import { ExternalLink } from "lucide-react"
import { useMemo } from "react"
import type { AgentRef, LegalFindings } from "../types"

const sourceLabel: Record<string, string> = {
  legal: "Legal",
  stac: "Satelital",
  document: "Documento",
  vision: "Vision",
  satellite: "Satelital",
  web: "Web",
  memory: "Memoria",
  general: "General",
}

const prettySource = (source?: string) => {
  if (!source) return "referencia"
  return sourceLabel[source] || source.replace(/_/g, " ")
}

const normalizeSnippet = (snippet?: string) => {
  if (!snippet) return ""
  const compact = snippet.replace(/\s+/g, " ").trim()
  if (compact.length <= 180) return compact
  return `${compact.slice(0, 180)}...`
}

type RefGroup = {
  type: string
  label: string
  refs: AgentRef[]
}

export default function ReferencesList({ refs, legal }: { refs: AgentRef[]; legal?: LegalFindings | null }) {
  if (!refs?.length && !legal?.dossier) return null

  const groups = useMemo(() => {
    const map = new Map<string, AgentRef[]>()
    for (const ref of refs) {
      const source = ref.source || "general"
      if (!map.has(source)) map.set(source, [])
      map.get(source)!.push(ref)
    }

    const dossier = legal?.dossier
    if (dossier) {
      const legalGroup = map.get("legal") || []
      for (const ref of dossier.authoritative_references || []) {
        if (!legalGroup.some((r) => r.url === ref.url)) {
          legalGroup.push({ ref_id: `auth-${ref.url}`, title: ref.title, source: "legal", url: ref.url, snippet: ref.snippet })
        }
      }
      for (const ref of dossier.supporting_references || []) {
        if (!legalGroup.some((r) => r.url === ref.url)) {
          legalGroup.push({ ref_id: `sup-${ref.url}`, title: ref.title, source: "web", url: ref.url, snippet: ref.snippet })
        }
      }
      if (legalGroup.length) map.set("legal", legalGroup)
    }

    const order = ["legal", "document", "vision", "stac", "satellite", "web", "memory", "general"]
    const result: RefGroup[] = []
    for (const type of order) {
      const items = map.get(type)
      if (items?.length) {
        result.push({ type, label: prettySource(type), refs: items })
        map.delete(type)
      }
    }
    for (const [type, items] of map) {
      result.push({ type, label: prettySource(type), refs: items })
    }
    return result
  }, [refs, legal])

  return (
    <section className="references-panel">
      <header className="references-header">
        <div className="min-w-0">
          <p className="eyebrow">Trazabilidad</p>
          <h3 className="section-title text-2xl text-zinc-50">Fuentes y evidencias</h3>
        </div>
      </header>
      <div className="references-groups">
        {groups.map((group) => (
          <div key={group.type} className="references-group">
            <div className="references-group-header">
              <span className={`rail-badge ref-source-badge ref-source-badge-${group.type}`}>
                {group.label}
              </span>
              <span className="rail-badge rail-badge-soft">{group.refs.length}</span>
            </div>
            <div className="references-grid">
              {group.refs.map((ref) => {
                const isHttp = ref.url && /^https?:\/\//i.test(ref.url)
                const snippet = normalizeSnippet(ref.snippet)
                return (
                  <article key={ref.ref_id || `${ref.title}-${ref.url || ""}`} className="reference-card">
                    <div className="reference-card-head">
                      <p>{ref.title}</p>
                      {isHttp && (
                        <a href={ref.url} target="_blank" rel="noreferrer" aria-label={`Abrir ${ref.title}`}>
                          <ExternalLink size={14} />
                        </a>
                      )}
                    </div>
                    {snippet && <p className="reference-snippet-text">{snippet}</p>}
                  </article>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
