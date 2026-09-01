import type { LegalFindings } from "../types"

const statusCopy: Record<string, string> = {
  cumple: "verificado",
  no_cumple: "pendiente",
  insuficiente: "a contrastar",
}

const statusClass: Record<string, string> = {
  cumple: "legal-status-inline-ok",
  no_cumple: "legal-status-inline-missing",
  insuficiente: "legal-status-inline-review",
}

export default function LegalPanel({ legal }: { legal?: LegalFindings | null }) {
  if (!legal || (!legal.answer && !legal.checklist?.length)) return null

  const limitations = legal.limitations ?? []
  const checklist = legal.checklist ?? []
  const jurisdiction = legal.jurisdiction || checklist.find((item) => item.jurisdiction)?.jurisdiction
  const sourceStatus = legal.source_status || checklist.find((item) => item.source_status)?.source_status
  const updatedAt = legal.updated_at || checklist.find((item) => item.updated_at)?.updated_at
  const verified = checklist.filter((item) => item.status === "cumple").length
  const pending = checklist.filter((item) => item.status === "no_cumple").length
  const review = checklist.filter((item) => item.status === "insuficiente").length

  const metaParts = [jurisdiction, sourceStatus, updatedAt].filter(Boolean)

  return (
    <section data-capture="legal-panel" className="legal-panel">
      {legal.answer ? <p className="legal-lead">{legal.answer}</p> : null}

      {metaParts.length ? (
        <p className="legal-meta-inline">{metaParts.join(" · ")}</p>
      ) : null}

      {checklist.length ? (
        <p className="legal-score-inline">
          {verified > 0 ? `${verified} verificado${verified === 1 ? "" : "s"}` : null}
          {pending > 0 ? `${verified > 0 ? " · " : ""}${pending} pendiente${pending === 1 ? "" : "s"}` : null}
          {review > 0 ? `${verified > 0 || pending > 0 ? " · " : ""}${review} a contrastar` : null}
        </p>
      ) : null}

      {checklist.length ? (
        <ul className="legal-checklist-compact">
          {checklist.slice(0, 5).map((item) => (
            <li key={`${item.requirement}-${item.article || ""}`}>
              <span className={`legal-status-inline ${statusClass[item.status] || ""}`}>
                {statusCopy[item.status] || item.status}
              </span>
              <span className="legal-requirement-text">{item.requirement}</span>
              {item.article ? <span className="legal-article">{item.article}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}

      {limitations.length ? (
        <div className="legal-cautions-compact">
          <p className="legal-cautions-title">Cautelas</p>
          <ul>
            {limitations.slice(0, 3).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  )
}
