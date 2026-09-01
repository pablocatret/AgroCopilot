import { motion } from "motion/react"
import type { ImageInsight, ImageInsights, RemoteSensingChange, TemporalComparison, TemporalSceneSummary } from "../types"

const fmtMetric = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "sin dato"
  return value.toFixed(3)
}

const fmtPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return null
  return `${Math.round(value * 100)}%`
}

const fmtDate = (value?: string | null) => {
  if (!value) return "sin fecha"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleDateString("es-ES", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  })
}

const qualityTone = (label?: string | null) => {
  if (label === "alta") return "rs-quality rs-quality-high"
  if (label === "media") return "rs-quality rs-quality-medium"
  if (label === "baja") return "rs-quality rs-quality-low"
  return "rs-quality"
}

const signalOrder: Record<string, number> = { NDVI: 0, NDWI: 1, NDMI: 2, S1_VV: 10, S1_VH: 11, S1_VH_VV_RATIO: 12 }

const isRadarMetric = (metric?: string | null) => Boolean(metric?.toUpperCase().startsWith("S1_"))
const isLandcoverMetric = (metric?: string | null) => metric === "ESA_WORLDCOVER"

const metricLabel = (metric: string) => {
  if (metric === "S1_VH_VV_RATIO") return "Ratio VH/VV"
  if (metric === "S1_VV") return "Radar VV"
  if (metric === "S1_VH") return "Radar VH"
  if (metric === "ESA_WORLDCOVER") return "WorldCover"
  return metric
}

const fmtClassPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "sin dato"
  return `${value.toFixed(1)}%`
}

const firstSentence = (value?: string | null) => {
  if (!value) return null
  const [sentence] = value.split(/(?<=[.!?])\s+/)
  return sentence || value
}

const signalCaution = (metric: string, limitations?: string[]) => {
  const explicit = limitations?.find((item) => item.toUpperCase().includes(metric))
  if (explicit) return firstSentence(explicit) || explicit
  if (metric === "NDWI") {
    return "Mide la presencia relativa de agua o humedad superficial visible. Sirve para detectar laminas de agua, zonas encharcadas o contrastes de humedad, pero no confirma por si solo que el riego haya sido correcto."
  }
  if (metric === "NDMI") {
    return "Resume humedad relativa de vegetacion y canopia usando el SWIR. Es util para reforzar hipotesis de estres o perdida de humedad, aunque debe leerse con fenologia, cobertura y contraste de campo."
  }
  if (metric === "NDVI") {
    return "Es el indicador mas directo de vigor vegetal y uniformidad del cultivo. Ayuda a localizar caidas de actividad fotosintetica o zonas menos desarrolladas, pero conviene interpretarlo junto con fecha, variedad y calidad de escena."
  }
  if (isRadarMetric(metric)) {
    return "Senal radar auxiliar util cuando se quiere reforzar lectura sobre humedad superficial, estructura o rugosidad del terreno. Su interpretacion depende mucho de la geometria SAR, el laboreo y el estado del suelo."
  }
  return "Lectura satelital de apoyo: aporta contexto espacial, pero gana valor cuando se combina con calidad de escena, observaciones de campo y referencias temporales."
}

type SignalSummary = {
  metric: string
  mean?: number | null
  validPixels?: number
  confidence?: number
  delta?: number | null
  severity?: string
  caution: string
  auxiliary: boolean
}

const buildSignalSummaries = (insights: ImageInsight[], changes: RemoteSensingChange[]) => {
  const changesByMetric = new Map(changes.map((change) => [(change.metric || "").toUpperCase(), change]))
  return insights
    .filter((insight) => insight.stats?.index_name && !insight.stats.class_stats?.length)
    .reduce<SignalSummary[]>((acc, insight) => {
      const metric = (insight.stats?.index_name || "").toUpperCase()
      if (!metric || acc.some((item) => item.metric === metric)) return acc
      const change = changesByMetric.get(metric)
      acc.push({
        metric,
        mean: insight.stats?.mean,
        validPixels: insight.stats?.valid_pixels,
        confidence: change?.confidence ?? insight.confidence,
        delta: change?.delta_mean,
        severity: change?.severity,
        caution: signalCaution(metric, [...(change?.limitations ?? []), ...(insight.limitations ?? [])]),
        auxiliary: isRadarMetric(metric),
      })
      return acc
    }, [])
    .sort((a, b) => (signalOrder[a.metric] ?? 50) - (signalOrder[b.metric] ?? 50))
}

function SceneCard({
  scene,
  label,
  metric,
  compact = false,
}: {
  scene: TemporalSceneSummary
  label: string
  metric: string
  compact?: boolean
}) {
  return (
    <article className={`rs-scene-card ${compact ? "rs-scene-card-compact" : ""}`}>
      <div className="rs-scene-media-wrap">
        {scene.preview_href ? (
          <img src={scene.preview_href} alt={scene.item_id} loading="lazy" className="rs-scene-media" />
        ) : (
          <div className="rs-scene-fallback">sin preview</div>
        )}
      </div>
      <div className="rs-scene-body">
        <div className="rs-scene-topline">
          <p className="rs-scene-label">{label}</p>
          {scene.quality?.label ? <span className={qualityTone(scene.quality.label)}>{scene.quality.label}</span> : null}
        </div>
        <p className="rs-scene-id">{scene.item_id}</p>
        <div className="rs-scene-meta">
          <span>{fmtDate(scene.datetime)}</span>
          {scene.product_label ? <span>{scene.product_label}</span> : null}
        </div>
        {scene.stats ? (
          <p className="rs-scene-stats">
            Media {scene.stats.index_name || metric}: {fmtMetric(scene.stats.mean)} · pixeles {scene.stats.valid_pixels ?? 0}
          </p>
        ) : null}
        {scene.summary ? <p className="rs-scene-summary">{scene.summary}</p> : null}
      </div>
    </article>
  )
}

export default function RemoteSensingPanel({
  remoteSensing,
  comparison,
}: {
  remoteSensing?: ImageInsights | null
  comparison?: TemporalComparison | null
}) {
  const changes = remoteSensing?.temporal_changes ?? []
  const focusAreas = remoteSensing?.focus_areas ?? []
  const insights = remoteSensing?.insights ?? []
  const hasContent = Boolean(comparison?.available || changes.length || focusAreas.length || insights.length)
  if (!hasContent) return null

  const current = comparison?.current
  const previous = comparison?.previous
  const primaryChange = changes[0]
  const coverageInsight = insights.find((insight) => insight.stats?.class_stats?.length)
  const coverageStats = coverageInsight?.stats?.class_stats ?? []
  const signalInsights = buildSignalSummaries(insights, changes)
  const metric = comparison?.metric ?? primaryChange?.metric ?? current?.stats?.index_name ?? signalInsights[0]?.metric ?? "indice"
  const metricDisplay = metricLabel(metric.toUpperCase())
  const confidence = comparison?.confidence ?? primaryChange?.confidence
  const delta = comparison?.delta_mean ?? primaryChange?.delta_mean
  const metricNames = Array.from(new Set([
    ...signalInsights.map((item) => item.metric),
    ...changes.map((change) => (change.metric || "").toUpperCase()).filter(Boolean),
  ]))
  const hasRadarSignal = signalInsights.some((item) => item.auxiliary)
  const hasLandcover = coverageStats.length > 0 || metricNames.some(isLandcoverMetric)
  const chipNames = Array.from(new Set([
    ...(metricNames.length ? metricNames : [metric.toUpperCase()]),
    ...(hasLandcover ? ["ESA_WORLDCOVER"] : []),
  ].filter(Boolean)))
  const changePreview = comparison?.change_preview_href ?? primaryChange?.preview_href
  const insightLimitations = insights.flatMap((insight) => insight.limitations ?? [])
  const uniqueLimitations = Array.from(new Set([
    ...(comparison?.limitations ?? []),
    ...(primaryChange?.limitations ?? []),
    ...insightLimitations,
  ].filter(Boolean))).slice(0, 3)
  const keyChanges = comparison?.key_changes?.slice(0, 3) ?? []
  const severity = comparison?.severity || primaryChange?.severity || "sin clasificar"
  const severityTone = severity === "alta" || severity === "media" || severity === "baja" ? severity : "media"
  const changeLabel = metricNames.length > 1
    ? "Lectura multisenal"
    : hasRadarSignal
      ? "Senal radar auxiliar"
      : hasLandcover
        ? "Contexto de cobertura"
        : primaryChange?.reliable
          ? "Cambio satelital medido"
          : "Evidencia satelital disponible"
  const showOperationalStrip = Boolean(primaryChange || comparison?.available)

  return (
    <motion.section
      data-capture="remote-sensing-panel"
      className="rs-panel"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24, ease: [0.2, 0.8, 0.2, 1] }}
    >
      <header className="rs-header">
        <div className="rs-header-copy">
          <p className="eyebrow">Teledeteccion</p>
          <h4 className="section-title rs-title">{changeLabel}</h4>
          <p className="rs-overview">
            {remoteSensing?.overview || comparison?.rationale || "Lectura de escenas STAC asociadas al caso."}
          </p>
        </div>
        <div className="rs-chip-row" aria-label="Productos disponibles">
          {chipNames.map((name) => (
            <span className={`rs-chip ${isRadarMetric(name) ? "rs-chip-radar" : ""}`} key={name}>
              {metricLabel(name)}
            </span>
          ))}
          {comparison?.available ? <span className="rs-chip">temporal</span> : null}
          {confidence !== undefined && confidence !== null ? <span className="rs-chip">confianza {fmtPercent(confidence)}</span> : null}
        </div>
      </header>

      <div className="rs-workspace">
        <div className="rs-visual-column">
          {current ? <SceneCard scene={current} label={comparison?.available ? "Escena actual" : "Escena disponible"} metric={metric} /> : null}

          {(previous || changePreview) ? (
            <div className="rs-secondary-grid">
              {previous ? <SceneCard scene={previous} label="Referencia" metric={metric} compact /> : null}
              {changePreview ? (
                <article className="rs-diff-card">
                  <img src={changePreview} alt="Mapa diferencial satelital" loading="lazy" className="rs-diff-media" />
                  <div className="rs-diff-body">
                    <p className="rs-scene-label">Diferencial</p>
                    <p className="rs-diff-caption">{metricDisplay} actual menos referencia.</p>
                  </div>
                </article>
              ) : null}
            </div>
          ) : null}
        </div>

        <aside className="rs-analysis-rail">
          {signalInsights.length ? (
            <section className="rs-signal-panel">
              <div className="rs-section-head">
                <p className="rs-card-eyebrow">Senales calculadas</p>
                <span>{signalInsights.length} productos</span>
              </div>
              <div className="rs-signal-grid">
                {signalInsights.map((signal) => (
                  <article className={`rs-signal-item ${signal.auxiliary ? "rs-signal-auxiliary" : ""}`} key={signal.metric}>
                    <div className="rs-signal-topline">
                      <span className="rs-signal-name">{metricLabel(signal.metric)}</span>
                      {signal.confidence !== undefined ? <span className="rs-signal-confidence">{fmtPercent(signal.confidence)}</span> : null}
                    </div>
                    <div className="rs-signal-values">
                      <span>media {fmtMetric(signal.mean)}</span>
                      <span>delta {fmtMetric(signal.delta)}</span>
                      {signal.validPixels !== undefined ? <span>{signal.validPixels} px</span> : null}
                    </div>
                    <p className="rs-signal-caution">{signal.caution}</p>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          {coverageStats.length ? (
            <section className="rs-coverage-panel">
              <p className="rs-card-eyebrow">Cobertura</p>
              <div className="rs-class-list">
                {coverageStats.slice(0, 3).map((item) => (
                  <div key={`${item.code}-${item.label}`} className="rs-class-item">
                    <span className="rs-class-label">{item.label}</span>
                    <span className="rs-class-percent">{fmtClassPercent(item.percent)}</span>
                  </div>
                ))}
              </div>
              {coverageInsight?.summary ? <p className="rs-context-copy">{coverageInsight.summary}</p> : null}
            </section>
          ) : null}
        </aside>
      </div>

      {showOperationalStrip ? (
        <section className="rs-decision-strip">
          <div className="rs-decision-metric">
            <p className="rs-metric-label">{isRadarMetric(metric) ? "Delta radar" : "Delta medio"}</p>
            <p className={`rs-metric-value ${delta !== undefined && delta !== null && delta < 0 ? "rs-metric-warm" : "rs-metric-cool"}`}>
              {fmtMetric(delta)}
            </p>
          </div>
          <div className="rs-decision-copy">
            <div className="rs-card-head">
              <p className="rs-card-eyebrow">Lectura operativa</p>
              <span className={`rs-severity-chip rs-severity-${severityTone}`}>{severity}</span>
            </div>
            <p>{primaryChange?.detail || comparison?.rationale || "Comparacion temporal preparada con escenas compatibles."}</p>
          </div>
          {comparison?.rationale ? (
            <div className="rs-decision-frame">
              <p className="rs-metric-label">Marco</p>
              <p>{comparison.rationale}</p>
            </div>
          ) : null}
        </section>
      ) : null}

      {(keyChanges.length || focusAreas.length || uniqueLimitations.length) ? (
        <section className="rs-context-grid">
          {keyChanges.length ? (
            <article className="rs-context-block">
              <p className="rs-card-eyebrow">Hallazgos</p>
              <ul className="rs-list">
                {keyChanges.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          ) : null}

          {focusAreas.length ? (
            <article className="rs-context-block">
              <p className="rs-card-eyebrow">Focos a revisar</p>
              <div className="rs-focus-list">
                {focusAreas.slice(0, 3).map((focus) => (
                  <div key={`${focus.title}-${focus.parcel || "general"}`} className="rs-focus-item">
                    <div className="rs-focus-topline">
                      <span className="rs-focus-title">{focus.title}</span>
                      <span className={`rs-priority-chip rs-priority-${focus.priority}`}>{focus.priority}</span>
                    </div>
                    <p className="rs-focus-detail">{focus.detail}</p>
                    {focus.parcel ? <p className="rs-focus-parcel">{focus.parcel}</p> : null}
                  </div>
                ))}
              </div>
            </article>
          ) : null}

          {uniqueLimitations.length ? (
            <article className="rs-context-block rs-caution-card">
              <p className="rs-card-eyebrow">Cautelas</p>
              <ul className="rs-list">
                {uniqueLimitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          ) : null}
        </section>
      ) : null}
    </motion.section>
  )
}
