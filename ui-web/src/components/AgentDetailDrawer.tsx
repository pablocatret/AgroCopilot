import React, { memo } from "react"
import { AnimatePresence, motion } from "motion/react"
import { X, Cpu, Wrench, FileText, Activity, Clock, Coins, Brain, Target } from "lucide-react"
import type { AgentDetailData } from "../types"

const agentLabels: Record<string, string> = {
  organizer: "Organizador",
  legal: "Legal y normativa",
  case_manager: "Gestor del caso",
  stac: "Busqueda satelital",
  rs_analyst: "Analisis satelital",
  document_analyst: "Documentos",
  spreadsheet_analyst: "Tablas",
  vision_ocr: "Vision y OCR",
  writer: "Redactor",
  direct_writer: "Respuesta",
  free: "Investigacion",
}

type Props = {
  agent: string
  runKey: string
  detail?: AgentDetailData
  isOpen: boolean
  onClose: () => void
}

function AgentDetailDrawer({ agent, runKey, detail, isOpen, onClose }: Props) {
  const label = agentLabels[agent] || agent.replace(/_/g, " ")

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="agent-detail-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />
          <motion.aside
            className="agent-detail-drawer"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 360, damping: 36 }}
          >
            <header className="agent-detail-header">
              <div className="agent-detail-header-left">
                <Brain className="agent-detail-header-icon" />
                <div>
                  <h3>{label}</h3>
                  <p className="agent-detail-runkey">{runKey}</p>
                </div>
              </div>
              <button
                type="button"
                className="agent-detail-close"
                onClick={onClose}
                aria-label="Cerrar detalle"
              >
                <X className="h-4 w-4" />
              </button>
            </header>

            <div className="agent-detail-body">
              {detail?.model && (
                <section className="agent-detail-section">
                  <h4>
                    <Cpu className="agent-detail-section-icon" />
                    Modelo
                  </h4>
                  <p className="agent-detail-value">
                    {detail.model}
                    {detail.provider ? <span className="agent-detail-dim"> · {detail.provider}</span> : null}
                  </p>
                </section>
              )}

              {detail?.mission && (
                <section className="agent-detail-section">
                  <h4>
                    <Target className="agent-detail-section-icon" />
                    Mision
                  </h4>
                  <p className="agent-detail-value">{detail.mission}</p>
                </section>
              )}

              <section className="agent-detail-section">
                <h4>
                  <Wrench className="agent-detail-section-icon" />
                  Herramientas
                </h4>
                <p className="agent-detail-value agent-detail-dim">
                  Disponibles: {detail?.toolsAvailable || "no declaradas"}
                </p>
                {detail?.toolsUsed && detail.toolsUsed.length > 0 ? (
                  <p className="agent-detail-value">
                    Activadas: {detail.toolsUsed.join(", ")}
                  </p>
                ) : null}
              </section>

              {detail?.contextSummary && Object.keys(detail.contextSummary).length > 0 && (
                <section className="agent-detail-section">
                  <h4>
                    <FileText className="agent-detail-section-icon" />
                    Contexto recibido
                  </h4>
                  <pre className="agent-detail-pre">
                    {formatContext(detail.contextSummary)}
                  </pre>
                </section>
              )}

              {detail?.outputPreview && (
                <section className="agent-detail-section">
                  <h4>
                    <FileText className="agent-detail-section-icon" />
                    Resultado
                  </h4>
                  <p className="agent-detail-value">{detail.outputPreview}</p>
                </section>
              )}

              {detail?.executionLevel && detail.executionLevel !== "ok" && (
                <section className="agent-detail-section">
                  <h4>
                    <Activity className="agent-detail-section-icon" />
                    Estado
                  </h4>
                  <span className={`agent-detail-badge agent-detail-badge-${detail.executionLevel}`}>
                    {detail.executionLevel}
                  </span>
                </section>
              )}

              {detail?.trace && (
                <section className="agent-detail-section">
                  <h4>
                    <Activity className="agent-detail-section-icon" />
                    Traza
                  </h4>
                  <div className="agent-detail-trace">
                    <span className="agent-detail-trace-item">
                      <Clock className="agent-detail-trace-icon" />
                      {(detail.trace.durationMs / 1000).toFixed(1)}s
                    </span>
                    <span className="agent-detail-trace-item">
                      {detail.trace.tokensInput + detail.trace.tokensOutput} tokens
                    </span>
                    <span className="agent-detail-trace-item">
                      <Coins className="agent-detail-trace-icon" />
                      ${detail.trace.costUsd.toFixed(4)}
                    </span>
                  </div>
                </section>
              )}

              {!detail && (
                <section className="agent-detail-section">
                  <p className="agent-detail-value agent-detail-dim">
                    Detalle no disponible. El agente fue completado antes de la mejora del panel.
                  </p>
                </section>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

function formatContext(ctx: Record<string, any>): string {
  const lines: string[] = []
  for (const [key, value] of Object.entries(ctx)) {
    if (key.startsWith("_")) continue
    if (typeof value === "boolean") {
      lines.push(`${key}: ${value ? "si" : "no"}`)
    } else if (typeof value === "string") {
      lines.push(`${key}: ${value.slice(0, 120)}${value.length > 120 ? "..." : ""}`)
    } else if (Array.isArray(value)) {
      lines.push(`${key}: [${value.length} items]`)
    } else if (value !== null && value !== undefined) {
      lines.push(`${key}: ${typeof value}`)
    }
  }
  return lines.join("\n")
}

export default memo(AgentDetailDrawer)
