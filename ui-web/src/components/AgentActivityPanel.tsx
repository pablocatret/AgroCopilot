import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  AlertTriangle,
  BookOpenText,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileSearch,
  Globe2,
  Image,
  Loader2,
  Map,
  PenLine,
  Table2,
  TriangleAlert,
} from "lucide-react"
import type { AgentDetailData, AgentStatus, ExecutionLevel } from "../types"

type AgentRun = {
  key: string
  agent: string
  runId: number
  totalRuns: number
  status: AgentStatus
  attempt: number
  attemptLimit: number
  executionLevel?: ExecutionLevel
  detail?: string
}

type Props = {
  runs: AgentRun[]
  dependencies?: Record<string, string[]>
  isRunning: boolean
  completedRuns: number
  completionRate: number
  initiallyCollapsed?: boolean
  agentDetails?: Record<string, AgentDetailData>
  onSelectAgent?: (agent: string, runKey: string) => void
}

const agentMeta: Record<string, { label: string; action: string; icon: typeof Brain }> = {
  organizer: { label: "Organizador", action: "decide que especialistas deben intervenir", icon: Brain },
  legal: { label: "Legal y normativa", action: "contrasta requisitos y fuentes oficiales", icon: BookOpenText },
  case_manager: { label: "Gestor del caso", action: "ordena memoria, tareas y bloqueos", icon: CheckCircle2 },
  stac: { label: "Busqueda satelital", action: "localiza escenas e indices disponibles", icon: Map },
  rs_analyst: { label: "Analisis satelital", action: "interpreta vigor, humedad y cambios temporales", icon: Globe2 },
  document_analyst: { label: "Documentos", action: "extrae datos relevantes de adjuntos", icon: FileSearch },
  spreadsheet_analyst: { label: "Tablas", action: "lee hojas de calculo y datos estructurados", icon: Table2 },
  vision_ocr: { label: "Vision y OCR", action: "lee imagenes y texto visible", icon: Image },
  writer: { label: "Redactor", action: "convierte evidencias en respuesta clara", icon: PenLine },
  direct_writer: { label: "Respuesta", action: "prepara una contestacion directa", icon: PenLine },
  free: { label: "Investigacion", action: "busca informacion general en internet", icon: Globe2 },
}

type NodeStatus = "planned" | "running" | "done" | "error" | "idle"

type GraphPosition = { x: number; y: number; layer: number }

const NODE_W = 116
const NODE_H = 88
const LAYER_GAP = 110
const NODE_GAP = 16
const PAD_X = 60
const PAD_Y = 40

type LayoutResult = {
  positions: Record<string, GraphPosition>
  allAgents: string[]
  width: number
  height: number
}

function layoutGraph(
  steps: string[],
  dependencies: Record<string, string[]>,
): LayoutResult {
  const layerOf: Record<string, number> = {}
  const ordered = ["organizer", ...steps.filter((s) => s !== "organizer")]

  for (const agent of ordered) {
    const deps = dependencies[agent] || []
    let maxDepLayer = -1
    for (const dep of deps) {
      if (dep in layerOf && layerOf[dep] > maxDepLayer) {
        maxDepLayer = layerOf[dep]
      }
    }
    if (agent === "organizer") {
      layerOf[agent] = 0
    } else {
      layerOf[agent] = Math.max(maxDepLayer + 1, 1)
    }
  }

  const layerGroups: Record<number, string[]> = {}
  for (const agent of Object.keys(layerOf)) {
    const layer = layerOf[agent]
    if (!layerGroups[layer]) layerGroups[layer] = []
    layerGroups[layer].push(agent)
  }

  const layerKeys = Object.keys(layerGroups).map(Number).sort((a, b) => a - b)
  const numLayers = layerKeys.length
  let maxInLayer = 1
  for (const lk of layerKeys) {
    if (layerGroups[lk].length > maxInLayer) maxInLayer = layerGroups[lk].length
  }

  const dynamicGap = Math.min(NODE_GAP + (numLayers - 2) * 6, 48)
  const width = PAD_X * 2 + numLayers * NODE_W + Math.max(0, numLayers - 1) * LAYER_GAP
  const height = PAD_Y * 2 + maxInLayer * NODE_H + Math.max(0, maxInLayer - 1) * dynamicGap

  const positions: Record<string, GraphPosition> = {}
  const allAgents: string[] = []

  for (const layerIdx of layerKeys) {
    const agents = layerGroups[layerIdx]
    const totalH = agents.length * NODE_H + Math.max(0, agents.length - 1) * dynamicGap
    const startY = (height - totalH) / 2

    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i]
      positions[agent] = {
        x: PAD_X + layerIdx * (NODE_W + LAYER_GAP) + NODE_W / 2,
        y: startY + i * (NODE_H + dynamicGap) + NODE_H / 2,
        layer: layerIdx,
      }
      allAgents.push(agent)
    }
  }

  return { positions, allAgents, width, height }
}

function buildConnections(dependencies: Record<string, string[]>): Array<{ from: string; to: string }> {
  if (!dependencies || Object.keys(dependencies).length === 0) {
    return []
  }
  const conns: Array<{ from: string; to: string }> = []
  const seen = new Set<string>()

  for (const [child, parents] of Object.entries(dependencies)) {
    for (const parent of parents) {
      const key = `${parent}->${child}`
      if (!seen.has(key)) {
        seen.add(key)
        conns.push({ from: parent, to: child })
      }
    }
  }

  for (const child of Object.keys(dependencies)) {
    if (child === "organizer") continue
    const parents = dependencies[child] || []
    if (!parents.includes("organizer")) {
      const key = `organizer->${child}`
      if (!seen.has(key)) {
        seen.add(key)
        conns.push({ from: "organizer", to: child })
      }
    }
  }

  return conns
}

const AgentGraph = React.memo(function AgentGraph({
  runs,
  dependencies,
  agentDetails,
  onSelectAgent,
}: {
  runs: AgentRun[]
  dependencies: Record<string, string[]>
  agentDetails?: Record<string, AgentDetailData>
  onSelectAgent?: (agent: string, runKey: string) => void
}) {
  const steps = useMemo(() => {
    const seen: Record<string, boolean> = {}
    const result: string[] = []
    for (const r of runs) {
      if (!seen[r.agent]) {
        seen[r.agent] = true
        result.push(r.agent)
      }
    }
    return result
  }, [runs])

  const layout = useMemo(() => layoutGraph(steps, dependencies), [steps, dependencies])
  const connections = useMemo(() => buildConnections(dependencies), [dependencies])

  const getNodeStatus = useCallback(
    (nodeId: string): NodeStatus => {
      if (nodeId === "organizer") {
        if (runs.length > 0) {
          return runs.some((r) => r.status === "running" || r.status === "done") ? "done" : "idle"
        }
        return "idle"
      }
      const match = runs.find(
        (r) =>
          r.agent === nodeId ||
          (nodeId === "writer" && r.agent === "direct_writer"),
      )
      if (!match) return "idle"
      if (match.status === "queued") return "planned"
      return match.status as NodeStatus
    },
    [runs],
  )

  const getNodeRunKey = useCallback(
    (nodeId: string): string => {
      const match = runs.find(
        (r) =>
          r.agent === nodeId ||
          (nodeId === "writer" && r.agent === "direct_writer"),
      )
      return match?.key || `${nodeId}#1`
    },
    [runs],
  )

  const positions = layout.positions

  return (
    <div className="agent-graph-shell">
      <div className="agent-graph-grid" />
      <svg
        className="w-full z-10 overflow-visible"
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        style={{ maxHeight: `${Math.min(layout.height + 20, 420)}px` }}
      >
        <defs>
          <filter id="glow-running" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        {connections.map((conn, idx) => {
          const fromPos = positions[conn.from]
          const toPos = positions[conn.to]
          if (!fromPos || !toPos) return null
          const fromStatus = getNodeStatus(conn.from)
          const toStatus = getNodeStatus(conn.to)
          let connStatus: "done" | "running" | "planned" | "idle" = "idle"
          if (fromStatus === "done" && toStatus === "done") {
            connStatus = "done"
          } else if (fromStatus === "running" || toStatus === "running") {
            connStatus = "running"
          } else if (fromStatus === "done" && toStatus === "planned") {
            connStatus = "planned"
          } else if (fromStatus === "planned" || toStatus === "planned") {
            connStatus = "planned"
          }
          const dx = Math.abs(toPos.x - fromPos.x) * 0.55
          const path = `M ${fromPos.x} ${fromPos.y} C ${fromPos.x + dx} ${fromPos.y}, ${toPos.x - dx} ${toPos.y}, ${toPos.x} ${toPos.y}`
          const strokeColor =
            connStatus === "done"
              ? "rgba(16, 185, 129, 0.35)"
              : connStatus === "planned"
                ? "rgba(99, 102, 241, 0.18)"
                : "rgba(255, 255, 255, 0.05)"
          return (
            <g key={`conn-${idx}`}>
              <path
                d={path}
                fill="none"
                stroke={strokeColor}
                strokeWidth={2}
                className={`agent-graph-connection ${
                  connStatus === "done"
                    ? "agent-graph-connection-done"
                    : connStatus === "running"
                      ? "agent-graph-connection-running"
                      : connStatus === "planned"
                        ? "agent-graph-connection-planned"
                        : "agent-graph-connection-idle"
                }`}
              />
              {connStatus === "running" ? (
                <path
                  d={path}
                  fill="none"
                  stroke="rgba(52, 211, 153, 0.75)"
                  strokeWidth={2}
                  strokeDasharray="6, 6"
                  className="agent-graph-connection-running flowing-pulse"
                />
              ) : null}
            </g>
          )
        })}
        {layout.allAgents.map((nodeId) => {
          const pos = positions[nodeId]
          const meta = agentMeta[nodeId]
          if (!meta) return null
          const status = getNodeStatus(nodeId)
          const runData = runs.find(
            (r) =>
              r.agent === nodeId ||
              (nodeId === "writer" && r.agent === "direct_writer"),
          )
          const Icon = meta.icon
          const isRunning = status === "running"
          const isDone = status === "done"
          const nodeClass =
            status === "done"
              ? "agent-node-shell-done"
              : status === "running"
                ? "agent-node-shell-running"
                : status === "error"
                  ? "agent-node-shell-error"
                  : status === "planned"
                    ? "agent-node-shell-planned"
                    : "agent-node-shell-idle"
          const stateLabel =
            status === "done"
              ? "Listo"
              : status === "running"
                ? "Ejecutando"
                : status === "error"
                  ? "Error"
                  : status === "planned"
                    ? "Planificado"
                    : "Disponible"
          const execLevel = runData?.executionLevel
          const attempt = runData?.attempt ?? 1
          const attemptLimit = runData?.attemptLimit ?? 1
          const runKey = getNodeRunKey(nodeId)
          const hasDetail = isDone && agentDetails?.[runKey]
          return (
            <g
              key={nodeId}
              filter={isRunning ? "url(#glow-running)" : ""}
              className={`agent-node-clickable ${isDone ? "agent-node-clickable-done" : ""}`}
              onClick={isDone && hasDetail ? () => onSelectAgent?.(nodeId, runKey) : undefined}
              style={{ cursor: isDone && hasDetail ? "pointer" : "default" }}
            >
              <foreignObject x={pos.x - NODE_W / 2} y={pos.y - NODE_H / 2 + 4} width={NODE_W} height={NODE_H}>
                <div className={`agent-node-shell ${nodeClass}`}>
                  {(execLevel === "insufficient_data" || execLevel === "soft_error") && (
                    <span
                      className={`agent-node-execution-badge ${
                        execLevel === "insufficient_data"
                          ? "agent-node-execution-badge-insufficient"
                          : "agent-node-execution-badge-soft-error"
                      }`}
                    >
                      {execLevel === "insufficient_data" ? (
                        <TriangleAlert className="w-2.5 h-2.5" />
                      ) : (
                        <AlertTriangle className="w-2.5 h-2.5" />
                      )}
                    </span>
                  )}
                  <div className="agent-node-icon">
                    {isRunning ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Icon className="h-5 w-5" />
                    )}
                  </div>
                  <div className="agent-node-copy">
                    <p className="agent-node-title">{meta.label}</p>
                    <p className="agent-node-state">{stateLabel}</p>
                    {attempt > 1 && (
                      <p className="agent-node-retry-label">
                        intento {attempt}/{attemptLimit}
                      </p>
                    )}
                  </div>
                </div>
              </foreignObject>
            </g>
          )
        })}
      </svg>
    </div>
  )
})

function prettyAgent(agent: string) {
  return agentMeta[agent]?.label ?? agent.replace(/_/g, " ")
}

function AgentActivityPanel({
  runs,
  dependencies = {},
  isRunning,
  completedRuns,
  completionRate,
  initiallyCollapsed = false,
  agentDetails,
  onSelectAgent,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(!initiallyCollapsed)
  const wasRunningRef = useRef(isRunning)
  const activeRuns = runs.filter((run) => run.status === "running")
  const queuedRuns = runs.filter((run) => run.status === "queued")
  const doneRuns = runs.filter((run) => run.status === "done")
  const isCompleted = !isRunning && runs.length > 0
  const showCompactSummary = isRunning || !isCompleted || isExpanded
  const headerCopy = isRunning
    ? activeRuns.length > 1
      ? `${activeRuns.length} agentes en paralelo`
      : activeRuns.length === 1
        ? `${prettyAgent(activeRuns[0].agent)} trabajando...`
        : "Planificando ruta de agentes..."
    : runs.length
      ? "Analisis completado"
      : "Ruta pendiente"
  const headlineStats = useMemo(() => {
    if (isCompleted) {
      return [] as Array<{ label: string; value: number }>
    }
    return [
      { label: "Activos", value: activeRuns.length },
      { label: "En cola", value: queuedRuns.length },
      { label: "Listos", value: doneRuns.length },
    ].filter((item) => item.value > 0)
  }, [activeRuns.length, doneRuns.length, isCompleted, queuedRuns.length])
  const progressLabel = runs.length
    ? `${Math.max(completionRate, isRunning ? 0 : 100)}%`
    : "0%"
  const summaryLabel = runs.length
    ? `${doneRuns.length} especialista${doneRuns.length === 1 ? "" : "s"} completado${doneRuns.length === 1 ? "" : "s"}`
    : "Se activaran especialistas solo cuando hagan falta."

  useEffect(() => {
    if (isRunning) {
      setIsExpanded(true)
    } else if (wasRunningRef.current && !isRunning) {
      setIsExpanded(false)
    }
    wasRunningRef.current = isRunning
  }, [isRunning])

  return (
    <section
      data-capture="execution-route"
      className={`agent-activity-message select-none ${isCompleted ? "agent-activity-complete" : ""}`}
    >
      <div
        className={`agent-activity-head ${isCompleted && !isExpanded ? "agent-activity-head-compact" : ""}`}
      >
        <div className="agent-orbit" aria-hidden="true">
          <span className={isRunning ? "animate-ping" : ""} />
          <span />
          <span />
        </div>
        <div className="min-w-0 flex-1">
          <p className="eyebrow">Flujo multiagente</p>
          <h3 className="text-zinc-200">{headerCopy}</h3>
          {showCompactSummary ? (
            <p className="text-zinc-400">{summaryLabel}</p>
          ) : null}
          {headlineStats.length ? (
            <div className="agent-headline-stats">
              {headlineStats.map((item) => (
                <span key={item.label}>
                  <strong>{item.value}</strong>
                  {item.label}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="agent-activity-actions">
          <div
            className="agent-activity-progress"
            aria-label={`Progreso ${completionRate}%`}
          >
            <span>{progressLabel}</span>
          </div>
          <button
            type="button"
            className="agent-activity-toggle"
            onClick={() => setIsExpanded((prev) => !prev)}
            aria-expanded={isExpanded}
            aria-label={isExpanded ? "Ocultar flujo de agentes" : "Ver flujo de agentes"}
          >
            <span>{isExpanded ? "Ocultar flujo" : "Ver flujo"}</span>
            {isExpanded ? (
              <ChevronUp className="h-4 w-4 text-zinc-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-zinc-400" />
            )}
          </button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {isExpanded ? (
          <motion.div
            className="agent-activity-body"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
          >
            <AgentGraph
              runs={runs}
              dependencies={dependencies}
              agentDetails={agentDetails}
              onSelectAgent={onSelectAgent}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  )
}

export default memo(AgentActivityPanel)
