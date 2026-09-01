import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react"
import axios from "axios"
import { toast } from "sonner"

import { getConversation, postChat, uploadAttachments, getConversationMessages } from "../lib/api"
import { subscribeEvents, type SSEDebugSignal } from "../lib/sse"
import type { AgentDetailData, AgentStatus, AttachmentMeta, ChatResponse, ClarificationRequest, DecisionMode, ExecutionLevel, FinalAnswer, LogEntry, ServerEvent } from "../types"

const MAX_LOGS = 30

export type AgentRunStatus = {
  agent: string
  status: AgentStatus
  runId: number
  totalRuns: number
  attempt: number
  attemptLimit: number
  updatedAt: number
  executionLevel?: ExecutionLevel
  detail?: string
}

export type AgentRunView = {
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

export type ChatSubmitPayload = {
  query: string
  files: File[]
  userId?: string
  memoryEnabled: boolean
  caseId?: string
}

export type SessionProfile = {
  userId?: string
  memoryEnabled: boolean
  decisionMode: DecisionMode
}

export type LastUserRequest = {
  query: string
  fileNames: string[]
  memoryEnabled: boolean
  userId?: string
  submittedAt: number
}

export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  timestamp: number
  // User payload
  query?: string
  fileNames?: string[]
  memoryEnabled?: boolean
  userId?: string
  // Assistant payload
  answer?: FinalAnswer
  clarification?: ClarificationRequest
  isRunning?: boolean
  runViews?: AgentRunView[]
  planDependencies?: Record<string, string[]>
  completedRuns?: number
  completionRate?: number
  agentDetails?: Record<string, AgentDetailData>
}

function computeRunViews(
  planSteps: string[],
  planRuns: Record<string, number>,
  execution: any,
  runtimeStatuses?: Record<string, AgentRunStatus>,
): AgentRunView[] {
  const ordered = Array.from(new Set(planSteps))
  const baseAgents = [...ordered]
  if (!baseAgents.includes("organizer")) {
    baseAgents.unshift("organizer")
  }
  
  const views: AgentRunView[] = []
  const seen = new Set<string>()
  
  baseAgents.forEach((agent) => {
    if (!agent || seen.has(agent)) return
    seen.add(agent)
    const total = Math.max(1, planRuns[agent] ?? 1)
    for (let runId = 1; runId <= total; runId += 1) {
      const key = `${agent}#${runId}`
      const execState = execution?.[agent]
      const runtimeState = runtimeStatuses?.[key]
      let status: AgentStatus = "queued"
      let executionLevel: ExecutionLevel | undefined = runtimeState?.executionLevel
      let detail = runtimeState?.detail || ""
      
      if (agent === "organizer" && runId === 1) {
        status = "done"
      }
      
      if (execState) {
        const instances = execState.instances ?? []
        const inst = instances.find((i: any) => i.instance_id === runId) || { level: execState.final_level, message: "" }
        status = "done"
        executionLevel = inst.level
        detail = inst.message || ""
      }
      if (runtimeState?.status) {
        status = runtimeState.status
      }
      
      views.push({
        key,
        agent,
        runId,
        totalRuns: total,
        status,
        attempt: 1,
        attemptLimit: 1,
        executionLevel,
        detail,
      })
    }
  })
  return views
}

function finalizeTerminalRuns(views: AgentRunView[], planSteps: string[], hasAnswer: boolean): AgentRunView[] {
  if (!hasAnswer || !views.length) return views
  const pendingKeys = new Set(
    views.filter((view) => view.status === "queued" || view.status === "running").map((view) => view.key),
  )
  if (!pendingKeys.size) return views

  const terminalAgents = Array.from(new Set([...planSteps].reverse())).filter(Boolean)
  const preferredAgents = ["direct_writer", "writer", ...terminalAgents]
  const target =
    preferredAgents
      .flatMap((agent) => views.filter((view) => view.agent === agent))
      .find((view) => pendingKeys.has(view.key)) ??
    [...views].reverse().find((view) => pendingKeys.has(view.key))

  if (!target) return views

  return views.map((view) =>
    view.key === target.key && view.status !== "error" && view.status !== "skipped"
      ? {
          ...view,
          status: "done",
          detail: view.detail || "Finalizado con la respuesta final",
        }
      : view,
  )
}

const runKey = (agent: string, runId: number) => `${agent}#${runId}`

export function useChatSession() {
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversationCaseId, setConversationCaseId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [runStatuses, setRunStatuses] = useState<Map<string, AgentRunStatus>>(new Map())
  const [planSteps, setPlanSteps] = useState<string[]>([])
  const [planRuns, setPlanRuns] = useState<Record<string, number>>({})
  const [planDependencies, setPlanDependencies] = useState<Record<string, string[]>>({})
  const [answer, setAnswer] = useState<FinalAnswer | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [sseDebug, setSseDebug] = useState<string[]>([])
  const [agentDetails, setAgentDetails] = useState<Map<string, AgentDetailData>>(new Map())
  const [lastRequest, setLastRequest] = useState<LastUserRequest | null>(null)
  const [sessionProfile, setSessionProfile] = useState<SessionProfile>({
    userId: undefined,
    memoryEnabled: false,
    decisionMode: "case",
  })
  const unsubRef = useRef<null | (() => void)>(null)
  const statusSeenRef = useRef<Set<string>>(new Set())
  const logSeenRef = useRef<Set<string>>(new Set())
  const planRunsRef = useRef<Record<string, number>>({})
  const runStatusesRef = useRef<Record<string, AgentRunStatus>>({})
  const activeConversationRef = useRef<string | null>(null)
  const responseReceivedRef = useRef(false)
  const activeAssistantMsgIdRef = useRef<string | null>(null)
  const agentDetailsRef = useRef<Map<string, AgentDetailData>>(new Map())

  const stopStreaming = useCallback(() => {
    if (unsubRef.current) {
      unsubRef.current()
      unsubRef.current = null
    }
    activeConversationRef.current = null
  }, [])

  const reset = useCallback(() => {
    setConversationId(null)
    setConversationCaseId(null)
    setMessages([])
    setRunStatuses(new Map())
    setPlanSteps([])
    setPlanRuns({})
    setPlanDependencies({})
    setAnswer(null)
    setLogs([])
    setSseDebug([])
    setAgentDetails(new Map())
    setLastRequest(null)
    setIsRunning(false)
    setSessionProfile({
      userId: undefined,
      memoryEnabled: false,
      decisionMode: "case",
    })
    statusSeenRef.current = new Set()
    logSeenRef.current = new Set()
    responseReceivedRef.current = false
    activeAssistantMsgIdRef.current = null
    agentDetailsRef.current = new Map()
    stopStreaming()
  }, [stopStreaming])

  const loadConversation = useCallback(async (convId: string) => {
    stopStreaming()
    setConversationId(convId)
    setRunStatuses(new Map())
    setPlanSteps([])
    setPlanRuns({})
    setPlanDependencies({})
    setAnswer(null)
    setLogs([])
    setSseDebug([])
    setAgentDetails(new Map())
    setLastRequest(null)
    setIsRunning(false)
    statusSeenRef.current = new Set()
    logSeenRef.current = new Set()
    responseReceivedRef.current = false
    activeAssistantMsgIdRef.current = null
    agentDetailsRef.current = new Map()
    try {
      const [conversation, backendMessages] = await Promise.all([getConversation(convId), getConversationMessages(convId)])
      setConversationCaseId(conversation.case_id || null)
      const loaded: ChatMessage[] = []
      for (const msg of backendMessages) {
        if (msg.role === "user") {
          loaded.push({
            id: `loaded-user-${msg.id}`,
            role: "user",
            timestamp: new Date(msg.timestamp).getTime(),
            query: msg.query,
            fileNames: msg.file_names || [],
          })
        } else if (msg.role === "assistant") {
          let parsedAnswer: FinalAnswer | undefined
          try {
            parsedAnswer = JSON.parse(msg.answer_json || "{}") as FinalAnswer
          } catch (e) {
            console.warn("Failed to parse answer_json for message", msg.id, e)
            parsedAnswer = {
              executive_summary: msg.answer_summary || "",
              report_md: "",
              message_md: "",
            }
          }
          loaded.push({
            id: `loaded-assistant-${msg.id}`,
            role: "assistant",
            timestamp: new Date(msg.timestamp).getTime(),
            answer: parsedAnswer,
          })
        }
      }
      setMessages(loaded)
      const latestAnswer = [...loaded].reverse().find((item) => item.role === "assistant")?.answer
      setSessionProfile({
        userId: conversation.user_id || latestAnswer?.memory?.user_id || undefined,
        memoryEnabled: latestAnswer?.memory?.enabled ?? false,
        decisionMode: "case",
      })
    } catch (error) {
      toast.error("No se pudo cargar la conversacion")
    }
  }, [stopStreaming])

  const newConversation = useCallback(() => {
    setConversationId(null)
    setConversationCaseId(null)
    setMessages([])
    setRunStatuses(new Map())
    setPlanSteps([])
    setPlanRuns({})
    setPlanDependencies({})
    setAnswer(null)
    setLogs([])
    setSseDebug([])
    setLastRequest(null)
    setIsRunning(false)
    statusSeenRef.current = new Set()
    logSeenRef.current = new Set()
    responseReceivedRef.current = false
    activeAssistantMsgIdRef.current = null
    stopStreaming()
  }, [stopStreaming])

  useEffect(() => {
    planRunsRef.current = planRuns
  }, [planRuns])

  useEffect(() => () => stopStreaming(), [stopStreaming])

  const statusRecord = useMemo<Record<string, AgentRunStatus>>(() => {
    const entries: Record<string, AgentRunStatus> = {}
    runStatuses.forEach((value, key) => {
      entries[key] = value
    })
    return entries
  }, [runStatuses])

  useEffect(() => {
    runStatusesRef.current = statusRecord
  }, [statusRecord])

  const orderedAgents = useMemo(() => {
    const order = new Set<string>()
    for (const step of planSteps) {
      if (step) order.add(step)
    }
    Object.keys(planRuns).forEach((agent) => {
      if (agent) order.add(agent)
    })
    return Array.from(order)
  }, [planRuns, planSteps])

  const runViews = useMemo<AgentRunView[]>(() => {
    const views: AgentRunView[] = []
    const baseAgents = [...orderedAgents]
    if (runStatuses.has(runKey("organizer", 1))) {
      baseAgents.unshift("organizer")
    }
    const seen = new Set<string>()
    baseAgents.forEach((agent) => {
      if (!agent || seen.has(agent)) return
      seen.add(agent)
      const total = Math.max(1, planRuns[agent] ?? 1)
      for (let runId = 1; runId <= total; runId += 1) {
        const key = runKey(agent, runId)
        const meta = statusRecord[key]
        views.push({
          key,
          agent,
          runId,
          totalRuns: total,
          status: meta?.status ?? "queued",
          attempt: meta?.attempt ?? 1,
          attemptLimit: meta?.attemptLimit ?? 1,
          executionLevel: meta?.executionLevel,
          detail: meta?.detail,
        })
      }
    })
    return views
  }, [orderedAgents, planRuns, runStatuses, statusRecord])

  const completedRuns = useMemo(() => {
    return runViews.filter((run) => run.status === "done").length
  }, [runViews])

  const completionRate = useMemo(() => {
    return runViews.length ? Math.round((completedRuns / runViews.length) * 100) : 0
  }, [runViews, completedRuns])

  const handleServerEvent = useCallback((event: ServerEvent) => {
    if (event.type === "plan") {
      const steps = Array.isArray(event.steps) && event.steps.length ? event.steps : Array.isArray(event.agents) ? event.agents : []
      const runs = event.runs ?? {}
      startTransition(() => {
        setPlanDependencies(event.dependencies ?? {})
        statusSeenRef.current = new Set()
        setPlanSteps(steps)
        setPlanRuns(runs)
        setRunStatuses(() => {
          const map = new Map<string, AgentRunStatus>()
          const baseAgents = steps.length ? steps : Object.keys(runs)
          baseAgents.forEach((agent) => {
            if (!agent) return
            const total = Math.max(1, runs[agent] ?? 1)
            for (let runId = 1; runId <= total; runId += 1) {
              map.set(runKey(agent, runId), {
                agent,
                status: "queued",
                runId,
                totalRuns: total,
                attempt: 1,
                attemptLimit: 1,
                updatedAt: Date.now(),
              })
            }
          })
          return map
        })
      })
      return
    }

    if (event.type === "agent_status") {
      const actor = (event as any).actor || event.agent || "organizer"
      const ts = (event as any).ts || (event as any).timestamp || new Date().toISOString()
      const runId = event.run_id ?? 1
      const attempt = event.attempt ?? 1
      const dedupeKey = `${event.type}:${actor}:${runId}:${attempt}:${event.status}:${ts}`
      if (statusSeenRef.current.has(dedupeKey)) return
      statusSeenRef.current.add(dedupeKey)
      const totalRuns = event.total_runs ?? planRunsRef.current[actor] ?? 1
      const limit = event.attempt_limit ?? 1
      startTransition(() => {
        setRunStatuses((prev) => {
          const next = new Map(prev)
          next.set(runKey(actor, runId), {
            agent: actor,
            status: event.status,
            runId,
            totalRuns,
            attempt,
            attemptLimit: limit,
            updatedAt: Date.now(),
            executionLevel: prev.get(runKey(actor, runId))?.executionLevel,
            detail: prev.get(runKey(actor, runId))?.detail,
          })
          return next
        })
        setPlanRuns((prev) => ((prev[actor] ?? 1) === totalRuns ? prev : { ...prev, [actor]: totalRuns }))
        setPlanSteps((prev) => {
          if (prev.includes(actor)) return prev
          const writerAliases = ["writer", "direct_writer"]
          if (writerAliases.includes(actor) && prev.some((a) => writerAliases.includes(a))) return prev
          return [...prev, actor]
        })
      })
      return
    }

    if (event.type === "agent_detail") {
      const key = event.run_key || event.agent
      const data: AgentDetailData = {
        model: event.model,
        provider: event.provider,
        mission: event.mission,
        toolsAvailable: event.tools_available,
        toolsUsed: event.tools_used,
        contextSummary: event.context_summary,
        outputPreview: event.output_preview,
        executionLevel: event.execution_level,
        trace: event.trace ? {
          durationMs: event.trace.duration_ms,
          tokensInput: event.trace.tokens_input,
          tokensOutput: event.trace.tokens_output,
          costUsd: event.trace.cost_usd,
        } : null,
      }
      startTransition(() => {
        setAgentDetails((prev) => {
          const next = new Map(prev)
          next.set(key, data)
          return next
        })
      })
      agentDetailsRef.current.set(key, data)
      return
    }

    if (event.type === "status" && event.stage === "completed") {
      responseReceivedRef.current = true
      setIsRunning(false)
      stopStreaming()
      return
    }

    if (event.type === "log") {
      const ts = event.timestamp || (event as any).ts || new Date().toISOString()
      const actor = event.agent || (event as any).actor
      const key = `${event.type}:${actor || "system"}:${ts}:${event.message}:${event.level || "INFO"}`
      if (logSeenRef.current.has(key)) return
      logSeenRef.current.add(key)
      startTransition(() => {
        setLogs((prev) => {
          const next = [{ message: event.message, level: event.level || "INFO", timestamp: ts, agent: actor ?? undefined }, ...prev]
          return next.slice(0, MAX_LOGS)
        })
      })
      return
    }

    if (event.type === "error") {
      setIsRunning(false)
      const prefix = responseReceivedRef.current ? "Error SSE tras la respuesta" : "Error SSE"
      toast.error(`${prefix}: ${event.error || "desconocido"}`)
      stopStreaming()
    }
  }, [stopStreaming])

  const pushDebug = useCallback((label: string) => {
    const stamp = new Date().toLocaleTimeString()
    startTransition(() => {
      setSseDebug((prev) => {
        const next = [`${stamp} ${label}`, ...prev]
        return next.slice(0, 40)
      })
    })
  }, [])

  const describeSignal = useCallback((signal: SSEDebugSignal) => {
    switch (signal.kind) {
      case "connect":
        return `connect → ${signal.url}`
      case "open":
        return "open"
      case "message":
        return `message ${signal.raw}`
      case "error":
        return "error (reintentando)"
      case "retry":
        return `retry in ${signal.delay}ms`
      case "stopped":
        return "stopped"
      default:
        return "unknown"
    }
  }, [])

  const startStreaming = useCallback((targetConversationId: string) => {
    stopStreaming()
    activeConversationRef.current = targetConversationId
    unsubRef.current = subscribeEvents(
      targetConversationId,
      (raw: ServerEvent) => {
        const actor = (raw as any).actor || (raw as any).agent || "?"
        const status = (raw as any).status ? ` ${raw.type}:${actor}:${(raw as any).status}` : ` ${raw.type}`
        pushDebug(`event${status}`)
        handleServerEvent(raw)
      },
      (signal) => {
        pushDebug(`sse ${describeSignal(signal)}`)
      },
    )
  }, [describeSignal, handleServerEvent, pushDebug, stopStreaming])

  const onSubmit = useCallback(async (payload: ChatSubmitPayload) => {
    const cid = conversationId || (globalThis.crypto?.randomUUID?.() ?? `cid-${Date.now()}-${Math.random().toString(16).slice(2)}`)
    const userMsgId = `msg-user-${Date.now()}`
    const assistantMsgId = `msg-assistant-${Date.now()}`
    
    try {
      setRunStatuses(new Map())
      setPlanSteps([])
      setPlanRuns({})
      setPlanDependencies({})
      setAnswer(null)
      setIsRunning(true)
      responseReceivedRef.current = false
      agentDetailsRef.current = new Map()
      setAgentDetails(new Map())
      
      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        timestamp: Date.now(),
        query: payload.query,
        fileNames: payload.files.map((file) => file.name),
        memoryEnabled: payload.memoryEnabled,
        userId: payload.userId,
      }
      
      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        timestamp: Date.now(),
        isRunning: true,
      }
      
      activeAssistantMsgIdRef.current = assistantMsgId
      setMessages((prev) => [...prev, userMsg, assistantMsg])
      
      if (!conversationId) {
        setConversationId(cid)
      }
      if (payload.caseId) setConversationCaseId(payload.caseId)
      
      setSessionProfile({
        userId: payload.userId,
        memoryEnabled: payload.memoryEnabled,
        decisionMode: "case",
      })
      
      setRunStatuses(new Map([[runKey("organizer", 1), {
        agent: "organizer",
        status: "running",
        runId: 1,
        totalRuns: 1,
        attempt: 1,
        attemptLimit: 1,
        updatedAt: Date.now(),
      }]]))

      let attachments: AttachmentMeta[] = []
      if (payload.files.length) {
        toast.info("Subiendo adjuntos…")
        try {
          attachments = await uploadAttachments(payload.files)
          toast.success(`Adjuntos cargados (${attachments.length})`)
        } catch (error: any) {
          attachments = []
          toast.error(`Error subiendo adjuntos: ${error?.message || error}`)
        }
      }

      startStreaming(cid)

      const requestPayload = {
        query: payload.query,
        conversation_id: cid,
        decision_mode: "case" as const,
        attachment_ids: attachments.map((attachment) => attachment.attachment_id),
        user_id: payload.userId || undefined,
        memory_enabled: payload.memoryEnabled,
        continuity_mode: "auto" as const,
        case_id: payload.caseId,
      }
      
      const data: ChatResponse = await postChat(requestPayload)
      responseReceivedRef.current = true
      
      // Si hay solicitud de clarificación, mostrar opciones en vez de respuesta
      if (data.clarification) {
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id === assistantMsgId) {
              return {
                ...msg,
                clarification: data.clarification,
                isRunning: false,
              }
            }
            return msg
          })
        )
        const serverId = data.conversation_id ?? cid
        if (serverId && serverId !== cid) {
          setConversationId(serverId)
        }
        setIsRunning(false)
        return
      }

      const steps = Array.isArray(data.plan?.steps) ? data.plan.steps : []
      const runs = data.plan?.runs ?? {}
      if (steps.length) {
        setPlanSteps(steps)
        setPlanRuns(runs)
        setPlanDependencies(data.plan?.dependencies ?? {})
      }
      setAnswer(data.answer ?? null)
      
      const finalRunViews = finalizeTerminalRuns(
        computeRunViews(steps, runs, data.answer?.execution, runStatusesRef.current),
        steps,
        Boolean(data.answer),
      )
      const finalCompleted = finalRunViews.filter((r) => r.status === "done").length
      const finalRate = finalRunViews.length ? Math.round((finalCompleted / finalRunViews.length) * 100) : 0

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === assistantMsgId) {
            return {
              ...msg,
              answer: data.answer ?? undefined,
              runViews: finalRunViews,
              planDependencies: data.plan?.dependencies ?? {},
              completedRuns: finalCompleted,
              completionRate: finalRate,
              agentDetails: Object.fromEntries(agentDetailsRef.current),
              isRunning: false,
            }
          }
          return msg
        })
      )

      const serverId = data.conversation_id ?? cid
      if (serverId && serverId !== cid) {
        setConversationId(serverId)
        pushDebug(`conversation reassigned ${serverId}`)
        if (activeConversationRef.current === cid) {
          startStreaming(serverId)
        }
      }
    } catch (error: any) {
      setIsRunning(false)
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === assistantMsgId) {
            return {
              ...msg,
              isRunning: false,
            }
          }
          return msg
        })
      )
      const detail = axios.isAxiosError(error) ? error.response?.data?.detail : undefined
      if (detail?.error === "missing_attachments") {
        const ids = Array.isArray(detail.attachment_ids) ? detail.attachment_ids.join(", ") : "desconocidos"
        toast.error(`Faltan adjuntos requeridos para ejecutar el caso (${ids}). Vuelva a subirlos e inténtelo de nuevo.`)
      } else {
        const message = detail?.message || error?.message || error
        toast.error(`Error al llamar a la API: ${message}`)
      }
      stopStreaming()
    }
  }, [conversationId, startStreaming, stopStreaming, pushDebug])

  return {
    answer,
    conversationId,
    conversationCaseId,
    isRunning,
    logs,
    lastRequest,
    onSubmit,
    runViews,
    planDependencies,
    sessionProfile,
    sseDebug,
    messages,
    reset,
    loadConversation,
    newConversation,
    completedRuns,
    completionRate,
    agentDetails,
  }
}
