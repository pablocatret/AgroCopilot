import { useDeferredValue, useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AnimatePresence, motion } from "motion/react"
import { Maximize2, Minimize2, ChevronRight, Paperclip } from "lucide-react"
import Header from "./components/Header"
import ChatForm from "./components/ChatForm"
import ConversationSidebar from "./components/ConversationSidebar"
import AnswerView from "./components/AnswerView"
import ClarificationCard from "./components/ClarificationCard"
import ReferencesList from "./components/ReferencesList"
import Gallery from "./components/Gallery"
import LogPanel from "./components/LogPanel"
import AgentActivityPanel from "./components/AgentActivityPanel"
import AgentDetailDrawer from "./components/AgentDetailDrawer"
import CasesPanel from "./components/CasesPanel"
import CaseWorkspace from "./components/CaseWorkspace"
import TraceLogo from "./components/TraceLogo"
import { useChatSession } from "./hooks/useChatSession"
import { productDemoAnswer, productDemoQuery, productDemoRuns } from "./demo/productDemo"
import type { CaseRecord, FinalAnswer } from "./types"
import { getWorkspaceId } from "./lib/workspace"

export type ThemeMode = "dark" | "light"

const welcomePrompts = [
  "En que te puedo ayudar hoy?",
  "Que necesitas resolver hoy?",
  "Que quieres revisar ahora?",
  "Como te ayudo con este caso?",
]

function themeFromUrl(): ThemeMode | null {
  if (typeof window === "undefined") return null
  const requestedTheme = new URLSearchParams(window.location.search).get("theme")
  return requestedTheme === "dark" || requestedTheme === "light" ? requestedTheme : null
}

export default function App() {
  const productDemoEnabled = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("demo") === "product"
  const productDemoTheme = productDemoEnabled ? themeFromUrl() : null
  const [debugOpen, setDebugOpen] = useState(false)
  const [contextRailOpen, setContextRailOpen] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [selectedAgent, setSelectedAgent] = useState<{ agent: string; runKey: string } | null>(null)
  const [workspaceId] = useState(() => getWorkspaceId())
  const [welcomePrompt] = useState(() => welcomePrompts[Math.floor(Math.random() * welcomePrompts.length)])
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "light"
    const requestedTheme = themeFromUrl()
    if (requestedTheme) return requestedTheme
    const storedTheme = window.localStorage.getItem("ui-theme")
    return storedTheme === "dark" ? "dark" : "light"
  })
  const { isRunning, logs, onSubmit, planDependencies, runViews, sessionProfile, sseDebug, messages, reset, loadConversation, newConversation, completedRuns, completionRate, conversationId, conversationCaseId, agentDetails } =
    useChatSession()

  const visibleMessages = useMemo(() => {
    if (productDemoEnabled) {
      return [
        {
          id: "demo-user-1",
          role: "user" as const,
          timestamp: Date.now() - 10000,
          query: productDemoQuery,
          fileNames: ["nota_norte.txt", "parcela_norte.png"],
          memoryEnabled: false,
          userId: "demo-las-lomas",
        },
        {
          id: "demo-assistant-1",
          role: "assistant" as const,
          timestamp: Date.now(),
          answer: productDemoAnswer,
          runViews: productDemoRuns,
          planDependencies: {},
          completedRuns: productDemoRuns.filter((r) => r.status === "done").length,
          completionRate: 100,
          isRunning: false,
        },
      ]
    }
    return messages
  }, [messages, productDemoEnabled])

  const visibleIsRunning = productDemoEnabled ? false : isRunning
  const visibleRunViews = productDemoEnabled ? productDemoRuns : runViews
  const deferredRunViews = useDeferredValue(visibleRunViews)
  const deferredMessages = useDeferredValue(visibleMessages)
  const visibleSessionProfile = productDemoEnabled
    ? { userId: undefined, memoryEnabled: false, decisionMode: "case" as const }
    : sessionProfile

  const latestAssistantMessage = useMemo(() => {
    for (let i = visibleMessages.length - 1; i >= 0; i -= 1) {
      if (visibleMessages[i].role === "assistant") return visibleMessages[i]
    }
    return null
  }, [visibleMessages])

  const latestAnswer = latestAssistantMessage?.answer || null
  const showGallery = useMemo(() => {
    const items = latestAnswer?.stac?.items || []
    const hasThumbs = items.some((it) => (it.assets || []).some((a) => a.thumbnail))
    return hasThumbs || Boolean(latestAnswer?.temporal_comparison?.available)
  }, [latestAnswer])
  const activeCompletedRuns = latestAssistantMessage?.completedRuns ?? completedRuns
  const activeTotalRuns = latestAssistantMessage?.runViews?.length ?? visibleRunViews.length
  const activeWorkspaceId = latestAnswer?.memory?.user_id || visibleSessionProfile.userId || workspaceId
  const activeCaseId = selectedCaseId || latestAnswer?.continuity?.case_id || latestAnswer?.case_id || conversationCaseId || null

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.style.colorScheme = theme
    window.localStorage.setItem("ui-theme", theme)
  }, [theme])

  useEffect(() => {
    if (productDemoEnabled) setTheme(productDemoTheme || "light")
  }, [productDemoEnabled, productDemoTheme])

  return (
    <div className="app-bg min-h-screen overflow-x-hidden text-zinc-100">
      <div className="grain" />
      <div className="terrain terrain-left" />
      <div className="terrain terrain-right" />
      <Header isRunning={visibleIsRunning} completedRuns={activeCompletedRuns} totalRuns={activeTotalRuns} theme={theme} onThemeChange={setTheme} onReset={reset} sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen((p) => !p)} />

      <main
        data-capture="frontend-shell"
        className={`chat-shell ${sidebarOpen ? "chat-shell-sidebar-open" : ""} ${contextRailOpen ? "chat-shell-rail-open" : "chat-shell-rail-closed"} relative z-10 mx-auto grid w-full max-w-[1560px] gap-5 px-3 py-5 sm:gap-6 sm:px-6 lg:px-8`}
      >
        <ConversationSidebar
          open={sidebarOpen}
          activeConversationId={conversationId}
          onSelect={(id) => { loadConversation(id); setSidebarOpen(false) }}
          onNewConversation={() => { newConversation(); setSidebarOpen(false) }}
        />

        <section className="chat-workspace animate-rise">
          <div className="chat-workspace-toolbar">
            <button
              type="button"
              className="rail-toggle"
              onClick={() => setContextRailOpen((prev) => !prev)}
              aria-pressed={contextRailOpen}
              aria-label={contextRailOpen ? "Activar pantalla completa del chat" : "Salir de pantalla completa del chat"}
              title={contextRailOpen ? "Pantalla completa" : "Restaurar contexto"}
            >
              {contextRailOpen ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
            </button>
          </div>
          <div className="chat-thread" data-capture="chat-thread">
            {deferredMessages.length === 0 ? <WelcomeMessage prompt={welcomePrompt} /> : null}
            {deferredMessages.map((msg) => {
              if (msg.role === "user") {
                return (
                  <UserCaseMessage
                    key={msg.id}
                    query={msg.query || ""}
                    fileNames={msg.fileNames || []}
                  />
                )
              }

              const displayRuns = msg.isRunning ? deferredRunViews : msg.runViews || []
              const displayDeps = msg.isRunning ? planDependencies : msg.planDependencies || {}
              const displayCompleted = msg.isRunning ? completedRuns : msg.completedRuns || 0
              const displayRate = msg.isRunning ? completionRate : msg.completionRate || 0

              return (
                <div key={msg.id} className="chat-row assistant-row">
                  <div className="assistant-avatar" aria-hidden="true">
                    <TraceLogo className="trace-logo trace-logo-avatar" />
                  </div>
                  <div className="chat-message-stack w-full space-y-4">
                    {(msg.isRunning || displayRuns.length > 0) ? (
                      <AgentActivityPanel
                        runs={displayRuns}
                        dependencies={displayDeps}
                        isRunning={msg.isRunning || false}
                        completedRuns={displayCompleted}
                        completionRate={displayRate}
                        initiallyCollapsed={!msg.isRunning}
                        agentDetails={msg.isRunning ? Object.fromEntries(agentDetails) : msg.agentDetails}
                        onSelectAgent={(agent, runKey) => setSelectedAgent({ agent, runKey })}
                      />
                    ) : null}
                    {msg.clarification ? (
                      <ClarificationCard
                        clarification={msg.clarification}
                        isProcessing={visibleIsRunning}
                        onSelect={(enrichedQuery) => onSubmit({
                          query: enrichedQuery,
                          files: [],
                          userId: activeWorkspaceId,
                          memoryEnabled: false,
                        })}
                      />
                    ) : (
                      <div className="space-y-3">
                        <AnswerView
                          answer={msg.answer || null}
                          completedRuns={displayCompleted}
                          totalRuns={displayRuns.length}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="chat-composer-wrap" data-capture="case-entry">
            <ChatForm
              onSubmit={onSubmit}
              workspaceId={activeWorkspaceId}
              caseId={activeCaseId}
              initialQuery={productDemoEnabled ? productDemoQuery : undefined}
              demoMode={productDemoEnabled}
              isProcessing={visibleIsRunning}
            />
          </div>
        </section>

        <aside className={`context-rail space-y-5 lg:sticky lg:top-24 lg:h-[calc(100svh-7rem)] lg:overflow-y-auto lg:pr-1 ${contextRailOpen ? "" : "lg:hidden"}`}>
          {!productDemoEnabled ? (
            <>
              <CasesPanel
                workspaceId={activeWorkspaceId}
                selectedCaseId={activeCaseId}
                onSelect={(item: CaseRecord | null) => setSelectedCaseId(item?.case_id || null)}
              />
              {activeCaseId ? <CaseWorkspace caseId={activeCaseId} workspaceId={activeWorkspaceId} /> : null}
            </>
          ) : null}

          {(showGallery || latestAnswer?.references?.length) ? (
            <div data-capture="evidence-traceability" className="space-y-4">
              {showGallery ? (
                <section className="context-rail-card">
                  <Gallery stac={latestAnswer?.stac} comparison={latestAnswer?.temporal_comparison} />
                </section>
              ) : null}
              {latestAnswer?.references?.length ? (
                <section className="context-rail-card">
                  <ReferencesList refs={latestAnswer.references} legal={latestAnswer?.legal} />
                </section>
              ) : null}
            </div>
          ) : null}

          {/* Legacy memory and campaign drawers remain available in source for migration but are not part of the product flow.
          {!productDemoEnabled ? (
            <Dialog.Root open={campaignOpen} onOpenChange={setCampaignOpen}>
              <Dialog.Trigger asChild>
                <section className="context-rail-card">
                  <button type="button" className="rail-collapse-trigger">
                    <span>Observaciones de campo</span>
                    <ChevronRight size={14} />
                  </button>
                </section>
              </Dialog.Trigger>
              <AnimatePresence>
                {campaignOpen ? (
                  <Dialog.Portal forceMount>
                    <Dialog.Overlay asChild>
                      <motion.div className="debug-overlay fixed inset-0 z-50 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} />
                    </Dialog.Overlay>
                    <Dialog.Content asChild>
                      <motion.aside
                        className="debug-drawer fixed right-0 top-0 z-50 flex h-full w-full max-w-xl flex-col"
                        initial={{ x: "100%", opacity: 0.85 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: "100%", opacity: 0.85 }}
                        transition={{ type: "spring", stiffness: 360, damping: 36 }}
                      >
                        <div className="flex items-center justify-between gap-4 border-b border-white/10 px-5 py-4">
                          <div className="min-w-0 flex-1">
                            <p className="eyebrow">Campaña</p>
                            <Dialog.Title className="section-title mt-1 text-2xl text-zinc-50">Seguimiento de campaña</Dialog.Title>
                            <Dialog.Description className="sr-only">Historial y registro de observaciones de campo.</Dialog.Description>
                          </div>
                          <Dialog.Close asChild>
                            <button type="button" className="btn btn-ghost">Cerrar</button>
                          </Dialog.Close>
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto p-5">
                          <CampaignPanel userId={userId} />
                        </div>
                      </motion.aside>
                    </Dialog.Content>
                  </Dialog.Portal>
                ) : null}
              </AnimatePresence>
            </Dialog.Root>
          ) : null}

          {memoryEnabled && !productDemoEnabled ? (
            <>
              <section className="context-rail-card">
                <MemorySelector
                  userId={userId}
                  memoryEnabled={memoryEnabled}
                  activeMemoryId={activeMemoryId}
                  onOpenEditor={() => setMemoryOpen(true)}
                />
              </section>

              <Dialog.Root open={memoryOpen} onOpenChange={setMemoryOpen}>
                <AnimatePresence>
                  {memoryOpen ? (
                    <Dialog.Portal forceMount>
                      <Dialog.Overlay asChild>
                        <motion.div className="debug-overlay fixed inset-0 z-50 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} />
                      </Dialog.Overlay>
                      <Dialog.Content asChild>
                        <motion.aside
                          className="debug-drawer fixed right-0 top-0 z-50 flex h-full w-full max-w-xl flex-col"
                          initial={{ x: "100%", opacity: 0.85 }}
                          animate={{ x: 0, opacity: 1 }}
                          exit={{ x: "100%", opacity: 0.85 }}
                          transition={{ type: "spring", stiffness: 360, damping: 36 }}
                        >
                          <div className="flex items-center justify-between gap-4 border-b border-white/10 px-5 py-4">
                            <div className="min-w-0 flex-1">
                              <p className="eyebrow">Memoria</p>
                              <Dialog.Title className="section-title mt-1 text-2xl text-zinc-50">Memoria del perfil</Dialog.Title>
                              <Dialog.Description className="sr-only">Editor de memoria persistente del agricultor.</Dialog.Description>
                            </div>
                            <Dialog.Close asChild>
                              <button type="button" className="btn btn-ghost">Cerrar</button>
                            </Dialog.Close>
                          </div>
                          <div className="min-h-0 flex-1 overflow-y-auto p-5">
                            <MemoryPanel userId={userId} memoryEnabled={memoryEnabled} memoryUsage={latestAnswer?.memory} />
                          </div>
                        </motion.aside>
                      </Dialog.Content>
                    </Dialog.Portal>
                  ) : null}
                </AnimatePresence>
              </Dialog.Root>
            </>
          ) : null}
          */}

          <Dialog.Root open={debugOpen} onOpenChange={setDebugOpen}>
            <Dialog.Trigger asChild>
              <section className="context-rail-card">
                <button type="button" className="rail-collapse-trigger">
                  <span>Trazas tecnicas</span>
                  <ChevronRight size={14} />
                </button>
              </section>
            </Dialog.Trigger>
            <AnimatePresence>
              {debugOpen ? (
                <Dialog.Portal forceMount>
                  <Dialog.Overlay asChild>
                    <motion.div className="debug-overlay fixed inset-0 z-50 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} />
                  </Dialog.Overlay>
                  <Dialog.Content asChild>
                    <motion.aside
                      className="debug-drawer fixed right-0 top-0 z-50 flex h-full w-full max-w-xl flex-col"
                      initial={{ x: "100%", opacity: 0.85 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ x: "100%", opacity: 0.85 }}
                      transition={{ type: "spring", stiffness: 360, damping: 36 }}
                    >
                      <div className="flex items-center justify-between gap-4 border-b border-white/10 px-5 py-4">
                        <div className="min-w-0 flex-1">
                          <p className="eyebrow">Trazas tecnicas</p>
                          <Dialog.Title className="section-title mt-1 text-2xl text-zinc-50">Trazas de ejecucion</Dialog.Title>
                          <Dialog.Description className="sr-only">Panel tecnico con resumen de ejecucion, logs recientes y eventos SSE del caso actual.</Dialog.Description>
                        </div>
                        <Dialog.Close asChild>
                          <button type="button" className="btn btn-ghost">Cerrar</button>
                        </Dialog.Close>
                      </div>
                      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5">
                        <div className="surface">
                          <p className="eyebrow">Resumen</p>
                          <div className="mt-4 grid grid-cols-2 gap-3">
                            <DebugMetric label="Agentes" value={visibleRunViews.length || 0} />
                            <DebugMetric label="Completados" value={completedRuns} />
                            <DebugMetric label="Avance" value={`${completionRate}%`} />
                            <DebugMetric label="Respuesta" value="Conversacion" />
                          </div>
                        </div>
                        <div className="surface">
                          <LogPanel logs={logs} />
                        </div>
                        <div className="surface">
                          <p className="eyebrow">Eventos SSE</p>
                          {!sseDebug.length ? (
                            <p className="mt-3 text-sm text-zinc-500">Sin eventos tecnicos registrados.</p>
                          ) : (
                            <ul className="mt-3 max-h-72 space-y-1 overflow-y-auto pr-1 text-xs text-zinc-300">
                              {sseDebug.map((line, idx) => (
                                <li key={`${line}-${idx}`} className="mono rounded-lg bg-black/20 px-3 py-2 text-zinc-300">{line}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>
                    </motion.aside>
                  </Dialog.Content>
                </Dialog.Portal>
              ) : null}
            </AnimatePresence>
          </Dialog.Root>
        </aside>
      </main>

      <AgentDetailDrawer
        agent={selectedAgent?.agent || ""}
        runKey={selectedAgent?.runKey || ""}
        detail={selectedAgent ? agentDetails.get(selectedAgent.runKey) : undefined}
        isOpen={!!selectedAgent}
        onClose={() => setSelectedAgent(null)}
      />
    </div>
  )
}

function WelcomeMessage({ prompt }: { prompt: string }) {
  return (
    <div className="chat-row assistant-row">
      <div className="assistant-avatar" aria-hidden="true">
        <TraceLogo className="trace-logo trace-logo-avatar" />
      </div>
      <div className="chat-message-stack">
        <section className="welcome-message">
          <p>{prompt}</p>
        </section>
      </div>
    </div>
  )
}

function UserCaseMessage({
  query,
  fileNames,
}: {
  query: string
  fileNames: string[]
}) {
  return (
    <div className="chat-row user-row">
      <div className="user-message">
        <p>{query}</p>
        {fileNames.length ? (
          <div className="user-file-list">
            {fileNames.map((name) => (
              <span key={name}><Paperclip size={11} className="shrink-0" />{name}</span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function DebugMetric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="debug-metric px-4 py-3">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-zinc-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-zinc-100">{value}</p>
    </div>
  )
}
