import { PanelLeft, Plus } from "lucide-react"
import type { ThemeMode } from "../App"
import TraceLogo from "./TraceLogo"

type Props = {
  isRunning: boolean
  completedRuns: number
  totalRuns: number
  theme: ThemeMode
  onThemeChange: (theme: ThemeMode) => void
  onReset: () => void
  sidebarOpen: boolean
  onToggleSidebar: () => void
}

export default function Header({ isRunning, completedRuns, totalRuns, theme, onThemeChange, onReset, sidebarOpen, onToggleSidebar }: Props) {
  return (
    <header className="app-header sticky top-0 z-40 border-b border-white/10 bg-[rgba(7,17,13,0.78)] shadow-[0_10px_34px_rgba(0,0,0,.18)] backdrop-blur-xl">
      <div className="mx-auto flex min-h-16 w-full max-w-[1480px] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="rail-toggle"
            aria-pressed={sidebarOpen}
            aria-label={sidebarOpen ? "Cerrar panel de conversaciones" : "Abrir panel de conversaciones"}
            title={sidebarOpen ? "Cerrar conversaciones" : "Conversaciones"}
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <span className="brand-mark grid h-10 w-10 shrink-0 place-items-center rounded-[1.15rem] border border-emerald-200/25 bg-[radial-gradient(circle_at_35%_20%,rgba(255,255,255,.28),transparent_28%),rgba(189,230,178,.12)] text-sm font-extrabold text-emerald-50 shadow-[inset_0_1px_0_rgba(255,255,255,.16)]">
            <TraceLogo className="trace-logo trace-logo-mark" />
          </span>
          <div className="min-w-0">
            <p className="display truncate text-xl text-zinc-50 sm:text-2xl">AgroCopilot</p>
            <p className="hidden text-xs text-zinc-500 sm:block">Casos, evidencias y respuestas agricolas</p>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <button
            type="button"
            onClick={onReset}
            className="nav-new-case-button"
          >
            <Plus size={14} />
            <span>Nuevo caso</span>
          </button>

          <div className="theme-toggle hidden sm:inline-flex" role="group" aria-label="Selector de tema">
            <button
              type="button"
              className={theme === "light" ? "theme-toggle-active" : ""}
              onClick={() => onThemeChange("light")}
              aria-pressed={theme === "light"}
            >
              Claro
            </button>
            <button
              type="button"
              className={theme === "dark" ? "theme-toggle-active" : ""}
              onClick={() => onThemeChange("dark")}
              aria-pressed={theme === "dark"}
            >
              Oscuro
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
