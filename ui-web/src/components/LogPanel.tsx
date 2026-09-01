import { Activity } from "lucide-react"
import type { LogEntry } from "../types"

const levelTone: Record<string, string> = {
  INFO: "bg-zinc-900/60 text-zinc-200",
  WARNING: "bg-amber-500/10 text-amber-100",
  ERROR: "bg-rose-500/10 text-rose-100",
  DEBUG: "bg-sky-500/10 text-sky-100",
}

export default function LogPanel({ logs }: { logs: LogEntry[] }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-semibold text-zinc-50">Actividad</h3>
        </div>
        <span className="mono text-[0.6rem] uppercase tracking-[0.4em] text-zinc-500">SSE</span>
      </div>
      {!logs.length ? (
        <div className="text-xs text-zinc-500">Los eventos del sistema aparecerán aquí cuando los agentes comiencen a razonar.</div>
      ) : (
        <ul className="space-y-2 font-mono">
          {logs.map((log, idx) => (
            <li
              key={`${log.timestamp}-${idx}`}
              className={`rounded-md px-3 py-2 text-xs transition ${levelTone[log.level] ?? levelTone.INFO}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold capitalize text-zinc-100">{log.agent ? log.agent.replace(/_/g, " ") : "sistema"}</span>
                <span className="mono text-[0.6rem] uppercase tracking-[0.3em] text-zinc-400">{log.level}</span>
              </div>
              <p className="mt-1 text-zinc-200">{log.message}</p>
              <span className="mono text-[0.6rem] text-zinc-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
