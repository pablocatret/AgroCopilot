import { useState } from "react"
import { ChevronDown } from "lucide-react"
import type { CostSummary } from "../types"

const money = (value?: number | null) => `$${(value ?? 0).toFixed((value ?? 0) < 0.01 ? 4 : 3)}`
const tokens = (value?: number | null) => {
  const n = value ?? 0
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return `${n}`
}

export default function CostPanel({ summary }: { summary?: CostSummary | null }) {
  const [open, setOpen] = useState(false)
  if (!summary || summary.event_count === 0) return null

  const agents = Object.entries(summary.by_agent || {}).sort((a, b) => b[1].cost_usd - a[1].cost_usd)
  const models = Object.entries(summary.by_model || {}).sort((a, b) => b[1].cost_usd - a[1].cost_usd)

  return (
    <section className={`rounded-3xl border p-4 ${summary.warning ? "border-amber-300/25 bg-amber-950/20" : "border-cyan-300/15 bg-cyan-950/10"}`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="eyebrow">Costes y modelos</p>
          <h4 className="section-title mt-1 text-2xl text-zinc-50">
            {money(summary.total_cost_usd)} {summary.estimated ? "estimado" : "registrado"}
          </h4>
          <p className="mt-1 text-sm text-zinc-400">
            {tokens(summary.total_tokens)} tokens - {summary.web_calls} llamadas web - modelo principal {summary.top_model || "sin dato"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {summary.warning ? (
            <span className="rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[0.65rem] uppercase tracking-[0.22em] text-amber-100">
              supera {money(summary.warning_threshold_usd)}
            </span>
          ) : null}
          <button className="btn btn-secondary h-9 text-xs" onClick={() => setOpen((value) => !value)}>
            Detalle <ChevronDown size={15} className={open ? "rotate-180 transition" : "transition"} />
          </button>
        </div>
      </div>

      {open ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
            <p className="text-xs uppercase tracking-[0.26em] text-zinc-500">Por agente</p>
            <div className="mt-3 space-y-2">
              {agents.slice(0, 8).map(([agent, item]) => (
                <div key={agent} className="flex items-center justify-between gap-3 rounded-xl bg-white/[0.04] px-3 py-2 text-sm">
                  <span className="text-zinc-200">{agent}</span>
                  <span className="text-zinc-400">{money(item.cost_usd)} - {tokens(item.total_tokens)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
            <p className="text-xs uppercase tracking-[0.26em] text-zinc-500">Por modelo</p>
            <div className="mt-3 space-y-2">
              {models.slice(0, 8).map(([model, item]) => (
                <div key={model} className="flex items-center justify-between gap-3 rounded-xl bg-white/[0.04] px-3 py-2 text-sm">
                  <span className="truncate text-zinc-200">{model}</span>
                  <span className="shrink-0 text-zinc-400">{money(item.cost_usd)} - {tokens(item.total_tokens)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}
