import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Archive, CheckCircle2, CirclePlus, FolderOpen, Loader2 } from "lucide-react"
import { toast } from "sonner"

import { createCase, listCases } from "../lib/api"
import type { CaseRecord } from "../types"

type Props = {
  workspaceId?: string
  selectedCaseId?: string | null
  onSelect: (item: CaseRecord | null) => void
}

export default function CasesPanel({ workspaceId, selectedCaseId, onSelect }: Props) {
  const queryClient = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [title, setTitle] = useState("")
  const casesKey = ["cases", workspaceId]
  const casesQuery = useQuery({
    queryKey: casesKey,
    queryFn: () => listCases(workspaceId!),
    enabled: Boolean(workspaceId),
  })
  const createMutation = useMutation({
    mutationFn: () => createCase(workspaceId!, title.trim()),
    onSuccess: (item) => {
      queryClient.invalidateQueries({ queryKey: casesKey })
      onSelect(item)
      setTitle("")
      setCreating(false)
      toast.success("Caso creado")
    },
    onError: (error: any) => toast.error(`No se pudo crear el caso: ${error?.message || error}`),
  })

  if (!workspaceId) return null
  const items = casesQuery.data || []
  return (
    <section className="context-rail-card space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="eyebrow">Casos</p>
          <p className="text-xs text-zinc-500">Expedientes de {workspaceId}</p>
        </div>
        <button type="button" className="composer-icon-btn" title="Crear caso" onClick={() => setCreating((value) => !value)}>
          <CirclePlus size={15} />
        </button>
      </div>
      {creating ? (
        <div className="space-y-2">
          <input className="input" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Título del caso" autoFocus />
          <button type="button" className="btn btn-primary w-full text-xs" disabled={!title.trim() || createMutation.isPending} onClick={() => createMutation.mutate()}>
            {createMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CirclePlus size={14} />} Crear caso
          </button>
        </div>
      ) : null}
      {casesQuery.isLoading ? <p className="text-xs text-zinc-500"><Loader2 className="mr-1 inline h-3 w-3 animate-spin" />Cargando casos…</p> : null}
      {!casesQuery.isLoading && items.length === 0 ? <p className="text-xs text-zinc-500">Crea un caso para conservar decisiones, evidencias y próximos pasos.</p> : null}
      <div className="space-y-1.5">
        {items.slice(0, 8).map((item) => {
          const active = item.case_id === selectedCaseId
          const Icon = item.status === "closed" || item.status === "archived" ? Archive : active ? CheckCircle2 : FolderOpen
          return (
            <button key={item.case_id} type="button" onClick={() => onSelect(active ? null : item)} className={`w-full rounded-xl px-3 py-2 text-left transition ${active ? "bg-emerald-500/12 ring-1 ring-emerald-400/25" : "hover:bg-white/5"}`}>
              <span className="flex items-center gap-2 text-xs font-medium text-zinc-200"><Icon size={13} className="text-emerald-400" />{item.title}</span>
              <span className="mt-1 block truncate text-[0.68rem] text-zinc-500">{item.summary || item.objective || "Sin resumen todavía"}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
