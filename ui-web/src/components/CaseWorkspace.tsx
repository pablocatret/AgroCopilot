import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Check, CircleAlert, FilePenLine, ListChecks, Plus, RotateCcw, X } from "lucide-react"
import { toast } from "sonner"

import { addCaseObservation, createCaseAssertion, createCaseTask, fetchCase, setAssertionStatus, setCaseTaskStatus, updateCaseStatus } from "../lib/api"
import type { CaseStatus } from "../types"

type Props = { caseId: string; workspaceId: string }
type Tab = "now" | "timeline" | "known"

export default function CaseWorkspace({ caseId, workspaceId }: Props) {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<Tab>("now")
  const [factKey, setFactKey] = useState("")
  const [factValue, setFactValue] = useState("")
  const [taskTitle, setTaskTitle] = useState("")
  const [observation, setObservation] = useState({ date: new Date().toISOString().slice(0, 10), parcel: "", note: "" })
  const detailKey = ["case", caseId, workspaceId]
  const detailQuery = useQuery({ queryKey: detailKey, queryFn: () => fetchCase(caseId, workspaceId) })
  const refresh = () => queryClient.invalidateQueries({ queryKey: detailKey })
  const statusMutation = useMutation({ mutationFn: (status: CaseStatus) => updateCaseStatus(caseId, workspaceId, status), onSuccess: refresh, onError: () => toast.error("No se pudo actualizar el caso") })
  const assertionMutation = useMutation({ mutationFn: () => createCaseAssertion(caseId, workspaceId, factKey, factValue), onSuccess: () => { setFactKey(""); setFactValue(""); refresh() }, onError: () => toast.error("No se pudo guardar el dato") })
  const assertionStatusMutation = useMutation({ mutationFn: ({ id, status }: { id: string; status: string }) => setAssertionStatus(id, workspaceId, status), onSuccess: refresh })
  const taskMutation = useMutation({ mutationFn: () => createCaseTask(caseId, workspaceId, taskTitle), onSuccess: () => { setTaskTitle(""); refresh() }, onError: () => toast.error("No se pudo crear la tarea") })
  const taskStatusMutation = useMutation({ mutationFn: ({ id, status }: { id: string; status: "proposed" | "open" | "blocked" | "done" | "cancelled" }) => setCaseTaskStatus(id, workspaceId, status), onSuccess: refresh })
  const observationMutation = useMutation({
    mutationFn: () => addCaseObservation(caseId, workspaceId, { ...observation, severity: "media" }),
    onSuccess: () => { setObservation({ date: new Date().toISOString().slice(0, 10), parcel: "", note: "" }); refresh() },
    onError: () => toast.error("No se pudo guardar la observación"),
  })

  if (detailQuery.isLoading) return <section className="context-rail-card text-sm text-zinc-500">Cargando estado del caso…</section>
  if (detailQuery.isError || !detailQuery.data) return <section className="context-rail-card text-sm text-rose-300">No se pudo cargar este caso.</section>
  const detail = detailQuery.data
  const projection = detail.projection
  return (
    <section className="context-rail-card space-y-4">
      <header>
        <p className="eyebrow">Caso activo</p>
        <h3 className="section-title mt-1 text-xl text-zinc-100">{detail.case.title}</h3>
        <p className="mt-1 text-xs text-zinc-500">Actualizado {detail.case.last_activity_at.slice(0, 10)} · {projection.review_count} por revisar</p>
      </header>
      <div className="grid grid-cols-3 gap-1 rounded-xl bg-black/10 p-1">
        {(["now", "timeline", "known"] as Tab[]).map((item) => <button key={item} type="button" onClick={() => setTab(item)} className={`rounded-lg px-2 py-1.5 text-[0.65rem] ${tab === item ? "bg-emerald-500/15 text-emerald-200" : "text-zinc-500"}`}>{item === "now" ? "Ahora" : item === "timeline" ? "Cambios" : "Conoce"}</button>)}
      </div>
      {tab === "now" ? (
        <div className="space-y-3">
          <p className="text-sm leading-6 text-zinc-300">{projection.summary || "Aún no hay hechos confirmados. Añade la información esencial del caso."}</p>
          {projection.conflicts.length ? <div className="rounded-xl border border-amber-400/25 bg-amber-950/20 p-3 text-xs text-amber-100"><CircleAlert className="mr-1 inline h-3.5 w-3.5" />Hay {projection.conflicts.length} dato(s) contradictorio(s) que requieren revisión.</div> : null}
          <div className="space-y-2">
            <p className="eyebrow">Próximos pasos</p>
            {projection.active_tasks.length ? projection.active_tasks.slice(0, 4).map((task) => <div key={task.task_id} className="rounded-xl bg-black/10 p-3"><div className="flex gap-2"><ListChecks size={14} className="mt-0.5 shrink-0 text-emerald-400" /><div className="min-w-0 flex-1"><p className="text-xs text-zinc-200">{task.title}</p><p className="mt-1 text-[0.68rem] text-zinc-500">{task.status === "proposed" ? "Propuesta pendiente de confirmar." : task.rationale}</p></div>{task.status !== "done" ? <button type="button" title="Marcar hecha" onClick={() => taskStatusMutation.mutate({ id: task.task_id, status: "done" })} className="text-emerald-300"><Check size={15} /></button> : null}</div></div>) : <p className="text-xs text-zinc-500">Sin tareas pendientes.</p>}
            <div className="flex gap-2"><input className="input min-w-0 flex-1" value={taskTitle} onChange={(event) => setTaskTitle(event.target.value)} placeholder="Añadir próximo paso" /><button type="button" className="composer-icon-btn" disabled={!taskTitle.trim()} onClick={() => taskMutation.mutate()}><Plus size={15} /></button></div>
            <div className="border-t border-white/5 pt-3">
              <p className="eyebrow">Observación de campo</p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <input className="input" type="date" value={observation.date} onChange={(event) => setObservation((value) => ({ ...value, date: event.target.value }))} />
                <input className="input" placeholder="Parcela" value={observation.parcel} onChange={(event) => setObservation((value) => ({ ...value, parcel: event.target.value }))} />
              </div>
              <textarea className="input mt-2 min-h-[68px] w-full" placeholder="Qué has observado" value={observation.note} onChange={(event) => setObservation((value) => ({ ...value, note: event.target.value }))} />
              <button type="button" className="btn btn-secondary mt-2 w-full text-xs" disabled={!observation.parcel.trim() || !observation.note.trim() || observationMutation.isPending} onClick={() => observationMutation.mutate()}>Guardar observación</button>
            </div>
            {detail.observations?.length ? <div className="border-t border-white/5 pt-3"><p className="eyebrow">Observaciones recientes</p><div className="mt-2 space-y-2">{detail.observations.slice(0, 3).map((item) => <div key={item.observation_id} className="rounded-xl bg-black/10 p-3 text-xs"><p className="text-zinc-200">{item.note}</p><p className="mt-1 text-zinc-500">{item.date} · {item.parcel}</p></div>)}</div></div> : null}
          </div>
        </div>
      ) : null}
      {tab === "timeline" ? <div className="space-y-2">{detail.events.slice(0, 12).map((event) => <div key={event.event_id} className="border-l border-emerald-400/25 pl-3"><p className="text-xs text-zinc-300">{event.event_type.replace(/_/g, " ")}</p><p className="text-[0.65rem] text-zinc-500">{event.created_at.slice(0, 16).replace("T", " ")} · {event.actor_type}</p></div>)}</div> : null}
      {tab === "known" ? <div className="space-y-3"><div className="space-y-2">{detail.assertions.filter((item) => !["superseded", "retracted"].includes(item.status)).map((item) => <div key={item.assertion_id} className={`rounded-xl p-3 text-xs ${item.status === "proposed" ? "border border-amber-400/20 bg-amber-950/15" : "bg-black/10"}`}><p className="text-zinc-200">{item.display_text || `${item.key}: ${item.value_text}`}</p><p className="mt-1 text-[0.65rem] text-zinc-500">{item.status === "proposed" ? "Inferencia pendiente de confirmar" : `Confirmado · ${item.provenance}`}</p>{item.status === "proposed" ? <div className="mt-2 flex gap-2"><button type="button" className="text-emerald-300" onClick={() => assertionStatusMutation.mutate({ id: item.assertion_id, status: "confirmed" })}>Confirmar</button><button type="button" className="text-rose-300" onClick={() => assertionStatusMutation.mutate({ id: item.assertion_id, status: "retracted" })}>Rechazar</button></div> : null}</div>)}</div><div className="grid grid-cols-2 gap-2"><input className="input" value={factKey} onChange={(event) => setFactKey(event.target.value)} placeholder="Dato" /><input className="input" value={factValue} onChange={(event) => setFactValue(event.target.value)} placeholder="Valor" /></div><button type="button" className="btn btn-secondary w-full text-xs" disabled={!factKey.trim() || !factValue.trim()} onClick={() => assertionMutation.mutate()}><FilePenLine size={14} />Guardar dato confirmado</button></div> : null}
      <div className="flex justify-between border-t border-white/5 pt-3"><button type="button" className="text-xs text-zinc-500" onClick={() => statusMutation.mutate(detail.case.status === "closed" ? "active" : "closed")}><RotateCcw className="mr-1 inline h-3.5 w-3.5" />{detail.case.status === "closed" ? "Reabrir" : "Cerrar caso"}</button><button type="button" className="text-xs text-zinc-600" onClick={() => statusMutation.mutate("archived")}><X className="mr-1 inline h-3.5 w-3.5" />Archivar</button></div>
    </section>
  )
}
