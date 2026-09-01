import { useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { AnimatePresence, motion } from "motion/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Check, Database, Loader2, Pencil, Plus, Trash2, X } from "lucide-react"
import { toast } from "sonner"

import { createMemory, deleteMemoryById, listMemories, renameMemory, switchMemory } from "../lib/api"
import type { MemoryListItem } from "../types"

type Props = {
  userId: string
  onMemorySelected: (userId: string, memoryId: string | null) => void
  isProcessing?: boolean
}

export default function ComposerMemoryDropdown({ userId, onMemorySelected, isProcessing = false }: Props) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [lookupInput, setLookupInput] = useState("")
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState("")
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState("")

  const lookupUserId = lookupInput.trim()
  const canQuery = Boolean(lookupUserId)
  const memoriesKey = ["memories", lookupUserId]

  const memoriesQuery = useQuery({
    queryKey: memoriesKey,
    queryFn: () => listMemories(lookupUserId!),
    enabled: canQuery && open,
  })

  const switchMutation = useMutation({
    mutationFn: (memoryId: string) => switchMemory(lookupUserId!, memoryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      onMemorySelected(lookupUserId, switchMutation.variables!)
      setOpen(false)
      setCreating(false)
    },
    onError: (error: any) => {
      toast.error(`No se pudo cambiar: ${error?.message || error}`)
    },
  })

  const createMutation = useMutation({
    mutationFn: (name: string) => createMemory(lookupUserId!, name),
    onSuccess: (meta) => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      onMemorySelected(lookupUserId, meta.memory_id)
      setCreating(false)
      setNewName("")
      setOpen(false)
      toast.success(`Memoria "${meta.name}" creada`)
    },
    onError: (error: any) => {
      toast.error(`No se pudo crear: ${error?.message || error}`)
    },
  })

  const renameMutation = useMutation({
    mutationFn: ({ memoryId, name }: { memoryId: string; name: string }) => renameMemory(lookupUserId!, memoryId, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      setEditingId(null)
      setEditName("")
      toast.success("Memoria renombrada")
    },
    onError: (error: any) => {
      toast.error(`No se pudo renombrar: ${error?.message || error}`)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (memoryId: string) => deleteMemoryById(lookupUserId!, memoryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      toast.success("Memoria eliminada")
    },
    onError: (error: any) => {
      toast.error(`No se pudo borrar: ${error?.message || error}`)
    },
  })

  const items = memoriesQuery.data?.items || []
  const selectedName = items.find((m) => m.is_current)?.name || null
  const isSwitching = switchMutation.isPending

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button
          type="button"
          className={`composer-memory-btn ${selectedName ? "is-on" : ""}`}
          title="Seleccionar memoria"
          disabled={isProcessing}
        >
          {isSwitching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Database className="h-4 w-4" />
          )}
          <span className="truncate">{selectedName || "Seleccionar memoria"}</span>
        </button>
      </Dialog.Trigger>

      <AnimatePresence>
        {open ? (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild>
              <motion.div
                className="debug-overlay fixed inset-0 z-50 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
              />
            </Dialog.Overlay>
            <Dialog.Content asChild>
              <motion.div
                className="memory-modal fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-3xl border border-white/10 bg-[#0f0f0f]/95 p-0 shadow-2xl backdrop-blur-xl"
                initial={{ opacity: 0, scale: 0.96, y: 8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 8 }}
                transition={{ type: "spring", stiffness: 400, damping: 32 }}
              >
                <div className="flex items-center justify-between gap-3 border-b border-white/8 px-5 py-4">
                  <div className="min-w-0 flex-1">
                    <Dialog.Title className="text-sm font-extrabold uppercase tracking-[0.18em] text-zinc-200">
                      Memoria persistente
                    </Dialog.Title>
                    <Dialog.Description className="mt-1 text-xs text-zinc-500">
                      Contexto reutilizable entre consultas. Seleccione, cree o gestione memorias.
                    </Dialog.Description>
                  </div>
                  <Dialog.Close asChild>
                    <button type="button" className="rounded-lg p-1.5 text-zinc-500 hover:text-zinc-300">
                      <X size={16} />
                    </button>
                  </Dialog.Close>
                </div>

                <div className="p-5">
                  <div className="relative">
                    <input
                      type="text"
                      value={lookupInput}
                      onChange={(e) => setLookupInput(e.target.value)}
                      placeholder="Ej: finca-norte, olivar-rivera"
                      className="h-10 w-full rounded-xl border border-white/10 bg-black/30 px-3.5 text-sm text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-emerald-400/40"
                      onKeyDown={(e) => {
                        if (e.key === "Escape") setOpen(false)
                      }}
                      autoFocus
                    />
                    {lookupInput && (
                      <button
                        type="button"
                        onClick={() => setLookupInput("")}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-300"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>

                  <div className="mt-3 max-h-72 overflow-y-auto">
                    {!canQuery ? (
                      <p className="py-6 text-center text-xs text-zinc-600">
                        Escriba un perfil o explotacion para buscar memorias.
                      </p>
                    ) : memoriesQuery.isLoading ? (
                      <div className="flex items-center justify-center gap-2 py-6 text-xs text-zinc-500">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Cargando...
                      </div>
                    ) : items.length === 0 ? (
                      <p className="py-6 text-center text-xs text-zinc-600">
                        No hay memorias asociadas. Cree una para empezar.
                      </p>
                    ) : (
                      <div className="space-y-1">
                        {items.map((item) => (
                          <div
                            key={item.memory_id}
                            className={`flex items-center gap-2 rounded-xl px-3 py-3 transition-colors ${
                              item.is_current ? "bg-emerald-500/8" : "hover:bg-white/4"
                            }`}
                          >
                            {editingId === item.memory_id ? (
                              <div className="flex flex-1 items-center gap-1.5">
                                <input
                                  type="text"
                                  value={editName}
                                  onChange={(e) => setEditName(e.target.value)}
                                  className="flex-1 rounded-lg border border-white/10 bg-black/30 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-emerald-400/40"
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter" && editName.trim()) {
                                      renameMutation.mutate({ memoryId: item.memory_id, name: editName.trim() })
                                    }
                                    if (e.key === "Escape") setEditingId(null)
                                  }}
                                  autoFocus
                                />
                                <button
                                  type="button"
                                  onClick={() => editName.trim() && renameMutation.mutate({ memoryId: item.memory_id, name: editName.trim() })}
                                  className="rounded p-1 text-emerald-400 hover:text-emerald-300"
                                >
                                  <Check size={12} />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setEditingId(null)}
                                  className="rounded p-1 text-zinc-500 hover:text-zinc-300"
                                >
                                  <X size={12} />
                                </button>
                              </div>
                            ) : (
                              <>
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (!item.is_current) switchMutation.mutate(item.memory_id)
                                    else {
                                      onMemorySelected(lookupUserId, item.memory_id)
                                      setOpen(false)
                                    }
                                  }}
                                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                                >
                                  {item.is_current && <Check size={13} className="shrink-0 text-emerald-400" />}
                                  <span className="truncate text-sm text-zinc-200">{item.name}</span>
                                  {item.used_sections.length > 0 && (
                                    <span
                                      className="shrink-0 rounded-full bg-emerald-950/50 px-1.5 py-0.5 text-[0.6rem] font-medium text-emerald-300"
                                      title="Secciones con contenido"
                                    >
                                      {item.used_sections.length}
                                    </span>
                                  )}
                                </button>
                                <div className="flex shrink-0 items-center gap-0.5">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setEditingId(item.memory_id)
                                      setEditName(item.name)
                                    }}
                                    className="rounded p-1 text-zinc-600 hover:text-zinc-300"
                                    title="Renombrar"
                                  >
                                    <Pencil size={12} />
                                  </button>
                                  {items.length > 1 && (
                                    <button
                                      type="button"
                                      onClick={() => {
                                        if (confirm(`Eliminar memoria "${item.name}"?`)) {
                                          deleteMutation.mutate(item.memory_id)
                                        }
                                      }}
                                      className="rounded p-1 text-zinc-600 hover:text-rose-400"
                                      title="Eliminar"
                                    >
                                      <Trash2 size={11} />
                                    </button>
                                  )}
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {canQuery && (
                    <div className="mt-3 border-t border-white/6 pt-3">
                      {creating ? (
                        <div className="space-y-2">
                          <label className="block text-[0.65rem] font-medium uppercase tracking-wider text-zinc-500">
                            Nombre de la nueva memoria
                          </label>
                          <div className="flex items-center gap-1.5">
                            <input
                              type="text"
                              value={newName}
                              onChange={(e) => setNewName(e.target.value)}
                              placeholder="Ej: Mi olivar, Campaña 2026"
                              className="flex-1 rounded-lg border border-white/10 bg-black/30 px-2.5 py-2 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-emerald-400/40"
                              onKeyDown={(e) => {
                                if (e.key === "Enter" && newName.trim()) createMutation.mutate(newName.trim())
                                if (e.key === "Escape") { setCreating(false); setNewName("") }
                              }}
                              autoFocus
                            />
                            <button
                              type="button"
                              onClick={() => newName.trim() && createMutation.mutate(newName.trim())}
                              disabled={!newName.trim() || createMutation.isPending}
                              className="btn btn-primary h-9 px-4 text-[0.7rem]"
                            >
                              {createMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
                              Crear
                            </button>
                            <button
                              type="button"
                              onClick={() => { setCreating(false); setNewName("") }}
                              className="btn btn-ghost h-9 px-3 text-[0.7rem]"
                            >
                              Cancelar
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => { setCreating(true); setNewName(lookupUserId) }}
                          className="btn btn-ghost w-full px-4 text-[0.7rem]"
                        >
                          <Plus size={13} /> Nueva memoria
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </AnimatePresence>
    </Dialog.Root>
  )
}
