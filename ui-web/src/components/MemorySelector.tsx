import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Check, ChevronDown, Edit3, Loader2, Plus, Trash2, X } from "lucide-react"
import { toast } from "sonner"

import { createMemory, deleteMemoryById, listMemories, renameMemory, switchMemory } from "../lib/api"
import type { MemoryListItem } from "../types"

type Props = {
  userId?: string
  memoryEnabled: boolean
  activeMemoryId?: string | null
  onOpenEditor: () => void
}

export default function MemorySelector({ userId, memoryEnabled, activeMemoryId, onOpenEditor }: Props) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState("")
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState("")
  const dropdownRef = useRef<HTMLDivElement>(null)

  const canLoad = Boolean(memoryEnabled && userId)
  const memoriesKey = ["memories", userId]

  const memoriesQuery = useQuery({
    queryKey: memoriesKey,
    queryFn: () => listMemories(userId!),
    enabled: canLoad,
  })

  useEffect(() => {
    if (!canLoad) return
    memoriesQuery.refetch()
  }, [canLoad, activeMemoryId])

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false)
        setCreating(false)
        setEditingId(null)
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside)
      return () => document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [open])

  const switchMutation = useMutation({
    mutationFn: (memoryId: string) => switchMemory(userId!, memoryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      queryClient.invalidateQueries({ queryKey: ["memory", userId] })
      setOpen(false)
    },
    onError: (error: any) => {
      toast.error(`No se pudo cambiar: ${error?.message || error}`)
    },
  })

  const createMutation = useMutation({
    mutationFn: (name: string) => createMemory(userId!, name),
    onSuccess: (meta) => {
      queryClient.setQueryData(memoriesKey, (prev: any) => {
        if (!prev) return prev
        return {
          ...prev,
          items: [
            ...prev.items,
            { memory_id: meta.memory_id, name: meta.name, is_current: true, used_sections: [] },
          ].map((item: MemoryListItem) => ({ ...item, is_current: item.memory_id === meta.memory_id })),
        }
      })
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      queryClient.invalidateQueries({ queryKey: ["memory", userId] })
      setCreating(false)
      setNewName("")
      toast.success(`Memoria "${meta.name}" creada`)
    },
    onError: (error: any) => {
      toast.error(`No se pudo crear: ${error?.message || error}`)
    },
  })

  const renameMutation = useMutation({
    mutationFn: ({ memoryId, name }: { memoryId: string; name: string }) => renameMemory(userId!, memoryId, name),
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
    mutationFn: (memoryId: string) => deleteMemoryById(userId!, memoryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoriesKey })
      queryClient.invalidateQueries({ queryKey: ["memory", userId] })
      toast.success("Memoria eliminada")
    },
    onError: (error: any) => {
      toast.error(`No se pudo borrar: ${error?.message || error}`)
    },
  })

  if (!memoryEnabled || !userId) return null

  const items = memoriesQuery.data?.items || []
  const current = items.find((m) => m.is_current)

  return (
    <div ref={dropdownRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="rail-collapse-trigger w-full"
      >
        <span className="truncate">{current?.name || "Memoria"}</span>
        <ChevronDown size={14} className={`shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="memory-dropdown">
          <div className="memory-dropdown-header">
            <span className="text-[0.64rem] font-extrabold uppercase tracking-[0.18em] text-zinc-400">Memorias</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-md p-1 text-zinc-500 hover:text-zinc-300"
            >
              <X size={14} />
            </button>
          </div>

          <div className="memory-dropdown-list">
            {memoriesQuery.isLoading ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-400">
                <Loader2 className="h-3 w-3 animate-spin" /> Cargando...
              </div>
            ) : items.length === 0 ? (
              <p className="px-3 py-2 text-xs text-zinc-500">Sin memorias. Cree una nueva.</p>
            ) : (
              items.map((item) => (
                <div
                  key={item.memory_id}
                  className={`memory-dropdown-item ${item.is_current ? "is-current" : ""}`}
                >
                  {editingId === item.memory_id ? (
                    <div className="flex items-center gap-1">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="flex-1 rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-emerald-400/40"
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
                          setOpen(false)
                          onOpenEditor()
                        }}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                      >
                        {item.is_current && <Check size={12} className="shrink-0 text-emerald-400" />}
                        <span className="truncate text-xs text-zinc-200">{item.name}</span>
                        {item.used_sections.length > 0 && (
                          <span className="shrink-0 rounded-full bg-emerald-950/40 px-1.5 py-0.5 text-[0.6rem] text-emerald-300">
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
                          className="rounded p-1 text-zinc-500 hover:text-zinc-300"
                          title="Renombrar"
                        >
                          <Edit3 size={11} />
                        </button>
                        {items.length > 1 && (
                          <button
                            type="button"
                            onClick={() => {
                              if (confirm(`Eliminar memoria "${item.name}"?`)) {
                                deleteMutation.mutate(item.memory_id)
                              }
                            }}
                            className="rounded p-1 text-zinc-500 hover:text-rose-400"
                            title="Eliminar"
                          >
                            <Trash2 size={11} />
                          </button>
                        )}
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="memory-dropdown-footer">
            {creating ? (
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Nombre de la memoria"
                  className="flex-1 rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-emerald-400/40"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newName.trim()) createMutation.mutate(newName.trim())
                    if (e.key === "Escape") setCreating(false)
                  }}
                  autoFocus
                />
                <button
                  type="button"
                  onClick={() => newName.trim() && createMutation.mutate(newName.trim())}
                  className="rounded p-1 text-emerald-400 hover:text-emerald-300"
                >
                  <Check size={14} />
                </button>
                <button
                  type="button"
                  onClick={() => setCreating(false)}
                  className="rounded p-1 text-zinc-500 hover:text-zinc-300"
                >
                  <X size={14} />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setCreating(true)}
                className="flex w-full items-center gap-2 text-xs text-zinc-400 hover:text-zinc-200"
              >
                <Plus size={12} /> Nueva memoria
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
