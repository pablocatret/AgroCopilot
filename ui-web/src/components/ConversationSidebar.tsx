import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, MessageSquarePlus, Search, Trash2 } from "lucide-react"
import { toast } from "sonner"

import { deleteConversation, listConversations } from "../lib/api"
import type { ConversationSummary } from "../types"

type Props = {
  open: boolean
  activeConversationId?: string | null
  onSelect: (conversationId: string) => void
  onNewConversation: () => void
}

function groupByDate(items: ConversationSummary[]): { label: string; items: ConversationSummary[] }[] {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const weekAgo = new Date(today)
  weekAgo.setDate(weekAgo.getDate() - 7)

  const groups: Record<string, ConversationSummary[]> = {
    "Hoy": [],
    "Ayer": [],
    "Ultimos 7 dias": [],
    "Anterior": [],
  }

  for (const item of items) {
    const d = new Date(item.updated_at)
    if (d >= today) groups["Hoy"].push(item)
    else if (d >= yesterday) groups["Ayer"].push(item)
    else if (d >= weekAgo) groups["Ultimos 7 dias"].push(item)
    else groups["Anterior"].push(item)
  }

  return Object.entries(groups)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }))
}

export default function ConversationSidebar({ open, activeConversationId, onSelect, onNewConversation }: Props) {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState("")

  const conversationsQuery = useQuery({
    queryKey: ["conversations"],
    queryFn: () => listConversations(),
    enabled: open,
  })

  useEffect(() => {
    if (open) conversationsQuery.refetch()
  }, [open])

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteConversation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] })
      toast.success("Conversacion eliminada")
    },
    onError: (error: any) => {
      toast.error(`No se pudo eliminar: ${error?.message || error}`)
    },
  })

  const items = conversationsQuery.data || []
  const filtered = useMemo(() => {
    if (!search.trim()) return items
    const q = search.toLowerCase()
    return items.filter((c) => c.title.toLowerCase().includes(q))
  }, [items, search])

  const grouped = useMemo(() => groupByDate(filtered), [filtered])

  return (
    <aside className={`conversation-sidebar ${open ? "is-open" : ""}`}>
      {open ? (
        <>
          <div className="conversation-sidebar-header">
            <button
              type="button"
              onClick={onNewConversation}
              className="btn btn-ghost flex w-full items-center justify-center gap-2 px-3 py-2"
            >
              <MessageSquarePlus size={14} />
              Nueva consulta
            </button>
          </div>

          <div className="conversation-sidebar-search">
            <Search size={13} className="conversation-sidebar-search-icon" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Buscar..."
              className="conversation-sidebar-search-input"
            />
          </div>

          <div className="conversation-sidebar-list">
            {conversationsQuery.isLoading ? (
              <div className="flex items-center justify-center gap-2 py-8 text-[.75rem] font-medium" style={{ color: "rgba(var(--text,107,114,128),.45)" }}>
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Cargando...
              </div>
            ) : grouped.length === 0 ? (
              <p className="px-4 py-8 text-center text-[.75rem] font-medium" style={{ color: "rgba(var(--text,107,114,128),.35)" }}>
                {search ? "Sin resultados." : "Sin conversaciones."}
              </p>
            ) : (
              grouped.map((group) => (
                <div key={group.label}>
                  <p className="conversation-sidebar-group-label">{group.label}</p>
                  {group.items.map((conv) => (
                    <div
                      key={conv.conversation_id}
                      className={`conversation-sidebar-item ${conv.conversation_id === activeConversationId ? "is-active" : ""}`}
                    >
                      <button
                        type="button"
                        onClick={() => onSelect(conv.conversation_id)}
                        className="conversation-sidebar-item-btn"
                      >
                        <span className="conversation-sidebar-item-title">{conv.title || "Sin titulo"}</span>
                        {conv.message_count > 0 && (
                          <span className="conversation-sidebar-item-count">{conv.message_count}</span>
                        )}
                        {conv.case_id ? <span className="conversation-sidebar-item-count" title="Vinculada a un seguimiento">↗</span> : null}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (confirm("Eliminar esta conversacion?")) {
                            deleteMutation.mutate(conv.conversation_id)
                          }
                        }}
                        className="conversation-sidebar-item-delete"
                        title="Eliminar"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        </>
      ) : null}
    </aside>
  )
}
