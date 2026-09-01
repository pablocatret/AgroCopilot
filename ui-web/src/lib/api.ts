import axios from "axios"
import type { AttachmentMeta, CaseDetail, CaseRecord, CaseStatus, CaseTaskRecord, ConversationMessage, ConversationSummary, DecisionMode, MemoryListItem, MemoryMeta, UserMemoryRecord, WorkspaceContext } from "../types"
import { API_BASE } from "./apiBase"

function resolveChatTimeoutMs(): number {
  const raw = (import.meta.env.VITE_CHAT_TIMEOUT_MS as string | undefined)?.trim()
  if (!raw) return 0
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed < 0) return 0
  return parsed
}

const CHAT_TIMEOUT_MS = resolveChatTimeoutMs()

export type ChatPayload = {
  query: string
  conversation_id?: string
  decision_mode?: DecisionMode
  language?: string
  attachment_ids?: string[]
  user_id?: string
  memory_enabled?: boolean
  continuity_mode?: "auto" | "off" | "explicit"
  case_id?: string
}

export async function postChat(payload: ChatPayload) {
  const url = `${API_BASE}/chat`
  console.debug("[postChat] URL:", url)
  const { data } = await axios.post(url, payload, { timeout: CHAT_TIMEOUT_MS })
  return data
}

export async function uploadAttachments(files: File[]): Promise<AttachmentMeta[]> {
  const url = `${API_BASE}/attachments`
  const formData = new FormData()
  files.forEach((file) => formData.append("files", file))
  const { data } = await axios.post(url, formData, { headers: { "Content-Type": "multipart/form-data" } })
  return data.attachments || []
}

export async function fetchMemory(userId: string): Promise<UserMemoryRecord> {
  const { data } = await axios.get(`${API_BASE}/memory/${encodeURIComponent(userId)}`)
  return data
}

export async function updateMemory(userId: string, sections: UserMemoryRecord["sections"]): Promise<UserMemoryRecord> {
  const { data } = await axios.put(`${API_BASE}/memory/${encodeURIComponent(userId)}`, { sections })
  return data
}

export async function deleteMemory(userId: string): Promise<void> {
  await axios.delete(`${API_BASE}/memory/${encodeURIComponent(userId)}`)
}

export async function listMemories(userId: string): Promise<{ user_id: string; items: MemoryListItem[] }> {
  const { data } = await axios.get(`${API_BASE}/memory/${encodeURIComponent(userId)}/list`)
  return data
}

export async function createMemory(userId: string, name: string): Promise<MemoryMeta> {
  const { data } = await axios.post(`${API_BASE}/memory/${encodeURIComponent(userId)}/create`, { name })
  return data
}

export async function switchMemory(userId: string, memoryId: string): Promise<MemoryMeta> {
  const { data } = await axios.put(`${API_BASE}/memory/${encodeURIComponent(userId)}/current`, { memory_id: memoryId })
  return data
}

export async function renameMemory(userId: string, memoryId: string, name: string): Promise<MemoryMeta> {
  const { data } = await axios.put(`${API_BASE}/memory/${encodeURIComponent(userId)}/${encodeURIComponent(memoryId)}/rename`, { name })
  return data
}

export async function deleteMemoryById(userId: string, memoryId: string): Promise<void> {
  await axios.delete(`${API_BASE}/memory/${encodeURIComponent(userId)}/${encodeURIComponent(memoryId)}`)
}

export async function listConversations(userId?: string): Promise<ConversationSummary[]> {
  const params = userId ? { user_id: userId } : {}
  const { data } = await axios.get(`${API_BASE}/conversations`, { params })
  return data.conversations || []
}

export async function getConversationMessages(conversationId: string): Promise<ConversationMessage[]> {
  const { data } = await axios.get(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/messages`)
  return data.messages || []
}

export async function getConversation(conversationId: string): Promise<ConversationSummary> {
  const { data } = await axios.get(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}`)
  return data
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await axios.delete(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}`)
}

export async function listCases(workspaceId: string, status?: CaseStatus): Promise<CaseRecord[]> {
  const { data } = await axios.get(`${API_BASE}/cases`, { params: { workspace_id: workspaceId, status } })
  return data.items || []
}

export async function fetchWorkspaceContext(workspaceId: string): Promise<WorkspaceContext> {
  const { data } = await axios.get(`${API_BASE}/workspace-context/${encodeURIComponent(workspaceId)}`)
  return data
}

export async function saveWorkspaceContext(workspaceId: string, context: Omit<WorkspaceContext, "workspace_id" | "updated_at">): Promise<WorkspaceContext> {
  const { data } = await axios.put(`${API_BASE}/workspace-context/${encodeURIComponent(workspaceId)}`, context)
  return data
}

export async function createCase(workspaceId: string, title: string, objective = ""): Promise<CaseRecord> {
  const { data } = await axios.post(`${API_BASE}/cases`, { workspace_id: workspaceId, title, objective })
  return data
}

export async function fetchCase(caseId: string, workspaceId: string): Promise<CaseDetail> {
  const { data } = await axios.get(`${API_BASE}/cases/${encodeURIComponent(caseId)}`, { params: { workspace_id: workspaceId } })
  return data
}

export async function updateCaseStatus(caseId: string, workspaceId: string, status: CaseStatus): Promise<CaseRecord> {
  const { data } = await axios.post(`${API_BASE}/cases/${encodeURIComponent(caseId)}/status`, { workspace_id: workspaceId, status })
  return data
}

export async function createCaseAssertion(caseId: string, workspaceId: string, key: string, value: string): Promise<void> {
  await axios.post(`${API_BASE}/cases/${encodeURIComponent(caseId)}/assertions`, { workspace_id: workspaceId, key, value })
}

export async function setAssertionStatus(assertionId: string, workspaceId: string, status: string): Promise<void> {
  await axios.post(`${API_BASE}/assertions/${encodeURIComponent(assertionId)}/status`, { workspace_id: workspaceId, status })
}

export async function correctCaseAssertion(assertionId: string, workspaceId: string, value: string): Promise<void> {
  await axios.post(`${API_BASE}/assertions/${encodeURIComponent(assertionId)}/correct`, { workspace_id: workspaceId, value })
}

export async function createCaseTask(caseId: string, workspaceId: string, title: string): Promise<CaseTaskRecord> {
  const { data } = await axios.post(`${API_BASE}/cases/${encodeURIComponent(caseId)}/tasks`, { workspace_id: workspaceId, title, status: "open" })
  return data
}

export async function setCaseTaskStatus(taskId: string, workspaceId: string, status: CaseTaskRecord["status"]): Promise<CaseTaskRecord> {
  const { data } = await axios.post(`${API_BASE}/case-tasks/${encodeURIComponent(taskId)}/status`, { workspace_id: workspaceId, status })
  return data
}

export async function addCaseObservation(
  caseId: string,
  workspaceId: string,
  payload: { date: string; parcel: string; campaign?: string; note: string; severity: "baja" | "media" | "alta" },
): Promise<void> {
  await axios.post(`${API_BASE}/cases/${encodeURIComponent(caseId)}/observations`, { workspace_id: workspaceId, ...payload })
}
