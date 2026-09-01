const WORKSPACE_KEY = "agro-copilot-workspace-id"

export function getWorkspaceId(): string {
  if (typeof window === "undefined") return "local"
  const existing = window.localStorage.getItem(WORKSPACE_KEY)?.trim()
  if (existing) return existing
  const generated = `workspace-${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`}`
  window.localStorage.setItem(WORKSPACE_KEY, generated)
  return generated
}
