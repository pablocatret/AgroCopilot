const DEFAULT_DEV_API_BASE = "http://localhost:8000"

export function resolveApiBase(): string {
  const envBase = (import.meta.env.VITE_API_BASE as string | undefined)?.trim()
  if (envBase) {
    return envBase.replace(/\/$/, "")
  }
  if (import.meta.env.DEV) {
    return DEFAULT_DEV_API_BASE
  }
  return window.location.origin
}

export const API_BASE = resolveApiBase()
