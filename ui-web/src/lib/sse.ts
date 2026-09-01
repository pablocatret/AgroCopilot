import { API_BASE } from "./apiBase"

export type SSEDebugSignal =
  | { kind: "connect"; url: string }
  | { kind: "open" }
  | { kind: "message"; raw: string }
  | { kind: "error"; error?: Event }
  | { kind: "retry"; delay: number }
  | { kind: "stopped" }

export function subscribeEvents(conversationId: string, onEvent: (ev: any) => void, onDebug?: (info: SSEDebugSignal) => void) {
  const base = API_BASE.replace(/\/$/, "")
  let es: EventSource | null = null
  let stopped = false
  let retry = 1000

  const connect = () => {
    if (stopped) return
    const url = `${base}/events/${encodeURIComponent(conversationId)}`
    es = new EventSource(url)
    onDebug?.({ kind: "connect", url })
    es.onmessage = (e) => {
      onDebug?.({ kind: "message", raw: e.data })
      try {
        onEvent(JSON.parse(e.data))
      } catch (err) {
        console.warn("[SSE] JSON parse error", err)
      }
    }
    es.onopen = () => {
      retry = 1000
      onDebug?.({ kind: "open" })
    }
    es.onerror = (evt) => {
      if (stopped) return
      onDebug?.({ kind: "error", error: evt })
      es?.close()
      const delay = retry
      retry = Math.min(retry * 2, 10000)
      onDebug?.({ kind: "retry", delay })
      setTimeout(connect, delay)
    }
  }

  connect()

  return () => {
    stopped = true
    es?.close()
    onDebug?.({ kind: "stopped" })
  }
}
