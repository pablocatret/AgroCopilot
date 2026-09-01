import { useEffect, useRef, useState } from "react"
import { Loader2, Paperclip, Send, X } from "lucide-react"

type Props = {
  onSubmit: (payload: {
    query: string
    files: File[]
    userId?: string
    memoryEnabled: boolean
    caseId?: string
  }) => Promise<void> | void
  workspaceId: string
  caseId?: string | null
  initialQuery?: string
  demoMode?: boolean
  isProcessing?: boolean
}

const maxFiles = 6
const maxFileSizeMb = 10

export default function ChatForm({ onSubmit, workspaceId, caseId, initialQuery = "", demoMode = false, isProcessing = false }: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const queryInputRef = useRef<HTMLTextAreaElement | null>(null)
  const [q, setQ] = useState(initialQuery)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [fileWarnings, setFileWarnings] = useState<string[]>([])

  useEffect(() => {
    const textarea = queryInputRef.current
    if (!textarea) return
    textarea.style.height = "auto"
    textarea.style.height = `${Math.min(textarea.scrollHeight, 128)}px`
  }, [q])

  const handleFiles = (incoming: FileList | null) => {
    const picked = Array.from(incoming || [])
    const warnings: string[] = []
    const accepted: File[] = []
    for (const file of picked) {
      if (file.size > maxFileSizeMb * 1024 * 1024) {
        warnings.push(`${file.name}: supera ${maxFileSizeMb} MB`)
        continue
      }
      if (accepted.length >= maxFiles) {
        warnings.push(`${file.name}: excede el maximo de ${maxFiles} archivos`)
        continue
      }
      accepted.push(file)
    }
    setFiles(accepted)
    setFileWarnings(warnings)
  }

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== name))
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  async function handle() {
    const trimmedQuery = q.trim()
    if (!trimmedQuery || isProcessing) return
    const outgoingFiles = files
    const outgoingUserId = (demoMode ? "demo-las-lomas" : workspaceId) || undefined
    try {
      setIsSubmitting(true)
      setQ("")
      setFiles([])
      setFileWarnings([])
      await onSubmit({
        query: trimmedQuery,
        files: outgoingFiles,
        userId: outgoingUserId,
        memoryEnabled: false,
        caseId: caseId || undefined,
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const isLoading = isProcessing || isSubmitting

  return (
    <div className="chat-composer">
      <textarea
        ref={queryInputRef}
        rows={1}
        className="chat-composer-input"
        placeholder={demoMode ? "Caso demo OL-17 Norte" : "Describe la parcela, la duda, los adjuntos o la decisión que necesitas revisar..."}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        disabled={isProcessing}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault()
            handle()
          }
        }}
      />

      {files.length > 0 ? (
        <div className="chat-file-strip chat-file-strip-inline">
          {files.map((file) => (
            <button
              key={`${file.name}-${file.size}`}
              type="button"
              onClick={() => removeFile(file.name)}
              className="chat-file-pill"
              title="Quitar archivo"
            >
              <span>{file.name}</span>
              <small>{formatSize(file.size)}</small>
              <X className="h-3 w-3" />
            </button>
          ))}
        </div>
      ) : null}

      <div className="chat-composer-toolbar">
        <div className="chat-composer-tools">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
            className="sr-file-input"
            disabled={isProcessing}
          />
          <button
            type="button"
            className="composer-icon-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Anadir adjuntos"
            aria-label="Anadir adjuntos"
            disabled={isProcessing}
          >
            <Paperclip className="h-4 w-4" />
          </button>
        </div>

        <div className="chat-composer-actions">
          <button
            onClick={() => handle()}
            disabled={isLoading}
            aria-disabled={isLoading}
            className={`chat-respond-button btn h-10 px-4 text-sm ${isLoading ? "btn-primary-disabled" : "btn-primary"}`}
          >
            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4 text-white" />}
            <span>{isLoading ? "Procesando" : "Responder"}</span>
          </button>
        </div>
      </div>

      {fileWarnings.length > 0 && (
        <div className="rounded-2xl border border-amber-500/20 bg-amber-950/20 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-amber-300">Archivos omitidos</p>
          <ul className="mt-2 space-y-1 text-xs text-amber-100/90">
            {fileWarnings.map((warning) => (
              <li key={warning}>- {warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
