import { useState } from "react"
import { Copy, Check } from "lucide-react"
import type { ContentBlock } from "../types"

export default function CodeBlock({ block }: { block: ContentBlock }) {
  const language = (block.data.language as string) || ""
  const content = (block.data.content as string) || ""
  const [copied, setCopied] = useState(false)

  if (!content) return null

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="my-4 rounded-2xl border border-white/5 bg-zinc-900/50 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
        {language && (
          <span className="text-xs text-zinc-500 font-mono">{language}</span>
        )}
        <button
          onClick={handleCopy}
          className="text-zinc-500 hover:text-zinc-300 transition ml-auto"
          title="Copiar"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-400" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
        </button>
      </div>
      <pre className="px-4 py-3 overflow-x-auto text-sm text-zinc-300 font-mono leading-relaxed">
        <code>{content}</code>
      </pre>
    </div>
  )
}
