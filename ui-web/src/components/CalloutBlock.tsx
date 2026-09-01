import { Info, AlertTriangle, AlertCircle } from "lucide-react"
import type { ContentBlock } from "../types"

const VARIANT_STYLES: Record<string, { border: string; icon: any; iconColor: string }> = {
  info: {
    border: "border-white/8",
    icon: Info,
    iconColor: "text-zinc-400",
  },
  warning: {
    border: "border-amber-400/15",
    icon: AlertTriangle,
    iconColor: "text-amber-400/80",
  },
  alert: {
    border: "border-red-400/15",
    icon: AlertCircle,
    iconColor: "text-red-400/80",
  },
}

export default function CalloutBlock({ block }: { block: ContentBlock }) {
  const variant = (block.data.variant as string) || "info"
  const message = (block.data.message as string) || ""
  const style = VARIANT_STYLES[variant] || VARIANT_STYLES.info
  const Icon = style.icon

  return (
    <div className={`my-4 rounded-2xl border border-white/5 bg-white/[0.02] ${style.border} px-4 py-3`}>
      <div className="flex items-start gap-3">
        <Icon className={`h-4 w-4 mt-0.5 shrink-0 ${style.iconColor}`} />
        <div className="min-w-0">
          {block.title && (
            <p className="text-sm font-medium text-zinc-200 mb-1">{block.title}</p>
          )}
          <div className="text-sm text-zinc-400 whitespace-pre-wrap leading-relaxed">
            {message}
          </div>
        </div>
      </div>
    </div>
  )
}
