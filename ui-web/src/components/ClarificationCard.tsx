import { memo, useState } from "react"
import { motion } from "motion/react"
import { HelpCircle, Loader2, Satellite, Scale, Sprout } from "lucide-react"
import type { ClarificationRequest } from "../types"

type Props = {
  clarification: ClarificationRequest
  onSelect: (enrichedQuery: string) => void
  isProcessing?: boolean
}

const ICON_MAP: Record<string, typeof Satellite> = {
  satellite: Satellite,
  legal: Scale,
  general: Sprout,
}

function ClarificationCard({ clarification, onSelect, isProcessing = false }: Props) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null)

  const handleSelect = (key: string, enrichedQuery: string) => {
    if (isProcessing || selectedKey) return
    setSelectedKey(key)
    onSelect(enrichedQuery)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="rounded-2xl border border-emerald-500/20 bg-emerald-950/20 p-4 space-y-3"
    >
      <div className="flex items-center gap-2 text-zinc-200 text-sm font-medium">
        <HelpCircle className="h-4 w-4 text-emerald-400 shrink-0" />
        <span>{clarification.question}</span>
      </div>
      <div className="grid gap-2">
        {clarification.options.map((opt) => {
          const Icon = ICON_MAP[opt.key] ?? HelpCircle
          const isSelected = selectedKey === opt.key
          const isLoading = isProcessing && isSelected
          const isDisabled = isProcessing || (selectedKey !== null && !isSelected)
          return (
            <button
              key={opt.key}
              onClick={() => handleSelect(opt.key, opt.enriched_query)}
              disabled={isDisabled}
              className={`group text-left rounded-xl border px-4 py-3 transition focus:outline-none focus:ring-2 focus:ring-emerald-500/40 ${
                isSelected
                  ? "border-emerald-500/40 bg-emerald-500/10"
                  : "border-white/10 bg-white/[0.03] hover:bg-white/[0.07] hover:border-emerald-500/30"
              } ${isDisabled && !isSelected ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div className="flex items-center gap-2.5">
                {isLoading ? (
                  <Loader2 className="h-4 w-4 text-emerald-400 shrink-0 animate-spin" />
                ) : (
                  <Icon className={`h-4 w-4 shrink-0 transition ${isSelected ? "text-emerald-400" : "text-emerald-400/70 group-hover:text-emerald-400"}`} />
                )}
                <span className={`text-sm font-medium transition ${isSelected ? "text-zinc-100" : "text-zinc-200 group-hover:text-zinc-100"}`}>
                  {opt.label}
                </span>
              </div>
              {opt.description && (
                <p className={`mt-1 ml-[26px] text-xs leading-relaxed transition ${isSelected ? "text-zinc-300" : "text-zinc-400 group-hover:text-zinc-300"}`}>
                  {opt.description}
                </p>
              )}
            </button>
          )
        })}
      </div>
    </motion.div>
  )
}

export default memo(ClarificationCard)
