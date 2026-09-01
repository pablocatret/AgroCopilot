import { useEffect, useState } from "react"
import { X } from "lucide-react"
import type { ContentBlock } from "../types"

export default function ImageBlock({ block }: { block: ContentBlock }) {
  const src = block.data.src as string
  const alt = (block.data.alt as string) || block.title || "Imagen"
  const caption = (block.data.caption as string) || ""
  const [overlay, setOverlay] = useState(false)

  useEffect(() => {
    if (!overlay) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOverlay(false)
    }
    document.addEventListener("keydown", handler)
    document.body.style.overflow = "hidden"
    return () => {
      document.removeEventListener("keydown", handler)
      document.body.style.overflow = ""
    }
  }, [overlay])

  if (!src) return null

  return (
    <>
      <div className="my-4 rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden" style={{ isolation: "isolate" }}>
        {block.title && (
          <div className="px-4 py-2.5 border-b border-white/5">
            <span className="text-sm font-medium text-zinc-300">{block.title}</span>
          </div>
        )}
        <div className="relative group">
          <img
            src={src}
            alt={alt}
            className="w-full h-auto cursor-pointer transition group-hover:brightness-110"
            onClick={() => setOverlay(true)}
            loading="lazy"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition pointer-events-none" />
        </div>
        {caption && (
          <div className="px-4 py-2 border-t border-white/5">
            <span className="text-xs text-zinc-500">{caption}</span>
          </div>
        )}
      </div>

      {overlay && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 cursor-pointer"
          onClick={() => setOverlay(false)}
        >
          <button
            className="absolute top-4 right-4 text-white/70 hover:text-white"
            onClick={() => setOverlay(false)}
          >
            <X className="h-6 w-6" />
          </button>
          <img
            src={src}
            alt={alt}
            className="max-w-full max-h-full object-contain rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  )
}
