import { useState } from "react"
import { ExternalLink, X } from "lucide-react"
import type { StacResults, TemporalComparison } from "../types"

export default function Gallery({ stac, comparison }: { stac?: StacResults; comparison?: TemporalComparison | null }) {
  const items = stac?.items || []
  const thumbs: { href: string; id: string; dt?: string; full?: string; downloadName?: string; label?: string; quality?: string; mean?: number | null }[] = []
  for (const it of items) {
    const assets = it.assets || []
    const thumbAsset = assets.find((a) => a.thumbnail)
    const fullAsset = assets.find((a) => a.href && (!a.thumbnail || !a.thumbnail.startsWith("data:image")))
    if (thumbAsset?.thumbnail) {
      const isData = thumbAsset.thumbnail.startsWith("data:image")
      thumbs.push({
        href: thumbAsset.thumbnail,
        id: it.id,
        dt: it.datetime,
        full: isData ? thumbAsset.thumbnail : fullAsset?.href || thumbAsset.thumbnail,
        downloadName: isData ? `producto-${it.id || "image"}.svg` : undefined,
        label: it.product_label || it.index_name || it.collection || "Producto satelital",
        quality: it.quality?.label,
        mean: it.index_stats?.mean,
      })
    }
  }
  if (!thumbs.length && !comparison?.available) return null

  const [overlayImg, setOverlayImg] = useState<{ src: string; alt: string } | null>(null)

  return (
    <>
      <section className="gallery-compact">
        <header className="gallery-compact-header">
          <div className="min-w-0">
            <p className="eyebrow">Galería satelital</p>
            <h3 className="section-title text-2xl text-zinc-50">Productos satelitales</h3>
          </div>
          <span className="rail-badge rail-badge-soft">{thumbs.length} productos</span>
        </header>

        {comparison?.available && (comparison.previous || comparison.current) ? (
          <div className="gallery-comparison-compact">
            <div className="min-w-0">
              <p className="eyebrow">Seleccion temporal</p>
              <h4 className="text-sm font-semibold text-zinc-100">{comparison.label || "Antes vs ahora"}</h4>
            </div>
            <div className="gallery-comparison-pair">
              {[comparison.previous, comparison.current].map((scene, index) =>
                scene ? (
                  <button
                    key={`${scene.item_id}-${index}`}
                    type="button"
                    className="gallery-comparison-thumb"
                    onClick={() => scene.preview_href && setOverlayImg({ src: scene.preview_href, alt: scene.item_id })}
                  >
                    {scene.preview_href ? <img src={scene.preview_href} alt={scene.item_id} loading="lazy" /> : null}
                    <span>{index === 0 ? "Ref" : "Actual"}</span>
                  </button>
                ) : null
              )}
            </div>
          </div>
        ) : null}

        {thumbs.length > 0 && (
          <div className="gallery-strip">
            {thumbs.map((t) => (
              <button
                key={t.id}
                type="button"
                className="gallery-strip-item"
                onClick={() => setOverlayImg({ src: t.full || t.href, alt: t.label || t.id })}
              >
                <img src={t.href} alt={t.id} loading="lazy" />
                <div className="gallery-strip-meta">
                  <span>{t.label}</span>
                  {t.mean !== undefined && t.mean !== null ? <span>{t.mean.toFixed(3)}</span> : null}
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      {overlayImg && (
        <div className="gallery-overlay-backdrop" onClick={() => setOverlayImg(null)}>
          <div className="gallery-overlay-panel" onClick={(e) => e.stopPropagation()}>
            <div className="gallery-overlay-header">
              <span className="text-sm text-zinc-300 truncate">{overlayImg.alt}</span>
              <div className="flex items-center gap-2">
                <a href={overlayImg.src} target="_blank" rel="noreferrer" className="text-emerald-400 hover:text-emerald-300">
                  <ExternalLink size={16} />
                </a>
                <button type="button" onClick={() => setOverlayImg(null)} className="text-zinc-400 hover:text-zinc-200">
                  <X size={18} />
                </button>
              </div>
            </div>
            <img src={overlayImg.src} alt={overlayImg.alt} className="gallery-overlay-img" />
          </div>
        </div>
      )}
    </>
  )
}
