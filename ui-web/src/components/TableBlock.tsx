import type { ContentBlock } from "../types"

export default function TableBlock({ block }: { block: ContentBlock }) {
  const headers = (block.data.headers as string[]) || []
  const rows = (block.data.rows as string[][]) || []
  const highlightCol = block.data.highlight_col as number | undefined

  if (!headers.length && !rows.length) return null

  return (
    <div className="my-4 rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden">
      {block.title && (
        <div className="px-4 py-2.5 border-b border-white/5">
          <span className="text-sm font-medium text-zinc-300">{block.title}</span>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          {headers.length > 0 && (
            <thead>
              <tr className="border-b border-white/5">
                {headers.map((h, i) => (
                  <th
                    key={i}
                    className="px-4 py-2 text-left text-xs font-medium text-zinc-400 uppercase tracking-wider"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {rows.map((row, ri) => (
              <tr
                key={ri}
                className="border-b border-white/[0.03] last:border-b-0 hover:bg-white/[0.02] transition"
              >
                {row.map((cell, ci) => (
                  <td
                    key={ci}
                    className={`px-4 py-2 text-zinc-300 ${
                      ci === highlightCol ? "text-emerald-400 font-medium" : ""
                    }`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
