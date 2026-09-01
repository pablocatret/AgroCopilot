import { useMemo } from "react"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
} from "recharts"
import type { ContentBlock } from "../types"

interface SeriesPoint {
  date: string
  value: number | null
}

interface Threshold {
  label: string
  range: [number, number]
  status: string
}

function formatDateShort(dateStr: string) {
  if (!dateStr) return ""
  const d = dateStr.length > 10 ? dateStr.slice(0, 10) : dateStr
  const parts = d.split("-")
  if (parts.length === 3) return `${parts[2]}/${parts[1]}`
  return d
}

function tickFormatter(ts: number) {
  const d = new Date(ts)
  return `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}`
}

export default function ChartBlock({ block }: { block: ContentBlock }) {
  const series = (block.data.series as { label: string; points: SeriesPoint[] }[]) || []
  const thresholds = (block.data.thresholds as Threshold[]) || []

  const chartData = useMemo(() => {
    if (!series.length || !series[0].points.length) return []
    return series[0].points
      .filter((p) => p.value !== null && p.value !== undefined)
      .map((p) => ({ date: p.date, value: p.value as number, label: formatDateShort(p.date), timestamp: new Date(p.date + "T00:00:00Z").getTime() }))
      .sort((a, b) => a.date.localeCompare(b.date))
  }, [series])

  if (chartData.length < 2) return null

  const allValues = chartData.map((d) => d.value)
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const padding = (maxVal - minVal) * 0.15 || 0.05
  const yMin = minVal - padding
  const yMax = maxVal + padding
  const yTicks = useMemo(() => {
    const count = 5
    const step = (yMax - yMin) / (count - 1)
    return Array.from({ length: count }, (_, i) => yMin + step * i)
  }, [yMin, yMax])

  const refAreas = thresholds
    .filter((t) => t.range && t.range.length === 2)
    .map((t) => ({
      y1: t.range[0],
      y2: t.range[1],
      color: t.status === "below" ? "rgba(239,68,68,0.08)" : "rgba(34,197,94,0.08)",
    }))

  return (
    <div className="my-4 rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden">
      {block.title && (
        <div className="px-4 py-2.5 border-b border-white/5">
          <span className="text-sm font-medium text-zinc-300">{block.title}</span>
        </div>
      )}
      <div className="px-4 py-4">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={tickFormatter}
              tick={{ fill: "#71717a", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              tickLine={false}
            />
            <YAxis
              domain={[yMin, yMax]}
              ticks={yTicks}
              tickFormatter={(v) => Number(v).toFixed(2)}
              tick={{ fill: "#71717a", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              tickLine={false}
              width={40}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(24,24,27,0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "12px",
                fontSize: 12,
                color: "#d4d4d8",
              }}
              labelFormatter={(ts) => `Fecha: ${formatDateShort(new Date(ts).toISOString().slice(0, 10))}`}
              formatter={(value) => [Number(value).toFixed(3), "Valor"]}
            />
            {refAreas.map((area, i) => (
              <ReferenceArea
                key={i}
                y1={area.y1}
                y2={area.y2}
                fill={area.color}
                stroke="none"
              />
            ))}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#34d399"
              strokeWidth={2}
              dot={{ fill: "#34d399", r: 3, strokeWidth: 0 }}
              activeDot={{ fill: "#34d399", r: 5, stroke: "rgba(255,255,255,0.2)", strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
        {thresholds.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-3">
            {thresholds.map((t, i) => (
              <span key={i} className="text-xs text-zinc-500">
                {t.label}: {t.range[0].toFixed(2)}-{t.range[1].toFixed(2)}
                {t.status === "below" && (
                  <span className="ml-1 text-red-400">▼ por debajo</span>
                )}
                {t.status === "above" && (
                  <span className="ml-1 text-amber-400">▲ por encima</span>
                )}
                {t.status === "normal" && (
                  <span className="ml-1 text-emerald-400">✓ dentro</span>
                )}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
