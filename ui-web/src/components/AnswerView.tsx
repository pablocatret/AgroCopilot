import { memo, useMemo } from "react"
import { motion } from "motion/react"
import MarkdownIt from "markdown-it"
import DOMPurify from "dompurify"
import type { FinalAnswer, ContentBlock, AgentRef } from "../types"
import ContentBlockRenderer from "./ContentBlockRenderer"

const md = new MarkdownIt({ linkify: true, breaks: true })

function escapeHtmlAttr(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function renderCitations(html: string, references: AgentRef[]): string {
  return html.replace(/\[(\d+)\]/g, (_, rawIndex) => {
    const index = Number(rawIndex)
    const ref = references[index - 1]
    if (!ref) return `<sup class="citation-mark">[${index}]</sup>`
    const title = ref.url ? `${ref.title} — ${ref.url}` : ref.title
    const href = ref.url || "#"
    return `<sup class="citation-mark" title="${escapeHtmlAttr(title)}"><a href="${href}" target="_blank" rel="noopener" class="text-emerald-400 hover:text-emerald-300 no-underline">[${index}]</a></sup>`
  })
}

function renderBlocks(blocks: ContentBlock[], position: "before" | "after") {
  if (!blocks.length) return null
  return (
    <div className={position === "before" ? "space-y-3 mb-4" : "space-y-3 mt-4"}>
      {blocks.map((block) => <ContentBlockRenderer key={block.ref_id} block={block} />)}
    </div>
  )
}

function AnswerView({ answer, completedRuns = 0, totalRuns = 0 }: {
  answer: FinalAnswer | null
  isRunning?: boolean
  completedRuns?: number
  totalRuns?: number
}) {
  const references = answer?.references || []
  const contentBlocks = answer?.content_blocks || []
  const visibleMarkdown = answer?.message_md || answer?.report_md || answer?.executive_summary || ""
  const rendered = useMemo(() => {
    if (!visibleMarkdown) return ""
    return DOMPurify.sanitize(renderCitations(md.render(visibleMarkdown), references))
  }, [visibleMarkdown, references])
  const inlineBlockIds = useMemo(() => {
    const ids = new Set<string>()
    for (const match of rendered.matchAll(/\{ref:([^}]+)\}/g)) ids.add(match[1])
    return ids
  }, [rendered])
  const inlineBlocks = contentBlocks.filter((block) => inlineBlockIds.has(block.ref_id))
  const trailingBlocks = contentBlocks.filter((block) => !inlineBlockIds.has(block.ref_id))

  if (!answer) return null

  return (
    <motion.article
      data-capture="chat-answer"
      className="conversation-answer animate-rise space-y-4"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24, ease: [0.2, 0.8, 0.2, 1] }}
    >
      {references.length || answer.next_actions?.length || answer.missing_information?.length ? (
        <div className="answer-meta-row">
          {references.length ? <span>{references.length} fuentes</span> : null}
          {answer.next_actions?.length ? <span>{answer.next_actions.length} acciones</span> : null}
          {answer.missing_information?.length ? <span>{answer.missing_information.length} huecos</span> : null}
          {totalRuns ? <span>{completedRuns}/{totalRuns} pasos</span> : null}
        </div>
      ) : null}
      {inlineBlocks.length ? renderBlocks(inlineBlocks, "before") : null}
      <div className="prose prose-lg max-w-none prose-theme conversation-prose" dangerouslySetInnerHTML={{ __html: rendered }} />
      {trailingBlocks.length ? renderBlocks(trailingBlocks, "after") : null}

      {answer.case_state && (answer.case_state.open_tasks?.length || answer.case_state.blocked_by?.length) ? (
        <section className="inner-card p-4">
          <p className="eyebrow">Estado del caso</p>
          {answer.case_state.case_summary ? <p className="mt-2 text-sm text-zinc-300">{answer.case_state.case_summary}</p> : null}
          {answer.case_state.open_tasks?.slice(0, 4).map((task) => <p key={task.title} className="mt-2 text-xs text-zinc-400">[{task.priority}] {task.title}</p>)}
        </section>
      ) : null}

    </motion.article>
  )
}

export default memo(AnswerView)
