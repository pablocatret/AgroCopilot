import type { ContentBlock } from "../types"
import ImageBlock from "./ImageBlock"
import TableBlock from "./TableBlock"
import ChartBlock from "./ChartBlock"
import CalloutBlock from "./CalloutBlock"
import CodeBlock from "./CodeBlock"

export default function ContentBlockRenderer({ block }: { block: ContentBlock }) {
  switch (block.block_type) {
    case "image":
      return <ImageBlock block={block} />
    case "table":
      return <TableBlock block={block} />
    case "chart":
      return <ChartBlock block={block} />
    case "callout":
      return <CalloutBlock block={block} />
    case "code":
      return <CodeBlock block={block} />
    case "separator":
      return <hr className="my-6 border-white/10" />
    default:
      return null
  }
}
