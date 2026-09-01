export type DecisionMode = "case"

export type WriterFastPathPolicy = {
  enabled?: boolean
  allow_search?: boolean
  disclose_search_use?: boolean
  disclose_sources?: boolean
  escalate_when_specialized?: boolean
}
export type EffectivePlanPolicy = {
  allow_retries?: boolean
  max_rounds?: number
  retry_candidates?: string[]
  writer_search_allowed?: boolean
  fast_path?: WriterFastPathPolicy
}
export type PlanDiagnostics = {
  planner_source?: "llm"|"heuristic"|"simple_conversation"
  fallback_reason?: string | null
  rationale?: string
}
export type AgentPlan = {
  steps: string[]
  runs?: Record<string, number>
  dependencies?: Record<string, string[]>
  allow_replan?: boolean
  writer_mode?: "BRIEFING"|"STANDARD"|"DEEP_DIVE"
  writer_agent?: "direct_writer" | "writer"
  policy?: EffectivePlanPolicy
  diagnostics?: PlanDiagnostics
}
export type AgentRef = { ref_id: string; title: string; source: string; url?: string; snippet?: string; metadata?: Record<string, any> }
export type AttachmentMeta = { attachment_id: string; filename: string; content_type: string; size_bytes: number; summary?: string }
export type Citation = { title: string; url?: string; source?: string }
export type LegalFinding = {
  requirement: string
  status: "cumple"|"no_cumple"|"insuficiente"
  evidence: Citation[]
  jurisdiction?: string | null
  source_status?: string | null
  updated_at?: string | null
  official_source?: string | null
  article?: string | null
  limitations?: string[]
}
export type LegalReference = { title: string; url: string; snippet: string }
export type LegalDossier = {
  authoritative_references: LegalReference[]
  supporting_references: LegalReference[]
  verification_mode: "local"|"web"|"hybrid"
}
export type LegalFindings = {
  checklist: LegalFinding[]
  answer?: string
  citations?: Citation[]
  references?: LegalReference[]
  dossier?: LegalDossier
  jurisdiction?: string | null
  source_status?: string | null
  updated_at?: string | null
  official_source?: string | null
  article?: string | null
  limitations?: string[]
}

export type WebFinding = { claim: string; citations: Citation[] }
export type WebResearch = { findings: WebFinding[]; references?: { title: string; url: string; snippet: string }[] }

export type RemoteSensingClassStat = {
  code: number
  label: string
  pixels: number
  percent: number
}

export type RemoteSensingStats = {
  index_name?: string | null
  min?: number | null
  max?: number | null
  mean?: number | null
  std?: number | null
  valid_pixels?: number
  masked_pixels?: number
  quality_mask_applied?: boolean
  class_stats?: RemoteSensingClassStat[]
}
export type SceneQuality = {
  label: "alta"|"media"|"baja"|"desconocida"
  cloud_cover?: number | null
  reasons?: string[]
}
export type StacAsset = { href: string; title?: string; mime_type?: string; thumbnail?: string }
export type StacItem = {
  id: string
  datetime?: string
  bbox?: number[]
  collection?: string | null
  properties?: Record<string, any>
  cloud_cover?: number | null
  product_type?: string | null
  product_label?: string | null
  index_name?: string | null
  index_stats?: RemoteSensingStats | null
  quality?: SceneQuality | null
  change_preview_href?: string | null
  assets: StacAsset[]
}
export type TemporalSelection = {
  previous_item_id: string
  current_item_id: string
  rationale?: string
  strategy?: string
  preferred_min_gap_days?: number
  actual_gap_days?: number | null
  used_multi_window_search?: boolean
  query_windows?: string[]
}
export type StacResults = { items: StacItem[]; temporal_selection?: TemporalSelection | null }

export type ImageInsight = {
  item_id: string
  summary: string
  confidence: number
  product_label?: string | null
  stats?: RemoteSensingStats | null
  quality?: SceneQuality | null
  limitations?: string[]
  threshold?: ThresholdContext | null
  spatial_uniformity?: string | null
}
export type RemoteSensingChange = {
  from_item_id: string
  to_item_id: string
  label: string
  detail: string
  confidence: number
  metric?: string | null
  delta_mean?: number | null
  severity?: "alta"|"media"|"baja"
  reliable?: boolean
  limitations?: string[]
  preview_href?: string | null
  threshold?: ThresholdContext | null
}
export type RemoteSensingFocus = { title: string; detail: string; parcel?: string; priority: "alta"|"media"|"baja" }
export type ThresholdContext = {
  reference_range: [number, number]
  status: "below" | "normal" | "above"
  message: string
}
export type TrendData = {
  metric: string
  slope: number
  r_squared: number
  direction: "ascending" | "descending" | "stable"
  n_dates: number
  date_range: string
  interpretation: string
}
export type TimeSeriesPoint = {
  date: string
  mean?: number | null
  valid_pixels: number
  quality?: string | null
}
export type ImageInsights = {
  overview?: string
  insights: ImageInsight[]
  temporal_changes?: RemoteSensingChange[]
  focus_areas?: RemoteSensingFocus[]
  trends?: Record<string, TrendData>
  time_series?: TimeSeriesPoint[]
}

export type ContentBlock = {
  block_type: "image" | "table" | "chart" | "callout" | "code" | "separator"
  ref_id: string
  title: string
  data: Record<string, any>
}
export type ResolvedCitation = {
  index: number
  ref_id: string
  start_char: number
  end_char: number
}

export type FinancialEstimate = { item: string; unit?: string; capex?: number; opex?: number; payback_years?: number; assumptions?: string[]; notes?: string }
export type FinancialAdvice = { summary: string; estimates: FinancialEstimate[]; risks: string[]; tips: string[]; citations: Citation[] }

export type BotanistSpeciesSuggestion = { common_name: string; latin_name?: string; rationale: string; benefits?: string[]; compatibility?: string; management_notes?: string }
export type BotanistAdvice = { overview: string; species_suggestions: BotanistSpeciesSuggestion[]; soil_notes: string[]; pest_disease_notes: string[]; biodiversity_practices: string[]; citations: Citation[] }

export type SustainabilityMeasure = { title: string; description: string; expected_impact?: string; relative_cost?: string; co_benefits?: string[]; risks_or_tradeoffs?: string[]; implementation_notes?: string }
export type SustainabilityAdvice = { overview: string; hotspots: string[]; measures: SustainabilityMeasure[]; monitoring_kpis: string[]; citations: Citation[] }

export type NutrientIssue = { nutrient: string; evidence?: string; diagnosis_notes?: string }
export type FertilizationStrategy = { goal: string; recommended_sources: string[]; example_rates?: string; application_method?: string; timing_notes?: string; precautions?: string[] }
export type CropNutritionAdvice = { overview: string; likely_deficiencies: NutrientIssue[]; strategies: FertilizationStrategy[]; monitoring: string[]; citations: Citation[] }
export type AidPathway = { key: string; title: string; fit_reason: string; typical_documents: string[]; cautions: string[] }
export type CapAdvice = { overview: string; pathways: AidPathway[]; eligibility_signals: string[]; documents_required: string[]; gaps: string[]; next_steps: string[] }
export type DocumentStatus = { name: string; status: "verificado"|"pendiente"|"dudoso"; rationale: string; source_attachments: string[] }
export type DocumentReadiness = { submission_readiness: "alta"|"media"|"baja"; verified_documents: DocumentStatus[]; missing_documents: string[]; unclear_documents: string[]; document_quality_risks: string[]; next_steps: string[] }
export type IntakeQuestion = { question: string; why_it_matters: string; priority: "alta"|"media" }
export type FieldIntakeAdvice = { intake_summary: string; required_questions: IntakeQuestion[]; optional_questions: IntakeQuestion[] }
export type CaseTask = {
  title: string
  priority: "alta"|"media"|"baja"
  status: "abierta"|"bloqueada"|"hecha"
  rationale: string
  source?: "remote_sensing"|"document"|"legal"|"general"
}
export type CaseEvidenceItem = {
  source: "remote_sensing"|"document"|"spreadsheet"|"vision"|"legal"|"memory"|"general"
  title: string
  summary: string
  confidence?: number | null
  status: "usable"|"partial"|"missing"|"failed"
  attachment_id?: string | null
  metadata?: Record<string, any>
}
export type CaseEvidenceModalitySummary = {
  source: "remote_sensing"|"document"|"spreadsheet"|"vision"|"legal"|"memory"|"general"
  title: string
  usable_items: number
  partial_items: number
  failed_items: number
  missing_items: number
  confidence?: number | null
  key_signals: string[]
  limitations: string[]
  metadata?: Record<string, any>
}
export type CaseEvidenceLedger = {
  items: CaseEvidenceItem[]
  modalities: CaseEvidenceModalitySummary[]
}
export type CaseState = {
  case_summary: string
  open_tasks: CaseTask[]
  blocked_by: string[]
  recommended_next_input: string[]
  evidence_ledger?: CaseEvidenceLedger
}
export type ContinuitySummary = {
  case_id?: string | null
  title?: string | null
  status: "none" | "created" | "matched" | "active" | "ambiguous"
  next_step?: string | null
  created: boolean
  candidates: string[]
}

export type AgentStatus = "queued"|"running"|"done"|"error"|"skipped"
export type ExecutionLevel = "ok"|"insufficient_data"|"soft_error"|"hard_error"
export type LogEntry = { message: string; level: string; timestamp: string; agent?: string }
export type ExecutionInstance = { instance_id: number; level: ExecutionLevel; message: string }
export type ExecutionAgentState = { final_level: ExecutionLevel; instances: ExecutionInstance[] }
export type ExecutionReport = Record<string, ExecutionAgentState>
export type CostGroup = {
  cost_usd: number
  input_tokens: number
  cached_input_tokens: number
  output_tokens: number
  total_tokens: number
  unit_count: number
  events: number
  estimated: boolean
}
export type CostEvent = {
  id: string
  ts?: string | null
  conversation_id?: string | null
  agent?: string | null
  operation?: string | null
  provider?: string | null
  model?: string | null
  pricing_mode?: string | null
  input_tokens: number
  cached_input_tokens: number
  output_tokens: number
  total_tokens: number
  unit_count: number
  estimated: boolean
  cost_usd: number
  metadata?: Record<string, any>
}
export type CostSummary = {
  conversation_id?: string | null
  total_cost_usd: number
  total_tokens: number
  input_tokens: number
  cached_input_tokens: number
  output_tokens: number
  web_calls: number
  estimated: boolean
  event_count: number
  top_model?: string | null
  top_model_cost_usd: number
  warning: boolean
  warning_threshold_usd: number
  by_model: Record<string, CostGroup>
  by_agent: Record<string, CostGroup>
  by_operation: Record<string, CostGroup>
  events: CostEvent[]
}
export type MemoryUsage = {
  enabled: boolean
  user_id?: string | null
  memory_id?: string | null
  memory_name?: string | null
  used_sections: string[]
}
export type EditableMemorySections = {
  profile: string
  preferences: string
  farm_context: string
  open_questions: string
}
export type UserMemoryRecord = {
  user_id: string
  sections: EditableMemorySections
  used_sections: string[]
}
export type MemoryMeta = {
  memory_id: string
  name: string
  created_at?: string
  updated_at?: string
}
export type MemoryListItem = {
  memory_id: string
  name: string
  is_current: boolean
  used_sections: string[]
}
export type FieldObservation = { observation_id?: string; case_id?: string; date: string; parcel: string; campaign?: string | null; note: string; severity: "baja"|"media"|"alta" }
export type ConversationSummary = {
  conversation_id: string
  user_id?: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  case_id?: string | null
}
export type CaseStatus = "active" | "on_hold" | "closed" | "archived" | "deleted"
export type CaseRecord = {
  case_id: string
  workspace_id: string
  title: string
  objective: string
  status: CaseStatus
  summary: string
  created_at: string
  updated_at: string
  last_activity_at: string
}
export type CaseEvent = {
  event_id: string
  case_id: string
  sequence_no: number
  event_type: string
  actor_type: string
  source_type?: string | null
  source_id?: string | null
  payload: Record<string, any>
  created_at: string
}
export type CaseAssertion = {
  assertion_id: string
  workspace_id: string
  case_id?: string | null
  scope: "case" | "global"
  assertion_type: string
  key: string
  value_text: string
  display_text: string
  provenance: string
  confidence?: number | null
  status: "proposed" | "confirmed" | "superseded" | "retracted" | "expired"
  valid_from?: string | null
  valid_until?: string | null
  supersedes_assertion_id?: string | null
  created_at: string
  updated_at: string
}
export type CaseTaskRecord = {
  task_id: string
  case_id: string
  title: string
  rationale: string
  priority: string
  status: "proposed" | "open" | "blocked" | "done" | "cancelled"
  created_by: string
  created_at: string
  updated_at: string
}
export type CaseProjection = {
  case_id: string
  summary: string
  confirmed_facts: CaseAssertion[]
  proposed_assertions: CaseAssertion[]
  active_tasks: CaseTaskRecord[]
  conflicts: Array<{ key: string; assertions: CaseAssertion[] }>
  review_count: number
  updated_at: string
}
export type CaseDetail = {
  case: CaseRecord
  projection: CaseProjection
  events: CaseEvent[]
  assertions: CaseAssertion[]
  tasks: CaseTaskRecord[]
  decisions: Array<Record<string, any>>
  observations?: FieldObservation[]
}
export type WorkspaceContext = {
  workspace_id: string
  name: string
  zone: string
  crops: string
  infrastructure: string
  constraints: string
  preferences: string
  updated_at?: string
}
export type ContextUsageItem = {
  source_type: "assertion" | "task" | "event"
  source_id: string
  label: string
  reason: string
  rank: number
}
export type ContextUsage = { case_id?: string | null; context_run_id?: string | null; items: ContextUsageItem[] }
export type ConversationMessage = {
  id: number
  role: "user" | "assistant"
  query?: string
  response_mode?: string
  answer_summary?: string
  answer_json?: string
  file_names?: string[]
  timestamp: string
}
export type TemporalSceneSummary = {
  item_id: string
  datetime?: string
  preview_href?: string
  summary?: string
  product_label?: string | null
  stats?: RemoteSensingStats | null
  quality?: SceneQuality | null
}
export type TemporalComparison = {
  available: boolean
  label: string
  rationale: string
  previous?: TemporalSceneSummary | null
  current?: TemporalSceneSummary | null
  key_changes: string[]
  metric?: string | null
  delta_mean?: number | null
  severity?: "alta"|"media"|"baja" | null
  confidence?: number | null
  limitations?: string[]
  change_preview_href?: string | null
}

export type WriterFastPathTrace = {
  enabled?: boolean
  search_allowed?: boolean
  search_used?: boolean
  disclose_search_use?: boolean
  disclose_sources?: boolean
  escalation_required?: boolean
  escalation_reason?: string | null
}
export type FinalAnswer = {
  executive_summary: string
  message_md?: string
  response_path?: "single_agent_fast_path"|"multi_agent_synthesis"
  search_used?: boolean
  escalation_required?: boolean
  escalation_reason?: string | null
  fast_path?: WriterFastPathTrace
  legal?: LegalFindings | null
  remote_sensing?: ImageInsights | null
  research?: WebResearch | null
  stac?: StacResults
  report_md?: string
  references?: AgentRef[]
  execution?: ExecutionReport
  cost_summary?: CostSummary | null
  case_state?: CaseState
  case_id?: string | null
  continuity?: ContinuitySummary
  context_usage?: ContextUsage
  temporal_comparison?: TemporalComparison
  recommendations?: string[]
  limitations?: string[]
  language?: string
  attachments?: AttachmentMeta[]
  next_actions?: string[]
  evidence_summary?: string[]
  missing_information?: string[]
  documents_needed?: string[]
  memory?: MemoryUsage
  evidence_ledger?: CaseEvidenceLedger
  content_blocks?: ContentBlock[]
  citations_resolved?: ResolvedCitation[]
}

export type ClarificationOption = {
  key: string
  label: string
  description?: string
  enriched_query: string
}

export type ClarificationRequest = {
  question: string
  options: ClarificationOption[]
  rationale?: string
}

export type ChatResponse = {
  conversation_id?: string
  plan: AgentPlan
  answer: FinalAnswer
  clarification?: ClarificationRequest
}

export type AgentDetailData = {
  model?: string | null
  provider?: string | null
  mission?: string
  toolsAvailable?: string
  toolsUsed?: string[]
  contextSummary?: Record<string, any>
  outputPreview?: string
  executionLevel?: ExecutionLevel
  trace?: {
    durationMs: number
    tokensInput: number
    tokensOutput: number
    costUsd: number
  } | null
}

type BaseEvent = { conversation_id: string; actor?: string; ts?: string }

export type ServerEvent =
  | (BaseEvent & { type: "status"; stage: "received"|"completed"; message: string })
  | (BaseEvent & { type: "plan"; steps?: string[]; agents?: string[]; runs?: Record<string, number>; dependencies?: Record<string, string[]>; replanned?: boolean })
  | (BaseEvent & { type: "agent_status"; agent: string; status: AgentStatus; message?: string; run_id?: number; total_runs?: number; attempt?: number; attempt_limit?: number; run_key?: string })
  | (BaseEvent & { type: "agent_detail"; agent: string; run_key?: string; model?: string | null; provider?: string | null; mission?: string; tools_available?: string; tools_used?: string[]; context_summary?: Record<string, any>; output_preview?: string; execution_level?: ExecutionLevel; trace?: { duration_ms: number; tokens_input: number; tokens_output: number; cost_usd: number } | null })
  | (BaseEvent & { type: "log"; level: string; timestamp: string; message: string; agent?: string; run_id?: number; attempt?: number })
  | (BaseEvent & { type: "error"; error: string })
