from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Tuple

DecisionMode = Literal["case"]
ResponseMode = Literal["conversation"]


class AttachmentMeta(BaseModel):
    attachment_id: str
    filename: str
    content_type: str
    size_bytes: int
    storage_path: Optional[str] = None
    extracted_text: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInput(BaseModel):
    query: str
    language: str = "es"
    attachments: List[AttachmentMeta] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    writer_mode: Optional[Literal["BRIEFING", "STANDARD", "DEEP_DIVE"]] = None
    user_id: Optional[str] = None
    decision_mode: DecisionMode = "case"
    response_mode: ResponseMode = "conversation"
    memory_enabled: bool = False


class AgentRef(BaseModel):
    ref_id: str
    title: str
    source: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRefs(BaseModel):
    items: List[AgentRef] = Field(default_factory=list)


class AgentTrace(BaseModel):
    duration_ms: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    cost_usd: Optional[float] = None


class CostGroup(BaseModel):
    cost_usd: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    unit_count: int = 0
    events: int = 0
    estimated: bool = False


class CostEvent(BaseModel):
    id: str
    ts: Optional[str] = None
    conversation_id: Optional[str] = None
    agent: Optional[str] = None
    operation: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    pricing_mode: Optional[str] = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    unit_count: int = 0
    estimated: bool = False
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CostSummary(BaseModel):
    conversation_id: Optional[str] = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    web_calls: int = 0
    estimated: bool = False
    event_count: int = 0
    top_model: Optional[str] = None
    top_model_cost_usd: float = 0.0
    warning: bool = False
    warning_threshold_usd: float = 0.0
    by_model: Dict[str, CostGroup] = Field(default_factory=dict)
    by_agent: Dict[str, CostGroup] = Field(default_factory=dict)
    by_operation: Dict[str, CostGroup] = Field(default_factory=dict)
    events: List[CostEvent] = Field(default_factory=list)


class BaseAgentOutput(BaseModel):
    agent: str
    status: Literal["ok", "error"] = "ok"
    summary: str = ""
    refs: AgentRefs = Field(default_factory=AgentRefs)
    trace: AgentTrace = Field(default_factory=AgentTrace)
    errors: List[str] = Field(default_factory=list)


class UserQuery(BaseModel):
    query: str
    language: str = "es"


class Citation(BaseModel):
    title: str
    url: str
    source: Literal[
        "web",
        "legal",
        "stac",
        "document",
        "vision",
    ]


class Reference(BaseModel):
    title: str
    url: str
    snippet: str


class LegalDossier(BaseModel):
    authoritative_references: List[Reference] = Field(default_factory=list)
    supporting_references: List[Reference] = Field(default_factory=list)
    verification_mode: Literal["local", "web", "hybrid"] = "local"


class LegalFinding(BaseModel):
    requirement: str
    status: Literal["cumple", "no_cumple", "insuficiente"]
    evidence: List[Citation] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    source_status: Optional[str] = None
    updated_at: Optional[str] = None
    official_source: Optional[str] = None
    article: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class LegalFindings(BaseModel):
    checklist: List[LegalFinding] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    dossier: LegalDossier = Field(default_factory=LegalDossier)
    jurisdiction: Optional[str] = None
    source_status: Optional[str] = None
    updated_at: Optional[str] = None
    official_source: Optional[str] = None
    article: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class TrendData(BaseModel):
    metric: str = ""
    slope: float = 0.0
    r_squared: float = 0.0
    direction: Literal["ascending", "descending", "stable"] = "stable"
    n_dates: int = 0
    date_range: str = ""
    interpretation: str = ""


class RSAnalysisConfig(BaseModel):
    severity_high: float = 0.15
    severity_medium: float = 0.07
    delta_trivial: float = 0.04
    confidence_base: float = 0.72
    confidence_floor: float = 0.35
    confidence_ceiling: float = 0.90
    min_temporal_gap_days: int = 10
    max_temporal_gap_days: int = 180
    phenological_gap_days: int = 45
    radar_penalty: float = 0.06
    quality_penalty: float = 0.18
    collection_mismatch_penalty: float = 0.20
    small_gap_penalty: float = 0.12
    large_gap_penalty: float = 0.08
    phenological_gap_penalty: float = 0.08
    index_mismatch_penalty: float = 0.20


class MeteoContext(BaseModel):
    total_precip_mm: Optional[float] = None
    avg_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    precipitation_irregularity_index: Optional[float] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class LLMImageInterpretation(BaseModel):
    item_id: str = ""
    visible_patterns: List[str] = Field(default_factory=list)
    health_indicators: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    supports_index_signal: Literal["supports", "conflicts", "unclear"] = "unclear"
    confidence: float = 0.6
    raw_description: str = ""


class RemoteSensingClassStat(BaseModel):
    code: int
    label: str
    pixels: int
    percent: float


class RemoteSensingStats(BaseModel):
    index_name: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    valid_pixels: int = 0
    masked_pixels: int = 0
    quality_mask_applied: bool = False
    class_stats: List[RemoteSensingClassStat] = Field(default_factory=list)
    percentile_2: Optional[float] = None
    percentile_98: Optional[float] = None
    cv: Optional[float] = None
    hotspots: Optional[Dict[str, Any]] = None


class SceneQuality(BaseModel):
    label: Literal["alta", "media", "baja", "desconocida"] = "desconocida"
    cloud_cover: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)


class ThresholdContext(BaseModel):
    reference_range: Tuple[float, float] = (0.0, 0.0)
    status: Literal["below", "normal", "above"] = "normal"
    message: str = ""


class StacAsset(BaseModel):
    href: str
    title: Optional[str] = None
    mime_type: Optional[str] = None
    thumbnail: Optional[str] = None


class StacItem(BaseModel):
    id: str
    datetime: Optional[str] = None
    bbox: Optional[List[float]] = None
    collection: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    cloud_cover: Optional[float] = None
    product_type: Optional[str] = None
    product_label: Optional[str] = None
    index_name: Optional[str] = None
    index_stats: Optional[RemoteSensingStats] = None
    quality: Optional[SceneQuality] = None
    change_preview_href: Optional[str] = None
    assets: List[StacAsset] = Field(default_factory=list)
    trend: Optional[TrendData] = None


class TemporalStrategySettings(BaseModel):
    strategy: Literal[
        "auto",
        "recent_pair",
        "monitoring_window",
        "seasonal_baseline",
        "annual_baseline",
        "long_term_change",
    ] = "auto"
    target_gap_days: Optional[int] = None
    force_same_collection: bool = True
    force_same_index: bool = True
    reasoning: str = ""


class TemporalSelection(BaseModel):
    previous_item_id: str
    current_item_id: str
    rationale: str = ""
    strategy: str = "temporal_pair"
    preferred_min_gap_days: int = 10
    actual_gap_days: Optional[int] = None
    used_multi_window_search: bool = False
    query_windows: List[str] = Field(default_factory=list)


class TemporalComparisonContract(BaseModel):
    strategy_settings: TemporalStrategySettings = Field(default_factory=TemporalStrategySettings)
    previous_item_id: str
    current_item_id: str
    previous_datetime: Optional[str] = None
    current_datetime: Optional[str] = None
    collection: Optional[str] = None
    index_name: Optional[str] = None
    rationale: str = ""
    preferred_min_gap_days: int = 10
    actual_gap_days: Optional[int] = None
    used_multi_window_search: bool = False
    query_windows: List[str] = Field(default_factory=list)


class StacResults(BaseModel):
    items: List[StacItem] = Field(default_factory=list)
    temporal_selection: Optional[TemporalSelection] = None
    temporal_contract: Optional[TemporalComparisonContract] = None


class ImageInsight(BaseModel):
    item_id: str
    summary: str
    confidence: float = 0.6
    product_label: Optional[str] = None
    stats: Optional[RemoteSensingStats] = None
    quality: Optional[SceneQuality] = None
    limitations: List[str] = Field(default_factory=list)
    llm_interpretation: Optional[LLMImageInterpretation] = None
    threshold: Optional[ThresholdContext] = None
    spatial_uniformity: Optional[str] = None


class RemoteSensingChange(BaseModel):
    from_item_id: str
    to_item_id: str
    label: str
    detail: str
    confidence: float = 0.6
    metric: Optional[str] = None
    collection: Optional[str] = None
    group_key: Optional[str] = None
    delta_mean: Optional[float] = None
    severity: Literal["alta", "media", "baja"] = "media"
    reliable: bool = False
    limitations: List[str] = Field(default_factory=list)
    trend_context: Optional[str] = None
    preview_href: Optional[str] = None
    threshold: Optional[ThresholdContext] = None


class RemoteSensingFocus(BaseModel):
    title: str
    detail: str
    parcel: Optional[str] = None
    priority: Literal["alta", "media", "baja"] = "media"


class TimeSeriesPoint(BaseModel):
    date: str = ""
    mean: Optional[float] = None
    valid_pixels: int = 0
    quality: Optional[str] = None


class ImageInsights(BaseModel):
    overview: str = ""
    insights: List[ImageInsight] = Field(default_factory=list)
    temporal_changes: List[RemoteSensingChange] = Field(default_factory=list)
    focus_areas: List[RemoteSensingFocus] = Field(default_factory=list)
    trends: Dict[str, TrendData] = Field(default_factory=dict)
    time_series: List[TimeSeriesPoint] = Field(default_factory=list)


class WebFinding(BaseModel):
    claim: str
    citations: List[Citation]


class WebResearch(BaseModel):
    findings: List[WebFinding] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)


class FinancialEstimate(BaseModel):
    item: str
    unit: Optional[str] = None
    capex: Optional[float] = None
    opex: Optional[float] = None
    payback_years: Optional[float] = None
    assumptions: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class FinancialAdvice(BaseModel):
    summary: str = ""
    estimates: List[FinancialEstimate] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)
    citations: List["Citation"] = Field(default_factory=list)  # usa tu Citation existente
    references: List[Reference] = Field(default_factory=list)


class BotanistSpeciesSuggestion(BaseModel):
    common_name: str
    latin_name: Optional[str] = None
    rationale: str
    benefits: List[str] = Field(default_factory=list)
    compatibility: Optional[str] = None
    management_notes: Optional[str] = None


class BotanistAdvice(BaseModel):
    overview: str = ""
    species_suggestions: List[BotanistSpeciesSuggestion] = Field(default_factory=list)
    soil_notes: List[str] = Field(default_factory=list)
    pest_disease_notes: List[str] = Field(default_factory=list)
    biodiversity_practices: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)


class SustainabilityMeasure(BaseModel):
    title: str
    description: str
    expected_impact: Optional[str] = None  # "alto|medio|bajo" (orientativo)
    relative_cost: Optional[str] = None  # "alto|medio|bajo"
    co_benefits: List[str] = Field(default_factory=list)
    risks_or_tradeoffs: List[str] = Field(default_factory=list)
    implementation_notes: Optional[str] = None


class SustainabilityAdvice(BaseModel):
    overview: str = ""
    hotspots: List[str] = Field(default_factory=list)
    measures: List[SustainabilityMeasure] = Field(default_factory=list)
    monitoring_kpis: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)


class NutrientIssue(BaseModel):
    nutrient: str
    evidence: Optional[str] = None
    diagnosis_notes: Optional[str] = None


class FertilizationStrategy(BaseModel):
    goal: str
    recommended_sources: List[str] = Field(default_factory=list)
    example_rates: Optional[str] = None
    application_method: Optional[str] = None
    timing_notes: Optional[str] = None
    precautions: List[str] = Field(default_factory=list)


class CropNutritionAdvice(BaseModel):
    overview: str = ""
    likely_deficiencies: List[NutrientIssue] = Field(default_factory=list)
    strategies: List[FertilizationStrategy] = Field(default_factory=list)
    monitoring: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)


class AidPathway(BaseModel):
    key: str
    title: str
    fit_reason: str
    typical_documents: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)


class CapAdvice(BaseModel):
    overview: str = ""
    pathways: List[AidPathway] = Field(default_factory=list)
    eligibility_signals: List[str] = Field(default_factory=list)
    documents_required: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)


class DocumentStatus(BaseModel):
    name: str
    status: Literal["verificado", "pendiente", "dudoso"]
    rationale: str
    source_attachments: List[str] = Field(default_factory=list)


class DocumentReadiness(BaseModel):
    submission_readiness: Literal["alta", "media", "baja"] = "baja"
    verified_documents: List[DocumentStatus] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    unclear_documents: List[str] = Field(default_factory=list)
    document_quality_risks: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class IntakeQuestion(BaseModel):
    question: str
    why_it_matters: str
    priority: Literal["alta", "media"] = "media"


class FieldIntakeAdvice(BaseModel):
    intake_summary: str = ""
    required_questions: List[IntakeQuestion] = Field(default_factory=list)
    optional_questions: List[IntakeQuestion] = Field(default_factory=list)


class CaseTask(BaseModel):
    title: str
    priority: Literal["alta", "media", "baja"] = "media"
    status: Literal["abierta", "bloqueada", "hecha"] = "abierta"
    rationale: str = ""
    source: Optional[Literal["remote_sensing", "document", "legal", "general"]] = None


class CaseEvidenceItem(BaseModel):
    source: Literal["remote_sensing", "document", "spreadsheet", "vision", "legal", "memory", "general"]
    title: str
    summary: str
    confidence: Optional[float] = None
    status: Literal["usable", "partial", "missing", "failed"] = "usable"
    attachment_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseEvidenceModalitySummary(BaseModel):
    source: Literal["remote_sensing", "document", "spreadsheet", "vision", "legal", "memory", "general"]
    title: str
    usable_items: int = 0
    partial_items: int = 0
    failed_items: int = 0
    missing_items: int = 0
    confidence: Optional[float] = None
    key_signals: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseEvidenceLedger(BaseModel):
    items: List[CaseEvidenceItem] = Field(default_factory=list)
    modalities: List[CaseEvidenceModalitySummary] = Field(default_factory=list)

    def by_status(self, status: str) -> List[CaseEvidenceItem]:
        return [item for item in self.items if item.status == status]

    def by_source(self, source: str) -> List[CaseEvidenceItem]:
        return [item for item in self.items if item.source == source]


class CaseState(BaseModel):
    case_summary: str = ""
    open_tasks: List[CaseTask] = Field(default_factory=list)
    blocked_by: List[str] = Field(default_factory=list)
    recommended_next_input: List[str] = Field(default_factory=list)
    evidence_ledger: CaseEvidenceLedger = Field(default_factory=CaseEvidenceLedger)


class CaseStateDraft(BaseModel):
    case_summary: str = ""
    open_tasks: List[CaseTask] = Field(default_factory=list)
    blocked_by: List[str] = Field(default_factory=list)
    recommended_next_input: List[str] = Field(default_factory=list)


class CaseSnapshot(BaseModel):
    title: str
    decision_mode: str = "decision"
    summary: str = ""
    next_actions: List[str] = Field(default_factory=list)
    blocked_by: List[str] = Field(default_factory=list)


class FieldObservation(BaseModel):
    date: str
    parcel: str
    campaign: Optional[str] = None
    note: str
    severity: Literal["baja", "media", "alta"] = "media"


class TemporalSceneSummary(BaseModel):
    item_id: str
    datetime: Optional[str] = None
    preview_href: Optional[str] = None
    summary: Optional[str] = None
    product_label: Optional[str] = None
    stats: Optional[RemoteSensingStats] = None
    quality: Optional[SceneQuality] = None


class TemporalComparison(BaseModel):
    available: bool = False
    label: str = ""
    rationale: str = ""
    previous: Optional[TemporalSceneSummary] = None
    current: Optional[TemporalSceneSummary] = None
    key_changes: List[str] = Field(default_factory=list)
    metric: Optional[str] = None
    delta_mean: Optional[float] = None
    severity: Optional[Literal["alta", "media", "baja"]] = None
    confidence: Optional[float] = None
    limitations: List[str] = Field(default_factory=list)
    change_preview_href: Optional[str] = None


class PlanPolicy(BaseModel):
    allow_retries: bool = False
    max_rounds: int = 0
    retry_candidates: List[str] = Field(default_factory=list)
    writer_search_allowed: bool = False


class WriterFastPathPolicy(BaseModel):
    enabled: bool = False
    allow_search: bool = False
    disclose_search_use: bool = True
    disclose_sources: bool = True
    escalate_when_specialized: bool = True


class WriterFastPathTrace(BaseModel):
    enabled: bool = False
    search_allowed: bool = False
    search_used: bool = False
    disclose_search_use: bool = True
    disclose_sources: bool = True
    escalation_required: bool = False
    escalation_reason: Optional[str] = None


class EffectivePlanPolicy(PlanPolicy):
    fast_path: WriterFastPathPolicy = Field(default_factory=WriterFastPathPolicy)


class PlanDiagnostics(BaseModel):
    planner_source: Literal["llm", "heuristic", "simple_conversation"] = "heuristic"
    fallback_reason: Optional[str] = None
    rationale: str = ""


class ClarificationOption(BaseModel):
    key: str
    label: str
    description: str = ""
    enriched_query: str


class ClarificationRequest(BaseModel):
    question: str
    options: List[ClarificationOption]
    rationale: str = ""


class MissionEntry(BaseModel):
    agent: str
    instruction: str


class AgentPlan(BaseModel):
    steps: List[str] = Field(default_factory=list)
    missions: List[MissionEntry] = Field(default_factory=list)
    runs: Dict[str, int] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    allow_replan: bool = False
    writer_mode: Optional[Literal["BRIEFING", "STANDARD", "DEEP_DIVE"]] = None
    writer_agent: Optional[Literal["direct_writer", "writer"]] = None
    response_mode: ResponseMode = "conversation"
    policy: EffectivePlanPolicy = Field(default_factory=EffectivePlanPolicy)
    diagnostics: PlanDiagnostics = Field(default_factory=PlanDiagnostics)
    clarification: Optional[ClarificationRequest] = None


class MemoryUsage(BaseModel):
    enabled: bool = False
    user_id: Optional[str] = None
    memory_id: Optional[str] = None
    memory_name: Optional[str] = None
    used_sections: List[str] = Field(default_factory=list)


class ContextUsageItem(BaseModel):
    source_type: Literal["assertion", "task", "event"]
    source_id: str
    label: str
    reason: str
    rank: int


class ContextUsage(BaseModel):
    case_id: Optional[str] = None
    context_run_id: Optional[str] = None
    items: List[ContextUsageItem] = Field(default_factory=list)


class RemoteSensingMemoryArtifact(BaseModel):
    generated_at: str
    query: str = ""
    query_intent: Literal["monitoring", "comparison", "diagnosis", "general"] = "general"
    evidence_level: Literal["retrieval_only", "analyzed_partial", "analyzed_temporal"] = "retrieval_only"
    decision_mode: str = "case"
    memory_id: Optional[str] = None
    memory_name: Optional[str] = None
    parcel: Optional[str] = None
    location_hint: Optional[str] = None
    campaign: Optional[str] = None
    bbox: Optional[List[float]] = None
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    latest_scene_date: Optional[str] = None
    stac_item_ids: List[str] = Field(default_factory=list)
    scene_count: int = 0
    summary: str = ""
    change_highlights: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    pipeline_version: str = "rs-memory-v1"


class MemoryReuseAssessment(BaseModel):
    domain: Literal["remote_sensing"] = "remote_sensing"
    status: Literal["hit", "stale", "miss"] = "miss"
    reason: str = ""
    artifact: Optional[RemoteSensingMemoryArtifact] = None


class MemoryReuseState(BaseModel):
    remote_sensing: MemoryReuseAssessment = Field(default_factory=MemoryReuseAssessment)


class MemoryMeta(BaseModel):
    memory_id: str
    name: str
    created_at: str = ""
    updated_at: str = ""


class MemoryListItem(BaseModel):
    memory_id: str
    name: str
    is_current: bool = False
    used_sections: List[str] = Field(default_factory=list)


class ContinuitySummary(BaseModel):
    case_id: Optional[str] = None
    title: Optional[str] = None
    status: Literal["none", "created", "matched", "active", "ambiguous"] = "none"
    next_step: Optional[str] = None
    created: bool = False
    candidates: List[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    executive_summary: str = ""
    message_md: str = ""
    response_path: Literal["single_agent_fast_path", "multi_agent_synthesis"] = (
        "multi_agent_synthesis"
    )
    search_used: bool = False
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    fast_path: WriterFastPathTrace = Field(default_factory=WriterFastPathTrace)
    legal: Optional[LegalFindings] = None
    remote_sensing: Optional[ImageInsights] = None
    research: Optional[WebResearch] = None
    stac: Optional[StacResults] = None
    recommendations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    report_md: Optional[str] = None
    references: List[AgentRef] = Field(default_factory=list)
    execution: Dict[str, Any] = Field(default_factory=dict)
    cost_summary: Optional[CostSummary] = None
    next_actions: List[str] = Field(default_factory=list)
    evidence_summary: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    documents_needed: List[str] = Field(default_factory=list)
    memory: MemoryUsage = Field(default_factory=MemoryUsage)
    case_id: Optional[str] = None
    continuity: ContinuitySummary = Field(default_factory=ContinuitySummary)
    context_usage: ContextUsage = Field(default_factory=ContextUsage)
    case_state: Optional[CaseState] = None
    temporal_comparison: Optional[TemporalComparison] = None
    attachments: List[AttachmentMeta] = Field(default_factory=list)
    language: Optional[str] = None
    evidence_ledger: CaseEvidenceLedger = Field(default_factory=CaseEvidenceLedger)
    content_blocks: List["ContentBlock"] = Field(default_factory=list)
    citations_resolved: List["ResolvedCitation"] = Field(default_factory=list)


class ContentBlock(BaseModel):
    block_type: Literal["image", "table", "chart", "callout", "code", "separator"]
    ref_id: str = ""
    title: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class ResolvedCitation(BaseModel):
    index: int
    ref_id: str
    start_char: int
    end_char: int


FinalAnswer.model_rebuild()


class LegalAgentOutput(BaseAgentOutput):
    data: LegalFindings = Field(default_factory=LegalFindings)


class ResearchAgentOutput(BaseAgentOutput):
    data: WebResearch = Field(default_factory=WebResearch)


class StacAgentOutput(BaseAgentOutput):
    data: StacResults = Field(default_factory=StacResults)


class RSAgentOutput(BaseAgentOutput):
    data: ImageInsights = Field(default_factory=ImageInsights)


class FinancialAgentOutput(BaseAgentOutput):
    data: FinancialAdvice = Field(default_factory=FinancialAdvice)


class BotanistAgentOutput(BaseAgentOutput):
    data: BotanistAdvice = Field(default_factory=BotanistAdvice)


class SustainabilityAgentOutput(BaseAgentOutput):
    data: SustainabilityAdvice = Field(default_factory=SustainabilityAdvice)


class CropNutritionAgentOutput(BaseAgentOutput):
    data: CropNutritionAdvice = Field(default_factory=CropNutritionAdvice)


class DocumentAgentOutput(BaseAgentOutput):
    data: Dict[str, Any] = Field(default_factory=dict)


class SpreadsheetAgentOutput(BaseAgentOutput):
    data: Dict[str, Any] = Field(default_factory=dict)


class VisionAgentOutput(BaseAgentOutput):
    data: Dict[str, Any] = Field(default_factory=dict)


class WriterAgentOutput(BaseAgentOutput):
    data: FinalAnswer = Field(default_factory=FinalAnswer)


class CapAdvisorAgentOutput(BaseAgentOutput):
    data: CapAdvice = Field(default_factory=CapAdvice)


class DocumentReadinessAgentOutput(BaseAgentOutput):
    data: DocumentReadiness = Field(default_factory=DocumentReadiness)


class FieldIntakeAgentOutput(BaseAgentOutput):
    data: FieldIntakeAdvice = Field(default_factory=FieldIntakeAdvice)


class CaseManagerAgentOutput(BaseAgentOutput):
    data: CaseState = Field(default_factory=CaseState)


class FreeAgentData(BaseModel):
    findings: str = ""
    sources: List[Reference] = Field(default_factory=list)
    confidence: Literal["alta", "media", "baja"] = "media"
    limitations: List[str] = Field(default_factory=list)


class FreeAgentOutput(BaseAgentOutput):
    data: Optional[FreeAgentData] = None
