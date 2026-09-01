"""Métricas deterministas para evaluación de modelos.

Métricas calculables sin LLM, reproducibles, y basadas en gold expectations.
"""
from __future__ import annotations

import re
import unicodedata

from evaluation.schemas import CaseSpec, ExecutionMetrics, NormalizedOutput


# ── Normalización ────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Normaliza texto para comparación: lowercase, sin acentos, espacios múltiples."""
    text = unicodedata.normalize("NFKD", text.lower().strip())
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Tokeniza texto en palabras alfabéticas."""
    return [t for t in re.findall(r"[a-záéíóúñü]+", _normalize(text)) if len(t) > 2]


def _tokens_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Calcula el recall de tokens de a sobre b."""
    if not tokens_a:
        return 0.0
    set_b = set(tokens_b)
    hits = sum(1 for t in tokens_a if t in set_b)
    return hits / len(tokens_a)


NEGATED_CLAIM_PATTERNS = (
    r"no\s+(?:hay\s+)?evidencia\s+de\s+",
    r"no\s+se\s+(?:observa|confirma|puede\s+afirmar)\s+",
    r"sin\s+evidencia\s+de\s+",
    r"no\s+es\s+",
    r"descart(?:a|ado|ada|ar)\s+",
)


def _is_explicitly_negated(response_text: str, claim: str) -> bool:
    """Detecta negaciones explícitas inmediatamente anteriores a un claim.

    Esto no intenta resolver negación lingüística general. Solo evita el falso
    positivo más importante: penalizar que el sistema mencione un claim para
    descartarlo o decir que no hay evidencia.
    """
    normalized_claim = _normalize(claim)
    if not normalized_claim or normalized_claim not in response_text:
        return False
    claim_start = response_text.find(normalized_claim)
    prefix = response_text[max(0, claim_start - 80):claim_start]
    return any(re.search(pattern + r"$", prefix) for pattern in NEGATED_CLAIM_PATTERNS)


# ── forbidden_claim_rate ─────────────────────────────────────────────


def compute_forbidden_claim_rate(
    case: CaseSpec,
    output: NormalizedOutput,
) -> float:
    """Mide la proporción de claims prohibidos que aparecen en la respuesta.

    Returns:
        Ratio [0,1] donde 0 = sin claims prohibidos (bueno).
    """
    forbidden = case.gold_expectations.forbidden_claims
    if not forbidden:
        return 0.0

    response_text = _normalize(output.evaluable_text)

    violations = 0
    for claim in forbidden:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            continue
        response_tokens = _tokenize(response_text)
        overlap = _tokens_overlap(claim_tokens, response_tokens)
        if overlap >= 0.6 and not _all_occurrences_negated(response_text, claim):
            violations += 1

    return violations / len(forbidden)


def _all_occurrences_negated(response_text: str, claim: str) -> bool:
    """Return True only when every occurrence of a claim is explicitly negated."""
    normalized_claim = _normalize(claim)
    starts: list[int] = []
    offset = 0
    while normalized_claim:
        index = response_text.find(normalized_claim, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + len(normalized_claim)
    return bool(starts) and all(
        any(re.search(pattern + r"$", response_text[max(0, start - 80):start])
            for pattern in NEGATED_CLAIM_PATTERNS)
        for start in starts
    )


# ── overclaim detection ──────────────────────────────────────────────

OVERCLAIM_PATTERNS = [
    r"\bgaranti[zzs]\w*\b",
    r"\bsin\s+riesgo\b",
    r"\b100\s*%\b",
    r"\bdefinitiv\w+\b",
    r"\bsegur[oa]\s+al\s+100\b",
    r"\bcur[oa]\w*\s+definitiv\w+\b",
    r"\bresolv\w*\s+totalmente\b",
]


def compute_overclaim_rate(output: NormalizedOutput) -> float:
    """Detecta afirmaciones excesivamente categóricas en la respuesta.

    Returns:
        Número de overclaims detectados (entero).
    """
    text = _normalize(output.evaluable_text)
    count = 0
    for pattern in OVERCLAIM_PATTERNS:
        count += len(re.findall(pattern, text))
    return float(count)


# ── actionability ────────────────────────────────────────────────────

VERB_PATTERNS = [
    r"\b(comprob\w*|medir|mid\w*|aplic\w*|analiz\w*|consult\w*|instal\w*|reg\w*|pod\w*|fertiliz\w*)\b",
    r"\b(revis\w*|verific\w*|observ\w*|control\w*|monitore\w*|trat\w*)\b",
    r"\b(registr\w*|solicit\w*|present\w*|declar\w*|prob\w*|cambi\w*)\b",
]


def _score_actions(actions: list[str]) -> float:
    """Mide si las acciones propuestas son concretas y específicas.

    Returns:
        Ratio [0,1] donde 1 = todas las acciones son específicas.
    """
    if not actions:
        return 0.0

    specific = 0
    for action in actions:
        text = _normalize(action)
        has_verb = any(re.search(p, text) for p in VERB_PATTERNS)
        tokens = _tokenize(action)
        if has_verb and len(tokens) >= 2:
            specific += 1

    return specific / len(actions)


def compute_actionability_structured(output: NormalizedOutput) -> float:
    """Score actions explicitly exposed in the structured response field."""
    return _score_actions(output.next_actions)


def compute_actionability_visible(output: NormalizedOutput) -> float:
    """Score concrete action sentences in a conversation response."""
    text = output.visible_text.strip()
    if not text:
        return 0.0
    segments = [segment.strip(" -*•") for segment in re.split(r"[\n.!?;]+", text)]
    candidates = [
        segment for segment in segments
        if segment and any(re.search(pattern, _normalize(segment)) for pattern in VERB_PATTERNS)
    ]
    return _score_actions(candidates)


def compute_actionability(output: NormalizedOutput) -> float:
    """Use structured actions when present, otherwise visible conversation text."""
    if output.next_actions:
        return compute_actionability_structured(output)
    return compute_actionability_visible(output)


def compute_answer_completeness(case: CaseSpec, output: NormalizedOutput) -> float:
    """Compute deterministic coverage of all structured gold expectations."""
    gold = case.gold_expectations
    response_tokens = _tokenize(output.evaluable_text)
    groups = [
        gold.must_contain_concepts,
        gold.must_mention_facts,
        gold.must_actions,
        gold.must_acknowledge_missing,
    ]
    scores: list[float] = []
    for expected_items in groups:
        if expected_items:
            scores.append(sum(
                1 for item in expected_items
                if _tokens_overlap(_tokenize(item), response_tokens) >= 0.6
            ) / len(expected_items))
    if gold.expects_clarification:
        scores.append(1.0 if detect_clarification(output) else 0.0)
    return sum(scores) / len(scores) if scores else (1.0 if output.parse_status == "ok" else 0.0)


# ── clarification detection ──────────────────────────────────────────

CLARIFICATION_PATTERNS = [
    r"\b(necesit|falt|requier|podrias?\s+(?:proporcionar|indicar|decir|aclarar))\b",
    r"\b(con\s+que\s+datos\b)",
    r"\b(podrias?\s+especificar)\b",
    r"\b(es\s+necesario\s+saber)\b",
    r"\b(que\s+(?:variedad|tipo|superficie|epoca|zona|cultivo))\b",
    r"\b(indicar\w*\s+(?:que|cual|como))\b",
    r"\b(sobre\s+la\s+superficie)\b",
]


def detect_clarification(output: NormalizedOutput) -> bool:
    """Detecta si la respuesta pide información adicional al usuario.

    Returns:
        True si la respuesta contiene patrones de solicitud de información.
    """
    text = _normalize(output.visible_text)
    return any(re.search(p, text) for p in CLARIFICATION_PATTERNS)


# ── Métricas de ejecución ───────────────────────────────────────────


def compute_execution_metrics(
    case: CaseSpec,
    output: NormalizedOutput,
    *,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    model_calls: int = 0,
    token_prompt_total: int = 0,
    token_completion_total: int = 0,
    agents_invoked: list[str] | None = None,
    agents_ok: int = 0,
    agents_error: int = 0,
    route_observed: list[str] | None = None,
) -> ExecutionMetrics:
    """Calcula todas las métricas deterministas para una ejecución."""
    forbidden_rate = compute_forbidden_claim_rate(case, output)
    overclaim_count = compute_overclaim_rate(output)
    actionability_structured = compute_actionability_structured(output)
    actionability_visible = compute_actionability_visible(output)
    actionability = (
        actionability_structured if output.next_actions else actionability_visible
    )
    clarification = detect_clarification(output)
    completeness = compute_answer_completeness(case, output)

    return ExecutionMetrics(
        success=output.parse_status == "ok",
        latency_ms=latency_ms,
        estimated_cost_usd=cost_usd,
        model_calls=model_calls,
        token_prompt_total=token_prompt_total,
        token_completion_total=token_completion_total,
        forbidden_claim_rate=forbidden_rate,
        overclaim_count=overclaim_count,
        actionability=actionability,
        actionability_structured=actionability_structured,
        actionability_visible=actionability_visible,
        answer_completeness=completeness,
        clarification_detected=clarification,
        agents_invoked=agents_invoked or [],
        agents_ok=agents_ok,
        agents_error=agents_error,
        route_observed=route_observed or [],
        execution_status=("ok" if output.parse_status == "ok" else "partial" if output.parse_status == "partial" else "failed"),
    )


# ── Routing assertions ───────────────────────────────────────────────


def compute_routing_score(
    case: CaseSpec,
    agents_invoked: list[str],
) -> float:
    """Evalúa si el routing observado coincide con el esperado.

    Returns:
        Score [0,1] donde 1 = routing perfecto, 0 = sin coincidencia.
    """
    optional = set(case.optional_route)
    expected = [agent for agent in case.expected_route if agent not in optional]
    if not expected:
        return 1.0

    invoked_set = set(agents_invoked)
    expected_set = set(expected)
    accepted_set = expected_set | optional

    if not invoked_set:
        return 0.0

    hits = invoked_set & expected_set
    recall = len(hits) / len(expected_set)
    precision = len(invoked_set & accepted_set) / len(invoked_set) if invoked_set else 0.0
    return recall * precision


def compute_routing_metrics(case: CaseSpec, agents_invoked: list[str]) -> tuple[float, float, float]:
    """Return precision, recall and order score for the observed route."""
    optional = set(case.optional_route)
    expected = [agent for agent in case.expected_route if agent not in optional]
    if not expected:
        return 1.0, 1.0, 1.0
    expected_set = set(expected)
    accepted_set = expected_set | optional
    observed = list(dict.fromkeys(agents_invoked))
    observed_set = set(observed)
    hits = expected_set & observed_set
    precision = len(observed_set & accepted_set) / len(observed_set) if observed_set else 0.0
    recall = len(hits) / len(expected_set)
    matched_pairs = sum(
        1 for left, right in zip(expected, expected[1:])
        if left in observed_set and right in observed_set and observed.index(left) < observed.index(right)
    )
    order = matched_pairs / max(1, len(expected) - 1)
    return precision, recall, order


def check_routing_assertion(
    case: CaseSpec,
    agents_invoked: list[str],
) -> bool:
    """Verifica si se cumple la aserción de routing del caso.

    Returns:
        True si se cumple la aserción, False si no.
    """
    assertion = case.routing_assertion
    if not assertion:
        return True

    assertion_lower = assertion.lower()
    agents_lower = [a.lower() for a in agents_invoked]

    known_agents = {"stac", "rs_analyst", "legal", "document_analyst", "spreadsheet_analyst", "vision_ocr", "free", "writer"}
    required = [agent for agent in known_agents if agent in assertion_lower]
    if "both" in assertion_lower or (" and " in assertion_lower and required):
        return bool(required) and all(agent in agents_lower for agent in required)
    if "either" in assertion_lower or " or " in assertion_lower:
        return bool(required) and any(agent in agents_lower for agent in required)
    if "must be invoked" in assertion_lower:
        return bool(required) and all(agent in agents_lower for agent in required)

    if "no specialized agents" in assertion_lower or "must not" in assertion_lower:
        specialized = {"stac", "rs_analyst", "legal", "document_analyst", "spreadsheet_analyst", "vision_ocr"}
        invoked_specialized = set(agents_lower) & specialized
        return len(invoked_specialized) == 0

    if "context" in assertion_lower or "follow-up" in assertion_lower or "turns" in assertion_lower:
        return False
    return True
