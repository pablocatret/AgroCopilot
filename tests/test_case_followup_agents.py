from agents.case_manager import CaseManagerAgent
from libs.schemas import (
    CaseStateDraft,
    CaseState,
    CaseTask,
    ImageInsight,
    ImageInsights,
    RSAgentOutput,
    RemoteSensingChange,
    RemoteSensingFocus,
)


def _rs_output() -> RSAgentOutput:
    return RSAgentOutput(
        agent="rs_analyst",
        summary="Serie temporal disponible con cambio reciente.",
        data=ImageInsights(
            overview="Serie temporal disponible con cambio reciente.",
            insights=[
                ImageInsight(
                    item_id="scene-new", summary="Menor vigor en borde oeste.", confidence=0.7
                )
            ],
            temporal_changes=[
                RemoteSensingChange(
                    from_item_id="scene-old",
                    to_item_id="scene-new",
                    label="Descenso de vigor en borde oeste",
                    detail="Se observa una pérdida relativa de vigor en la escena reciente.",
                    confidence=0.75,
                    metric="NDVI",
                    delta_mean=-0.12,
                    severity="media",
                    reliable=True,
                )
            ],
            focus_areas=[
                RemoteSensingFocus(
                    title="Comprobar borde oeste",
                    detail="La señal coincide con la observación reciente de campo.",
                    parcel="Parcela Norte",
                    priority="alta",
                )
            ],
        ),
    )


def test_case_manager_fallback_turns_rs_changes_into_tasks():
    agent = CaseManagerAgent()
    result = agent._fallback(
        type(
            "AgentInputLike",
            (),
            {
                "context": {
                    "rs_analyst": _rs_output(),
                    "legal": None,
                },
            },
        )()
    )

    assert result.open_tasks
    assert any("field-check" in task.title.lower() or "confirm" in task.title.lower() for task in result.open_tasks)
    assert any(task.source == "remote_sensing" for task in result.open_tasks)
    assert any("field photo" in item.lower() or "confirm" in item.lower() for item in result.recommended_next_input)
    assert result.evidence_ledger.items
    assert any(item.source == "remote_sensing" for item in result.evidence_ledger.items)
    assert result.evidence_ledger.modalities
    assert result.evidence_ledger.modalities[0].metadata["items"] >= 1
    assert not any("faltan adjuntos" in item.lower() for item in result.blocked_by)


def test_case_manager_merges_llm_state_over_deterministic_draft_without_losing_ledger():
    agent = CaseManagerAgent()
    draft = agent._deterministic_case_state_draft(
        type(
            "AgentInputLike",
            (),
            {
                "context": {
                    "rs_analyst": _rs_output(),
                    "legal": None,
                    "document_analyst": None,
                    "spreadsheet_analyst": None,
                    "vision_ocr": None,
                    "user_memory": "",
                },
            },
        )()
    )

    merged = agent._merge_case_state_with_draft(
        draft,
        CaseState(
            case_summary="Resumen refinado por LLM",
            open_tasks=[],
            blocked_by=[],
            recommended_next_input=["Confirma la observacion de campo mas reciente."],
        ),
        ledger=draft.evidence_ledger,
    )

    assert merged.case_summary == "Resumen refinado por LLM"
    assert merged.open_tasks
    assert merged.evidence_ledger.items


def test_case_manager_reconciles_tasks_and_inputs_instead_of_replacing_them():
    agent = CaseManagerAgent()
    draft = CaseState(
        case_summary="Resumen base suficientemente descriptivo del caso actual.",
        open_tasks=[
            CaseTask(
                title="Contrastar adjuntos y evidencias documentales",
                priority="alta",
                status="abierta",
                rationale="Hace falta revisar los documentos ya aportados.",
                source="document",
            )
        ],
        blocked_by=["No hay evidencia normativa suficiente para cerrar la decision."],
        recommended_next_input=["Aporta el justificante actualizado de la parcela."],
        evidence_ledger=CaseState().evidence_ledger,
    )
    llm_state = CaseState(
        case_summary="Resumen refinado por LLM con continuidad operativa suficiente.",
        open_tasks=[
            CaseTask(
                title="Contrastar adjuntos y evidencias documentales",
                priority="media",
                status="bloqueada",
                rationale="El documento principal sigue ambiguo y conviene validarlo con detalle.",
                source="document",
            ),
            CaseTask(
                title="Validar cambio detectado: descenso de vigor",
                priority="alta",
                status="abierta",
                rationale="La señal remota reciente justifica comprobación en campo.",
                source="remote_sensing",
            ),
        ],
        blocked_by=["Confirmar si el expediente aplica a la campaña vigente."],
        recommended_next_input=["Adjunta una foto reciente del borde oeste."],
        evidence_ledger=CaseState().evidence_ledger,
    )

    merged = agent._merge_case_state_with_draft(
        draft,
        llm_state,
        ledger=CaseState().evidence_ledger,
    )

    assert merged.case_summary == "Resumen refinado por LLM con continuidad operativa suficiente."
    assert len(merged.open_tasks) == 2
    assert any(task.status == "bloqueada" for task in merged.open_tasks)
    assert "Aporta el justificante actualizado de la parcela." in merged.recommended_next_input
    assert "Adjunta una foto reciente del borde oeste." in merged.recommended_next_input
    assert "No hay evidencia normativa suficiente para cerrar la decision." in merged.blocked_by
    assert "Confirmar si el expediente aplica a la campaña vigente." in merged.blocked_by


def test_case_manager_deterministic_draft_can_be_serialized_as_typed_contract():
    agent = CaseManagerAgent()
    draft = agent._deterministic_case_state_draft(
        type(
            "AgentInputLike",
            (),
            {
                "context": {
                    "rs_analyst": _rs_output(),
                    "legal": None,
                    "document_analyst": None,
                    "spreadsheet_analyst": None,
                    "vision_ocr": None,
                    "user_memory": "",
                },
            },
        )()
    )

    payload = CaseStateDraft.model_validate(draft.model_dump(exclude={"evidence_ledger"}))
    assert payload.open_tasks
    assert payload.recommended_next_input

