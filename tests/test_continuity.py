from backend.continuity import has_continuity_signal, should_create_case, resolve_case
from libs.schemas import FinalAnswer


def test_has_continuity_signal_detects_case_language_and_attachments():
    assert has_continuity_signal("Compara la evolución de la parcela norte", attachment_count=0)
    assert has_continuity_signal("¿Qué reviso hoy?", attachment_count=1)
    assert not has_continuity_signal("¿Qué es el mildiu?", attachment_count=0)


def test_resolve_case_prefers_explicit_id():
    result = resolve_case(
        [{"case_id": "case-1", "status": "active", "title": "Riego norte"}],
        "Compara la evolución",
        explicit_case_id="case-1",
    )

    assert result.case_id == "case-1"
    assert result.reason == "explicit"


def test_should_create_case_only_for_caseworthy_turns():
    assert should_create_case("¿Qué es el mildiu?") is False
    assert should_create_case("Tengo una duda sobre la parcela norte", attachment_count=1)
    assert should_create_case(
        "¿Qué reviso?", case_state={"open_tasks": [{"title": "Medir humedad"}]}
    )


def test_final_answer_exposes_compact_continuity_summary():
    answer = FinalAnswer()

    assert answer.continuity.case_id is None
    assert answer.continuity.status == "none"


def test_resolve_case_does_not_choose_between_equal_active_candidates():
    result = resolve_case(
        [
            {"case_id": "case-1", "status": "active", "title": "Vigor parcela norte"},
            {"case_id": "case-2", "status": "active", "title": "Riego parcela norte"},
            {"case_id": "case-3", "status": "closed", "title": "Parcela norte antigua"},
        ],
        "Revisar la parcela norte",
    )

    assert result.reason == "ambiguous"
    assert result.case_id is None
    assert result.candidates == ["case-2", "case-1"]


def test_closed_linked_case_is_not_reused_automatically():
    result = resolve_case(
        [{"case_id": "case-1", "status": "closed", "title": "Riego norte"}],
        "Continúa con el riego de la parcela norte",
        linked_case_id="case-1",
    )

    assert result.case_id is None
