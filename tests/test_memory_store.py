import json
from pathlib import Path

from backend.memory_store import UserMemoryStore
from libs.schemas import CaseState, CaseTask, FieldObservation, RemoteSensingMemoryArtifact


def test_memory_store_creates_markdown_files(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    memory = store.load("demo-user")

    user_dir = tmp_path / "demo-user" / "default"
    assert user_dir.exists()
    assert (user_dir / "profile.md").exists()
    assert memory.user_id == "demo-user"
    assert "profile" in memory.sections


def test_memory_store_appends_decision_log(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_decision_log(
        "demo-user",
        query="Revisar transicion ecologica en olivar",
        decision_mode="decision",
        executive_summary="Hay pasos previos antes de certificar.",
        next_actions=["Verificar certificacion organica"],
        missing_information=["Falta documentacion de parcela"],
    )

    log_text = (tmp_path / "demo-user" / "default" / "decision_log.md").read_text(encoding="utf-8")
    assert "Revisar transicion ecologica en olivar" in log_text
    assert "Verificar certificacion organica" in log_text
    assert "Falta documentacion de parcela" in log_text


def test_memory_store_replaces_editable_sections(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    memory = store.replace_sections(
        "demo-user",
        {
            "profile": "Cultivo principal: almendro",
            "preferences": "# Preferencias operativas\n\n- Objetivo principal: reducir riesgo\n",
        },
    )

    assert "Cultivo principal: almendro" in memory.sections["profile"]
    assert "reducir riesgo" in memory.sections["preferences"]


def test_memory_store_deletes_user_directory(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)
    store.load("demo-user")

    store.delete_user("demo-user")

    assert not (tmp_path / "demo-user").exists()


def test_memory_store_persists_case_state(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.save_case_state(
        "demo-user",
        CaseState(
            case_summary="Expediente de ayuda en preparacion.",
            open_tasks=[
                CaseTask(
                    title="Subir certificado",
                    priority="alta",
                    rationale="Falta soporte documental.",
                )
            ],
            blocked_by=["No consta certificado ecologico vigente."],
            recommended_next_input=["Adjunta el certificado o indica si no existe."],
        ),
    )

    loaded = store.load_case_state("demo-user")
    assert loaded.case_summary == "Expediente de ayuda en preparacion."
    assert loaded.open_tasks[0].title == "Subir certificado"
    assert loaded.blocked_by[0] == "No consta certificado ecologico vigente."
    assert (tmp_path / "demo-user" / "default" / "current_case.json").exists()


def test_memory_store_sanitizes_case_state_before_persisting(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.save_case_state(
        "demo-user",
        CaseState(
            case_summary="  Expediente   con   espacios   ",
            open_tasks=[
                CaseTask(title="  Subir certificado  ", priority="alta", rationale="  Falta soporte  "),
                CaseTask(title="Subir certificado", priority="media", rationale="Duplicada"),
            ],
            blocked_by=["  Bloqueo principal  "],
            recommended_next_input=["  Adjunta el certificado  "],
        ),
    )

    loaded = store.load_case_state("demo-user")
    assert loaded.case_summary == "Expediente con espacios"
    assert len(loaded.open_tasks) == 1
    assert loaded.open_tasks[0].title == "Subir certificado"
    assert loaded.blocked_by == ["Bloqueo principal"]
    assert loaded.recommended_next_input == ["Adjunta el certificado"]


def test_memory_store_appends_case_history(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_case_history(
        "demo-user",
        title="Preparar expediente PAC",
        decision_mode="compliance",
        summary="Expediente en preparacion.",
        next_actions=["Subir certificado", "Confirmar recintos"],
        blocked_by=["No consta certificado"],
    )

    history = store.load_case_history("demo-user")
    assert history
    assert history[0].title == "Preparar expediente PAC"
    assert history[0].decision_mode == "compliance"
    assert (tmp_path / "demo-user" / "default" / "case_history.json").exists()


def test_memory_store_sanitizes_case_history_structured_entries(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_case_history(
        "demo-user",
        title="  Preparar   expediente   PAC  ",
        decision_mode=" compliance ",
        summary="  Expediente   en   preparacion.  ",
        next_actions=["  Subir certificado  ", "Subir certificado"],
        blocked_by=["  No consta certificado  "],
    )

    history = store.load_case_history("demo-user")
    assert history[0].title == "Preparar expediente PAC"
    assert history[0].decision_mode == "compliance"
    assert history[0].next_actions[0] == "Subir certificado"
    assert history[0].blocked_by == ["No consta certificado"]


def test_memory_store_preserves_structured_case_history_chronology_on_append(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_case_history(
        "demo-user",
        title="Caso 1",
        decision_mode="case",
        summary="Primero",
        next_actions=[],
        blocked_by=[],
    )
    store.append_case_history(
        "demo-user",
        title="Caso 2",
        decision_mode="case",
        summary="Segundo",
        next_actions=[],
        blocked_by=[],
    )

    structured = json.loads((tmp_path / "demo-user" / "default" / "case_history.json").read_text(encoding="utf-8"))
    history = store.load_case_history("demo-user")

    assert [item["title"] for item in structured] == ["Caso 1", "Caso 2"]
    assert [item.title for item in history] == ["Caso 2", "Caso 1"]


def test_memory_store_keeps_valid_case_history_items_when_one_structured_item_is_corrupt(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)
    store.ensure_user_files("demo-user")
    structured_path = tmp_path / "demo-user" / "default" / "case_history.json"
    structured_path.write_text(
        json.dumps(
            [
                {"title": "Valido", "decision_mode": "case", "summary": "ok", "next_actions": [], "blocked_by": []},
                {"title": "Invalido", "summary": "falta decision_mode y next_actions no tipado", "next_actions": "oops"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history = store.load_case_history("demo-user")
    assert len(history) == 1
    assert history[0].title == "Valido"


def test_memory_store_logs_case_history_structured_fallback(tmp_path: Path, caplog):
    store = UserMemoryStore(base_dir=tmp_path)
    store.ensure_user_files("demo-user")
    structured_path = tmp_path / "demo-user" / "default" / "case_history.json"
    structured_path.write_text("{not-json}", encoding="utf-8")

    with caplog.at_level("WARNING"):
        history = store.load_case_history("demo-user")

    assert history == []
    assert any("memory.case_history_json_invalid" in message for message in caplog.messages)


def test_memory_store_appends_observations(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_observation(
        "demo-user",
        FieldObservation(
            date="2026-04-02",
            parcel="Parcela Norte",
            campaign="2026",
            note="Se observan sintomas de estres hidrico en el borde oeste.",
            severity="media",
        ),
    )

    observations = store.load_observations("demo-user")
    assert observations
    assert observations[0].parcel == "Parcela Norte"
    assert "estres hidrico" in observations[0].note
    assert (tmp_path / "demo-user" / "default" / "field_observations.json").exists()


def test_memory_store_sanitizes_observation_structured_entries(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_observation(
        "demo-user",
        FieldObservation(
            date=" 2026-04-02 ",
            parcel="  Parcela   Norte  ",
            campaign=" 2026 ",
            note="  Se observan sintomas de estres hidrico.  ",
            severity="media",
        ),
    )

    observations = store.load_observations("demo-user")
    assert observations[0].date == "2026-04-02"
    assert observations[0].parcel == "Parcela Norte"
    assert observations[0].campaign == "2026"
    assert observations[0].note == "Se observan sintomas de estres hidrico."


def test_memory_store_preserves_structured_observation_chronology_on_append(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.append_observation(
        "demo-user",
        FieldObservation(
            date="2026-04-02",
            parcel="Parcela Norte",
            campaign="2026",
            note="Primera observacion",
            severity="media",
        ),
    )
    store.append_observation(
        "demo-user",
        FieldObservation(
            date="2026-04-03",
            parcel="Parcela Sur",
            campaign="2026",
            note="Segunda observacion",
            severity="alta",
        ),
    )

    structured = json.loads(
        (tmp_path / "demo-user" / "default" / "field_observations.json").read_text(encoding="utf-8")
    )
    observations = store.load_observations("demo-user")

    assert [item["parcel"] for item in structured] == ["Parcela Norte", "Parcela Sur"]
    assert [item.parcel for item in observations] == ["Parcela Sur", "Parcela Norte"]


def test_memory_store_keeps_valid_observations_when_one_structured_item_is_corrupt(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)
    store.ensure_user_files("demo-user")
    structured_path = tmp_path / "demo-user" / "default" / "field_observations.json"
    structured_path.write_text(
        json.dumps(
            [
                {
                    "date": "2026-04-02",
                    "parcel": "Parcela Norte",
                    "campaign": "2026",
                    "note": "Observacion valida",
                    "severity": "media",
                },
                {
                    "date": "2026-04-03",
                    "parcel": "Parcela Sur",
                    "note": ["nota no valida"],
                    "severity": "media",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    observations = store.load_observations("demo-user")
    assert len(observations) == 1
    assert observations[0].parcel == "Parcela Norte"


def test_memory_store_logs_observation_structured_fallback(tmp_path: Path, caplog):
    store = UserMemoryStore(base_dir=tmp_path)
    store.ensure_user_files("demo-user")
    structured_path = tmp_path / "demo-user" / "default" / "field_observations.json"
    structured_path.write_text("{not-json}", encoding="utf-8")

    with caplog.at_level("WARNING"):
        observations = store.load_observations("demo-user")

    assert observations == []
    assert any(
        "memory.field_observations_json_invalid" in message for message in caplog.messages
    )


def test_memory_store_persists_remote_sensing_artifacts(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    artifact = RemoteSensingMemoryArtifact(
        generated_at="2026-07-01T10:00:00Z",
        query="Analiza la parcela norte con satelite",
        query_intent="monitoring",
        decision_mode="case",
        parcel="Parcela Norte",
        latest_scene_date="2026-06-29",
        stac_item_ids=["scene-1", "scene-2"],
        scene_count=2,
        summary="Analisis de 2 escenas satelitales.",
        change_highlights=["Descenso de vigor en borde oeste."],
    )

    store.save_remote_sensing_artifact("demo-user", artifact)
    loaded = store.load_remote_sensing_artifacts("demo-user")

    assert loaded
    assert loaded[0].parcel == "Parcela Norte"
    assert loaded[0].stac_item_ids == ["scene-1", "scene-2"]


def test_memory_store_switches_current_memory_with_consistent_name(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    created = store.create_memory("demo-user", "Seguimiento olivar")
    current = store.set_current_memory("demo-user", created.memory_id)
    raw_current = json.loads((tmp_path / "demo-user" / "current.json").read_text(encoding="utf-8"))

    assert current.name == "Seguimiento olivar"
    assert raw_current["memory_id"] == created.memory_id
    assert raw_current["name"] == "Seguimiento olivar"


def test_memory_store_preserves_remote_sensing_artifact_chronology_on_append(tmp_path: Path):
    store = UserMemoryStore(base_dir=tmp_path)

    store.save_remote_sensing_artifact(
        "demo-user",
        RemoteSensingMemoryArtifact(
            generated_at="2026-07-01T10:00:00Z",
            latest_scene_date="2026-06-30",
            query="Caso 1",
            evidence_level="analyzed_partial",
            summary="Primero",
        ),
    )
    store.save_remote_sensing_artifact(
        "demo-user",
        RemoteSensingMemoryArtifact(
            generated_at="2026-07-02T10:00:00Z",
            latest_scene_date="2026-07-01",
            query="Caso 2",
            evidence_level="analyzed_temporal",
            summary="Segundo",
        ),
    )

    structured = json.loads(
        (tmp_path / "demo-user" / "default" / "remote_sensing_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    loaded = store.load_remote_sensing_artifacts("demo-user")

    assert [item["query"] for item in structured] == ["Caso 1", "Caso 2"]
    assert [item.query for item in loaded] == ["Caso 2", "Caso 1"]
