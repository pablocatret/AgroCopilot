from pathlib import Path


def test_case_store_tracks_revisions_events_and_projection(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(
        workspace_id="finca-demo",
        title="Estrés hídrico en parcela norte",
        objective="Decidir la revisión de riego",
    )

    assertion = store.create_assertion(
        workspace_id="finca-demo",
        case_id=case["case_id"],
        key="cultivo",
        value="olivar",
        scope="case",
        provenance="user_statement",
        status="confirmed",
        actor_type="user",
    )
    replacement = store.correct_assertion(
        assertion["assertion_id"],
        value="olivar de regadío",
        actor_type="user",
    )
    task = store.create_task(
        case_id=case["case_id"],
        title="Comprobar humedad en el sector norte",
        status="proposed",
        actor_type="assistant",
    )
    store.update_task(task["task_id"], status="open", actor_type="user")

    detail = store.get_case(case["case_id"], workspace_id="finca-demo")
    assert detail["case"]["status"] == "active"
    assert detail["projection"]["confirmed_facts"][0]["value_text"] == "olivar de regadío"
    assert detail["projection"]["active_tasks"][0]["status"] == "open"
    assert replacement["supersedes_assertion_id"] == assertion["assertion_id"]
    assert any(event["event_type"] == "assertion_corrected" for event in detail["events"])
    store.close()


def test_linking_same_conversation_is_idempotent(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="finca-demo", title="Seguimiento")

    store.link_conversation(case_id=case["case_id"], conversation_id="conv-1")
    store.link_conversation(case_id=case["case_id"], conversation_id="conv-1")

    events = store.list_events(case["case_id"])
    assert [event["event_type"] for event in events].count("conversation_linked") == 1
    store.close()


def test_reading_case_does_not_rewrite_projection_or_case_timestamp(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="finca-demo", title="Seguimiento")
    before = store.get_case(case["case_id"], workspace_id="finca-demo")
    before_projection = before["projection"]["updated_at"]
    before_case_updated = before["case"]["updated_at"]

    writes_before_read = store.conn.total_changes
    after = store.get_case(case["case_id"], workspace_id="finca-demo")

    assert after["projection"]["updated_at"] == before_projection
    assert after["case"]["updated_at"] == before_case_updated
    assert store.conn.total_changes == writes_before_read
    store.close()


def test_case_mutations_require_matching_workspace(tmp_path: Path):
    import pytest

    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="workspace-a", title="Privado")
    task = store.create_task(case_id=case["case_id"], workspace_id="workspace-a", title="Revisar")
    assertion = store.create_assertion(
        workspace_id="workspace-a",
        case_id=case["case_id"],
        key="cultivo",
        value="olivar",
        scope="case",
        provenance="user_statement",
        status="confirmed",
    )

    with pytest.raises(KeyError):
        store.update_task(task["task_id"], workspace_id="workspace-b", status="done")
    with pytest.raises(KeyError):
        store.set_assertion_status(assertion["assertion_id"], workspace_id="workspace-b", status="retracted")
    with pytest.raises(KeyError):
        store.create_observation(
            case_id=case["case_id"],
            workspace_id="workspace-b",
            date="2026-07-16",
            parcel="Norte",
            note="No autorizado",
        )
    store.close()


def test_deleted_case_cannot_be_reopened(tmp_path: Path):
    import pytest

    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="workspace-a", title="Borrado")
    store.set_case_status(case["case_id"], workspace_id="workspace-a", status="deleted")

    with pytest.raises(ValueError):
        store.set_case_status(case["case_id"], workspace_id="workspace-a", status="active")
    store.close()


def test_case_events_have_stable_sequence(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="workspace-a", title="Orden")
    store.append_event(case["case_id"], event_type="first", actor_type="test")
    store.append_event(case["case_id"], event_type="second", actor_type="test")

    events = store.list_events(case["case_id"])
    assert [event["sequence_no"] for event in events] == sorted(
        (event["sequence_no"] for event in events), reverse=True
    )
    assert [event["event_type"] for event in reversed(events)][-2:] == ["first", "second"]
    store.close()


def test_rebinding_conversation_moves_relationship_once(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    first = store.create_case(workspace_id="workspace-a", title="Primero")
    second = store.create_case(workspace_id="workspace-a", title="Segundo")
    store.link_conversation(case_id=first["case_id"], conversation_id="conv-1")
    store.rebind_conversation(case_id=second["case_id"], conversation_id="conv-1")
    store.rebind_conversation(case_id=second["case_id"], conversation_id="conv-1")

    old_link = store.conn.execute(
        "SELECT 1 FROM case_conversations WHERE case_id=? AND conversation_id=?",
        (first["case_id"], "conv-1"),
    ).fetchone()
    new_link = store.conn.execute(
        "SELECT 1 FROM case_conversations WHERE case_id=? AND conversation_id=?",
        (second["case_id"], "conv-1"),
    ).fetchone()
    assert old_link is None
    assert new_link is not None
    assert [event["event_type"] for event in store.list_events(second["case_id"])].count("conversation_linked") == 1
    store.close()


def test_case_store_builds_bounded_explainable_context(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="finca-demo", title="Seguimiento olivar")
    store.create_assertion(
        workspace_id="finca-demo",
        case_id=case["case_id"],
        key="parcela",
        value="Parcela Norte",
        scope="case",
        provenance="user_statement",
        status="confirmed",
        actor_type="user",
    )
    store.create_assertion(
        workspace_id="finca-demo",
        case_id=None,
        key="preferencia",
        value="Priorizar acciones de bajo coste",
        scope="global",
        provenance="user_statement",
        status="confirmed",
        actor_type="user",
    )

    context = store.build_context(
        case_id=case["case_id"],
        workspace_id="finca-demo",
        query="¿Qué reviso en la parcela norte?",
    )

    assert "Parcela Norte" in context["text"]
    assert context["items"]
    assert all(item["reason"] for item in context["items"])
    store.close()


def test_case_store_persists_observations_inside_case(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    case = store.create_case(workspace_id="finca-demo", title="Riego norte")

    created = store.create_observation(
        case_id=case["case_id"],
        date="2026-07-11",
        parcel="Parcela Norte",
        campaign="2026",
        note="Borde oeste con menor vigor.",
        severity="media",
    )

    observations = store.list_observations(case["case_id"])
    assert created["case_id"] == case["case_id"]
    assert observations[0]["note"] == "Borde oeste con menor vigor."
    assert observations[0]["campaign"] == "2026"
    store.close()


def test_case_store_persists_compact_workspace_context(tmp_path: Path):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    saved = store.save_workspace_context(
        "finca-demo",
        {"name": "Finca demo", "zone": "Jaén", "crops": "olivar"},
    )

    assert saved["workspace_id"] == "finca-demo"
    assert saved["crops"] == "olivar"
    assert store.get_workspace_context("finca-demo")["zone"] == "Jaén"
    store.close()
