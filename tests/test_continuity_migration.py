from pathlib import Path

from backend.case_store import CaseStore
from backend.continuity_migration import migrate_legacy_memory
from backend.memory_store import UserMemoryStore
from libs.schemas import CaseState, CaseTask, FieldObservation


def test_legacy_memory_migration_is_idempotent(tmp_path: Path):
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    cases = CaseStore(tmp_path / "cases.db")
    memory.replace_sections(
        "finca-demo",
        {
            "profile": "Nombre o alias: Finca demo\nZona geografica: Jaén",
            "farm_context": "Cultivos principales: olivar\nInfraestructura relevante: riego",
        },
    )
    memory.save_case_state(
        "finca-demo",
        CaseState(
            case_summary="Seguimiento de vigor en parcela norte",
            open_tasks=[CaseTask(title="Revisar humedad")],
        ),
    )
    memory.append_observation(
        "finca-demo",
        FieldObservation(
            date="2026-07-11",
            parcel="Parcela Norte",
            note="Menor vigor",
        ),
    )

    first = migrate_legacy_memory("finca-demo", memory_store=memory, case_store=cases)
    second = migrate_legacy_memory("finca-demo", memory_store=memory, case_store=cases)

    assert first["migrated"] is True
    assert second["migrated"] is False
    assert len(cases.list_cases(workspace_id="finca-demo")) == 1
    case_id = first["case_id"]
    assert cases.get_workspace_context("finca-demo")["crops"] == "olivar"
    assert cases.list_observations(case_id)[0]["parcel"] == "Parcela Norte"
    cases.close()
