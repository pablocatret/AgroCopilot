from datetime import datetime, timezone

from backend.memory_reuse import resolve_remote_sensing_reuse
from libs.schemas import RemoteSensingMemoryArtifact


def _now() -> datetime:
    return datetime(2026, 7, 3, tzinfo=timezone.utc)


def test_remote_sensing_reuse_does_not_hit_for_retrieval_only_artifact():
    assessment = resolve_remote_sensing_reuse(
        query="Monitoriza la parcela norte",
        decision_mode="case",
        observations=[],
        artifacts=[
            RemoteSensingMemoryArtifact(
                generated_at="2026-07-02T10:00:00Z",
                latest_scene_date="2026-07-01",
                query="Monitoriza la parcela norte",
                query_intent="monitoring",
                evidence_level="retrieval_only",
                parcel="Parcela Norte",
                summary="2 escenas STAC recuperadas.",
            )
        ],
        now=_now(),
    )

    assert assessment.status == "miss"


def test_remote_sensing_reuse_uses_latest_scene_date_for_staleness():
    assessment = resolve_remote_sensing_reuse(
        query="Monitoriza la parcela norte",
        decision_mode="case",
        observations=[],
        artifacts=[
            RemoteSensingMemoryArtifact(
                generated_at="2026-07-02T10:00:00Z",
                latest_scene_date="2026-05-01",
                query="Monitoriza la parcela norte",
                query_intent="monitoring",
                evidence_level="analyzed_temporal",
                parcel="Parcela Norte",
                summary="Cambio temporal detectado.",
            )
        ],
        ttl_days=21,
        now=_now(),
    )

    assert assessment.status == "stale"
    assert "ultima escena" in assessment.reason.lower()


def test_remote_sensing_reuse_requires_strong_location_for_hit_when_query_is_ambiguous():
    assessment = resolve_remote_sensing_reuse(
        query="Monitoriza esto otra vez",
        decision_mode="case",
        observations=[
            {
                "date": "2026-07-01",
                "parcel": "Parcela Norte",
                "campaign": "2026",
                "note": "Seguimiento previo",
                "severity": "media",
            }
        ],
        artifacts=[
            RemoteSensingMemoryArtifact(
                generated_at="2026-07-02T10:00:00Z",
                latest_scene_date="2026-07-01",
                query="Monitoriza la parcela norte",
                query_intent="monitoring",
                evidence_level="analyzed_temporal",
                parcel="Parcela Norte",
                summary="Cambio temporal detectado.",
            )
        ],
        now=_now(),
    )

    assert assessment.status != "hit"
