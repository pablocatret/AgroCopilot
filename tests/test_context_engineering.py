from libs.context_engineering import summarize_temporal_focus
from libs.schemas import FieldObservation


def test_summarize_temporal_focus_uses_latest_observation_date():
    focus = summarize_temporal_focus(
        [
            FieldObservation(date="2026-03-10", parcel="A", note="Nota", severity="media"),
            FieldObservation(date="2026-04-01", parcel="B", note="Nota", severity="alta"),
        ]
    )
    assert "2026-04-01" in focus
    assert "comparación" in focus.lower()
