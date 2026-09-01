from pathlib import Path

from evaluation.baselines import _build_attachments
from evaluation.loaders import load_cases


def test_declared_attachment_fixtures_resolve_to_real_files():
    cases = load_cases("evaluation/cases/seed")
    attachment_cases = [case for case in cases if case.attachments]

    assert {case.case_id for case in attachment_cases} == {
        "att_001_leaf_disease",
        "att_002_compliance_doc",
        "att_003_yield_spreadsheet",
        "att_004_mixed_docs",
        "rt_003_document_route",
    }

    for case in attachment_cases:
        for attachment in _build_attachments(case):
            assert attachment.storage_path, case.case_id
            assert Path(attachment.storage_path).is_file(), case.case_id
