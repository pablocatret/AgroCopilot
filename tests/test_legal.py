import pytest
from datetime import datetime, timezone

from agents import legal
from libs.rag import retriever
from libs.schemas import AgentInput


def test_build_checklist_marks_no_cumple_when_query_declares_missing_certificate():
    docs = [
        {
            "title": "GlobalG.A.P. IFA v6",
            "snippet": "Requisitos de certificación GlobalG.A.P. IFA v6 para frutas y hortalizas.",
            "url": "https://example.org/globalgap",
        }
    ]

    findings = legal._build_checklist("No tengo certificado GlobalG.A.P. para exportar", docs)

    globalgap = findings[1]
    assert globalgap.status == "no_cumple"
    assert globalgap.evidence


def test_build_checklist_marks_cumple_when_query_declares_certificate():
    docs = [
        {
            "title": "Reglamento (UE) 2018/848",
            "snippet": "Producción ecológica y etiquetado de productos ecológicos.",
            "url": "https://example.org/organic",
        }
    ]

    findings = legal._build_checklist("Ya tengo certificado orgánico conforme al 2018/848", docs)

    organic = findings[0]
    assert organic.status == "cumple"


def test_build_checklist_uses_legalize_metadata_for_dynamic_findings():
    docs = [
        {
            "title": "Ley de agricultura sostenible",
            "snippet": "Artículo 1. Agricultura sostenible y uso eficiente del agua.",
            "url": "https://www.boe.es/demo",
            "corpus": "legalize",
            "jurisdiccion": "es",
            "estado": "vigente",
            "updated_at": "2025-01-01",
            "article": "Artículo 1",
        }
    ]

    findings = legal._build_checklist(
        "Necesito revisar obligaciones de agricultura sostenible", docs
    )

    assert findings[0].requirement == "Ley de agricultura sostenible - Artículo 1"
    assert findings[0].jurisdiction == "es"
    assert findings[0].source_status == "vigente"
    assert findings[0].official_source == "https://www.boe.es/demo"


def test_build_checklist_marks_derogated_sources_as_limitation():
    docs = [
        {
            "title": "Norma antigua de ayudas agrarias",
            "snippet": "Ayudas agrarias",
            "url": "https://www.boe.es/old",
            "corpus": "legalize",
            "estado": "derogada",
        }
    ]

    findings = legal._build_checklist("Ayudas agrarias", docs)

    assert findings[0].status == "insuficiente"
    assert any("derogada" in item for item in findings[0].limitations)


def test_needs_external_verification_for_currentness_queries():
    docs = [
        {
            "title": "Reglamento de produccion ecologica",
            "source_status": "vigente",
            "official_source": "https://eur-lex.europa.eu/demo",
        }
    ]

    assert legal._needs_external_verification("Normativa vigente hoy sobre produccion ecologica", docs)


def test_needs_external_verification_for_current_year_queries_without_hardcoded_years():
    docs = [
        {
            "title": "Reglamento de produccion ecologica",
            "source_status": "vigente",
            "official_source": "https://eur-lex.europa.eu/demo",
        }
    ]
    current_year = datetime.now(timezone.utc).year

    assert legal._needs_external_verification(
        f"Normativa aplicable en {current_year} para produccion ecologica",
        docs,
    )


def test_does_not_require_external_verification_for_historical_year_query_with_stable_official_evidence():
    docs = [
        {
            "title": "Reglamento de produccion ecologica",
            "source_status": "vigente",
            "official_source": "https://eur-lex.europa.eu/demo",
        }
    ]
    previous_year = datetime.now(timezone.utc).year - 1

    assert not legal._needs_external_verification(
        f"Normativa aplicable en {previous_year} para produccion ecologica",
        docs,
    )


def test_does_not_require_external_verification_for_stable_official_local_evidence():
    docs = [
        {
            "title": "Reglamento de produccion ecologica",
            "source_status": "vigente",
            "official_source": "https://eur-lex.europa.eu/demo",
        }
    ]

    assert not legal._needs_external_verification("Produccion ecologica", docs)


def test_requires_external_verification_when_local_docs_lack_official_source():
    docs = [
        {
            "title": "Resumen divulgativo",
            "source_status": "vigente",
            "url": "",
        }
    ]

    assert legal._needs_external_verification("Produccion ecologica", docs)


def test_build_legal_dossier_separates_authoritative_and_supporting_references():
    dossier = retriever.build_legal_dossier(
        [
            {
                "title": "Reglamento UE",
                "official_source": "https://eur-lex.europa.eu/demo",
                "source_status": "in_force",
                "corpus": "legalize",
                "snippet": "Texto oficial",
            },
            {
                "title": "Resumen divulgativo",
                "url": "https://example.org/blog",
                "snippet": "Resumen no oficial",
            },
        ],
        verification_mode="local",
    )

    assert len(dossier.authoritative_references) == 1
    assert dossier.authoritative_references[0].title == "Reglamento UE"
    assert len(dossier.supporting_references) == 1
    assert dossier.supporting_references[0].title == "Resumen divulgativo"


def test_build_legal_dossier_does_not_treat_non_official_legalize_mirror_as_authoritative():
    dossier = retriever.build_legal_dossier(
        [
            {
                "title": "Mirror de reglamento",
                "official_source": "https://example.org/mirror/reglamento",
                "source_status": "in_force",
                "corpus": "legalize",
                "snippet": "Texto espejado",
            }
        ],
        verification_mode="local",
    )

    assert not dossier.authoritative_references
    assert dossier.supporting_references[0].title == "Mirror de reglamento"


@pytest.mark.asyncio
async def test_legal_agent_returns_structured_dossier(monkeypatch):
    docs = [
        {
            "title": "Reglamento UE",
            "official_source": "https://eur-lex.europa.eu/demo",
            "source_status": "in_force",
            "corpus": "legalize",
            "snippet": "Texto oficial",
            "jurisdiccion": "eu",
        },
        {
            "title": "Resumen divulgativo",
            "url": "https://example.org/blog",
            "snippet": "Resumen no oficial",
        },
    ]

    async def fake_retrieve_legal_dossier(query: str, k: int = 8):
        return docs, retriever.build_legal_dossier(docs, verification_mode="local")

    monkeypatch.setattr(legal, "retrieve_legal_dossier", fake_retrieve_legal_dossier)

    agent = legal.LegalAgent()
    async def fake_answer_with_research(*args, **kwargs):
        return "Respuesta legal", []

    monkeypatch.setattr(agent, "_answer_with_research", fake_answer_with_research)

    output = await agent.run(AgentInput(query="Normativa de produccion ecologica"))

    assert output.data.dossier.authoritative_references
    assert output.data.dossier.authoritative_references[0].title == "Reglamento UE"
    assert output.data.dossier.supporting_references[0].title == "Resumen divulgativo"


@pytest.mark.asyncio
async def test_legal_agent_keeps_authoritative_docs_empty_when_local_dossier_has_none(monkeypatch):
    docs = [
        {
            "title": "Mirror de reglamento",
            "official_source": "https://example.org/mirror/reglamento",
            "source_status": "in_force",
            "corpus": "legalize",
            "snippet": "Texto espejado",
        }
    ]

    async def fake_retrieve_legal_dossier(query: str, k: int = 8):
        return docs, retriever.build_legal_dossier(docs, verification_mode="local")

    captured = {}

    async def fake_answer_with_research(query, authoritative_docs, supporting_docs, *, use_web):
        captured["authoritative_docs"] = authoritative_docs
        captured["supporting_docs"] = supporting_docs
        return "Respuesta legal", []

    monkeypatch.setattr(legal, "retrieve_legal_dossier", fake_retrieve_legal_dossier)

    agent = legal.LegalAgent()
    monkeypatch.setattr(agent, "_answer_with_research", fake_answer_with_research)

    await agent.run(AgentInput(query="Normativa de produccion ecologica"))

    assert captured["authoritative_docs"] == []
    assert captured["supporting_docs"][0]["title"] == "Mirror de reglamento"


