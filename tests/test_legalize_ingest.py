from pathlib import Path

from libs.rag.legalize import (
    build_chunks,
    is_agro_relevant,
    parse_markdown_document,
    split_legal_sections,
)


def test_parse_legalize_frontmatter_and_build_payload(tmp_path: Path):
    law = tmp_path / "BOE-A-demo.md"
    law.write_text(
        """---
titulo: "Ley de agricultura sostenible"
identificador: "BOE-A-DEMO"
pais: "es"
jurisdiccion: "es"
rango: "ley"
fecha_publicacion: "2024-01-01"
ultima_actualizacion: "2025-01-01"
estado: "vigente"
fuente: "https://www.boe.es/demo"
---
# Ley de agricultura sostenible

Artículo 1. Objeto

Esta ley regula la agricultura sostenible y el uso eficiente del agua.
""",
        encoding="utf-8",
    )

    document = parse_markdown_document(law, repo="legalize-es", repo_root=tmp_path)
    chunks = build_chunks(document)

    assert document.metadata["titulo"] == "Ley de agricultura sostenible"
    assert len(chunks) == 1
    payload = chunks[0].payload
    assert payload["corpus"] == "legalize"
    assert payload["estado"] == "vigente"
    assert payload["official_source"] == "https://www.boe.es/demo"
    assert payload["article"] == "Artículo 1. Objeto"


def test_agro_filter_excludes_irrelevant_law(tmp_path: Path):
    law = tmp_path / "BOE-A-civil.md"
    law.write_text(
        """---
titulo: "Ley procesal civil"
identificador: "BOE-A-CIVIL"
pais: "es"
estado: "vigente"
fuente: "https://www.boe.es/civil"
---
Artículo 1. Juzgados y procedimiento civil.
""",
        encoding="utf-8",
    )

    document = parse_markdown_document(law, repo="legalize-es", repo_root=tmp_path)

    assert not is_agro_relevant(document)
    assert build_chunks(document) == []
    assert build_chunks(document, include_all=True)


def test_demo_profile_is_narrower_than_agro(tmp_path: Path):
    law = tmp_path / "BOE-A-rural.md"
    law.write_text(
        """---
titulo: "Ley de desarrollo rural"
identificador: "BOE-A-RURAL"
pais: "es"
estado: "vigente"
fuente: "https://www.boe.es/rural"
---
Artículo 1. Desarrollo rural y planificación territorial.
""",
        encoding="utf-8",
    )

    document = parse_markdown_document(law, repo="legalize-es", repo_root=tmp_path)

    assert build_chunks(document, profile="agro")
    assert build_chunks(document, profile="demo") == []


def test_split_legal_sections_and_stable_ids(tmp_path: Path):
    law = tmp_path / "UE-demo.md"
    law.write_text(
        """---
titulo: "Reglamento de producción ecológica"
identificador: "UE-DEMO"
pais: "eu"
estado: "vigente"
ultima_actualizacion: "2025-02-03"
fuente: "https://eur-lex.europa.eu/demo"
---
Artículo 1. Producción ecológica

Normas de producción ecológica para agricultores.

Artículo 2. Etiquetado

Normas de etiquetado ecológico y trazabilidad.
""",
        encoding="utf-8",
    )

    sections = split_legal_sections(law.read_text(encoding="utf-8"))
    document = parse_markdown_document(law, repo="legalize-eu", repo_root=tmp_path)
    first = build_chunks(document)
    second = build_chunks(document)

    assert len(sections) >= 2
    assert [chunk.point_id for chunk in first] == [chunk.point_id for chunk in second]
    assert {chunk.payload["article"] for chunk in first} == {
        "Artículo 1. Producción ecológica",
        "Artículo 2. Etiquetado",
    }
