import os
import subprocess
import sys
from pathlib import Path


def test_ingest_legalize_cli_smoke_with_local_fixture(tmp_path: Path):
    repo = tmp_path / "legalize-es" / "es"
    repo.mkdir(parents=True)
    (repo / "BOE-A-demo.md").write_text(
        """---
titulo: "Ley de agricultura de prueba"
identificador: "BOE-A-DEMO"
pais: "es"
jurisdiccion: "es"
estado: "vigente"
ultima_actualizacion: "2025-01-01"
fuente: "https://www.boe.es/demo"
---
Artículo 1. Agricultura

Texto sobre agricultura, ayudas PAC y trazabilidad.
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "legal.db"
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "VECTOR_BACKEND": "sqlite",
        "DISABLE_EXTERNALS": "true",
        "QDRANT_COLLECTION": "legalize_test_chunks",
    }

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_legalize.py",
            "--repos",
            "es",
            "--data-dir",
            str(tmp_path),
            "--local-only",
            "--no-embeddings",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "chunks_insertados=1" in result.stdout
    assert db_path.exists()


def test_ingest_legalize_cli_dry_run_cost_does_not_insert(tmp_path: Path):
    repo = tmp_path / "legalize-es" / "es"
    repo.mkdir(parents=True)
    (repo / "BOE-A-demo.md").write_text(
        """---
titulo: "Reglamento de producción ecológica"
identificador: "UE-DEMO"
pais: "es"
jurisdiccion: "es"
estado: "vigente"
ultima_actualizacion: "2025-01-01"
fuente: "https://www.boe.es/demo"
---
Artículo 1. Producción ecológica

Texto sobre producción ecológica, trazabilidad y etiquetado.
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "legal.db"
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "VECTOR_BACKEND": "sqlite",
        "DISABLE_EXTERNALS": "true",
        "QDRANT_COLLECTION": "legalize_test_chunks",
    }

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_legalize.py",
            "--repos",
            "es",
            "--data-dir",
            str(tmp_path),
            "--local-only",
            "--dry-run-cost",
            "--profile",
            "demo",
            "--status",
            "vigente",
            "--max-chunks",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "coste_estimado_embeddings" in result.stdout
    assert "dry-run" in result.stdout
    assert "chunks_insertados" not in result.stdout


def test_ingest_legalize_cli_skips_existing_chunks(tmp_path: Path):
    repo = tmp_path / "legalize-es" / "es"
    repo.mkdir(parents=True)
    (repo / "BOE-A-demo.md").write_text(
        """---
titulo: "Ley de agricultura de prueba"
identificador: "BOE-A-DEMO"
pais: "es"
jurisdiccion: "es"
estado: "vigente"
ultima_actualizacion: "2025-01-01"
fuente: "https://www.boe.es/demo"
---
Artículo 1. Agricultura

Texto sobre agricultura, ayudas PAC y trazabilidad.
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "legal.db"
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "VECTOR_BACKEND": "sqlite",
        "DISABLE_EXTERNALS": "true",
        "QDRANT_COLLECTION": "legalize_test_chunks",
    }
    command = [
        sys.executable,
        "scripts/ingest_legalize.py",
        "--repos",
        "es",
        "--data-dir",
        str(tmp_path),
        "--local-only",
        "--no-embeddings",
    ]

    first = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "chunks_insertados=1" in first.stdout
    assert "No hay chunks nuevos" in second.stdout
