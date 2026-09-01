from __future__ import annotations

import argparse

from backend.case_store import case_store
from backend.continuity_migration import migrate_legacy_memory
from backend.memory_store import memory_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa continuidad legacy al almacén de seguimientos")
    parser.add_argument("user_id", help="Identificador legacy de la explotación")
    args = parser.parse_args()
    result = migrate_legacy_memory(
        args.user_id,
        memory_store=memory_store,
        case_store=case_store,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
