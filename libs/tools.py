from pydantic import BaseModel
from typing import Any, Dict


class Tool:
    name: str
    input_schema: BaseModel | None = None

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
