"""Progresso incremental resumível para baterias de query longas.

Generaliza o padrão usado em experimentos_graphrag.py (JSON com uma entrada
por prontuário, salvo a cada query) para qualquer experimento: se a bateria
for interrompida (rate limit, timeout, queda de conexão), rodar de novo só
processa o que falta. `run_experiments.py` não tinha isso — usava-o teria
evitado recomeçar do zero em interrupções.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class ProgressTracker:
    """Registra resultados por chave (ex.: prontuario_id) num JSON em disco."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._dados: dict = self._carrega()

    def _carrega(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def salva(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._dados, f, ensure_ascii=False, indent=2)

    def pendentes(self, chaves: Iterable[str]) -> list[str]:
        return [k for k in chaves if k not in self._dados]

    def registra(self, chave: str, registro: dict, salvar: bool = True) -> None:
        self._dados[chave] = registro
        if salvar:
            self.salva()

    def __contains__(self, chave: str) -> bool:
        return chave in self._dados

    def __getitem__(self, chave: str) -> dict:
        return self._dados[chave]

    def __len__(self) -> int:
        return len(self._dados)

    def valores(self):
        return self._dados.values()
