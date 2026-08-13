"""SQLite-backed LangGraph checkpoint lifecycle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


class WorkflowCheckpoint:
    """Own the SQLite connection used for resumable LangGraph execution."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.saver = SqliteSaver(self.connection)

    def close(self) -> None:
        self.connection.close()

    def delete_run(self, run_id: str) -> None:
        """Delete every checkpoint and pending write for one workflow run."""
        self.saver.delete_thread(run_id)

    def __enter__(self) -> WorkflowCheckpoint:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
