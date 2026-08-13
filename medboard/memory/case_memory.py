"""Durable audit repository separate from LangGraph workflow checkpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from medboard.graph.state import MedicalCaseSnapshot
from medboard.memory.database import Database
from medboard.models import HumanReviewCommand


class CaseMemoryRepository:
    """Persist cases, snapshots, messages, and human decisions transactionally."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def save_run(self, snapshot: MedicalCaseSnapshot, status: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO cases(case_id, synthetic, case_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET case_json = excluded.case_json
                """,
                (
                    snapshot.case_input.case_id,
                    int(snapshot.case_input.synthetic),
                    snapshot.case_input.model_dump_json(),
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO runs(run_id, case_id, status, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    snapshot_json = excluded.snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot.run_id,
                    snapshot.case_input.case_id,
                    status,
                    snapshot.model_dump_json(),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO agent_messages(
                    message_id, run_id, sender, recipient, message_type,
                    message_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                [
                    (
                        message.message_id,
                        snapshot.run_id,
                        message.sender,
                        message.recipient,
                        message.message_type.value,
                        message.model_dump_json(),
                        message.timestamp.isoformat(),
                    )
                    for message in snapshot.agent_messages
                ],
            )
            connection.executemany(
                """
                INSERT INTO trace_events(
                    trace_id, run_id, event_type, agent, event_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO NOTHING
                """,
                [
                    (
                        event.trace_id,
                        snapshot.run_id,
                        event.event_type.value,
                        event.agent,
                        event.model_dump_json(),
                        event.timestamp.isoformat(),
                    )
                    for event in snapshot.execution_trace
                ],
            )

    def save_human_feedback(self, run_id: str, command: HumanReviewCommand) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO human_feedback(
                    run_id, action, reviewer, feedback, command_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    command.action.value,
                    command.reviewer,
                    command.feedback,
                    command.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def load_run(self, run_id: str) -> MedicalCaseSnapshot | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return MedicalCaseSnapshot.model_validate_json(row[0]) if row else None

    def run_status(self, run_id: str) -> str | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def feedback_count(self, run_id: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM human_feedback WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row[0])

    def message_count(self, run_id: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM agent_messages WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row[0])

    def trace_count(self, run_id: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM trace_events WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row[0])

    def list_runs(self) -> list[dict[str, str]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, case_id, status, created_at, updated_at
                FROM runs ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_case(self, case_id: str) -> bool:
        """Delete a case and its cascading audit rows for privacy controls."""
        with self.database.connect() as connection:
            cursor = connection.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
        return cursor.rowcount > 0

    def run_ids_for_case(self, case_id: str) -> list[str]:
        """Return all workflow run IDs owned by a case before cascading deletion."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE case_id = ? ORDER BY run_id", (case_id,)
            ).fetchall()
        return [str(row[0]) for row in rows]
