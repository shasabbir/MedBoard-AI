"""Read-only operational views for knowledge, logs, and runtime settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from medboard.config import Settings
from medboard.rag.ingestion import load_directory
from medboard.rag.store import KnowledgeStore


def render_knowledge_base(settings: Settings, store: KnowledgeStore) -> None:
    st.header("Knowledge base")
    st.caption(
        "Source-attributed public educational material indexed in the local vector store."
    )
    documents = load_directory(settings.knowledge_directory)
    metrics = st.columns(3)
    metrics[0].metric("Documents", len(documents))
    metrics[1].metric("Indexed chunks", store.count)
    metrics[2].metric("Storage", "Local Chroma")
    for document in documents:
        with st.expander(f"{document.title} · {document.organization}"):
            st.write(document.text[:900] + ("…" if len(document.text) > 900 else ""))
            st.caption(
                f"{document.document_type} · {document.year} · "
                f"{Path(document.source_path).name}"
            )
            st.link_button("Open public source", document.source_url)


def render_system_logs(settings: Settings) -> None:
    st.header("System logs")
    st.caption("Recent structured application events. Secrets and case contents are excluded.")
    log_path = settings.log_directory / "medboard.jsonl"
    if not log_path.exists():
        st.info("No application log events have been recorded yet.")
        return
    records = _read_json_lines(log_path, limit=200)
    if not records:
        st.info("The log file exists but contains no readable events.")
        return
    levels = sorted({str(record.get("level", "UNKNOWN")) for record in records})
    selected = st.multiselect("Levels", levels, default=levels, key="log_levels")
    visible = [record for record in reversed(records) if record.get("level") in selected]
    st.dataframe(visible, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(visible)} of the latest {len(records)} events.")


def render_settings(settings: Settings) -> None:
    st.header("Runtime settings")
    st.info(
        "Settings are read from environment variables at startup. Restart the app after "
        "changing `.env`; credentials are never displayed here."
    )
    st.subheader("Model and workflow")
    st.table(
        [
            {"Setting": "Mode", "Value": settings.mode_label},
            {"Setting": "Provider", "Value": settings.llm_provider.value},
            {"Setting": "Model", "Value": _model_label(settings)},
            {"Setting": "Maximum critic revisions", "Value": settings.max_revisions},
            {"Setting": "Maximum agent retries", "Value": settings.max_agent_retries},
            {"Setting": "Agent timeout (seconds)", "Value": settings.agent_timeout_seconds},
            {"Setting": "RAG top-k", "Value": settings.rag_top_k},
            {
                "Setting": "Input cost / 1M tokens",
                "Value": settings.llm_input_cost_per_million,
            },
            {
                "Setting": "Output cost / 1M tokens",
                "Value": settings.llm_output_cost_per_million,
            },
            {"Setting": "Log level", "Value": settings.log_level},
        ]
    )
    st.subheader("Local storage")
    st.code(
        "\n".join(
            [
                f"Case history: {settings.database_path}",
                f"Checkpoints: {settings.workflow_checkpoint_path}",
                f"Knowledge index: {settings.chroma_persist_directory}",
                f"Logs: {settings.log_directory}",
            ]
        )
    )


def _read_json_lines(path: Path, *, limit: int) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _model_label(settings: Settings) -> str:
    if settings.demo_mode:
        return "deterministic-v1"
    return settings.openai_model or settings.gemini_model or "Not configured"
