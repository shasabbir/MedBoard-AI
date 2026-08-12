"""Cached Streamlit runtime resources."""

from __future__ import annotations

from dataclasses import dataclass

from medboard.config import Settings, get_settings
from medboard.graph.workflow import build_reviewable_workflow
from medboard.memory import CaseMemoryRepository, Database, WorkflowCheckpoint
from medboard.providers import DemoModelProvider
from medboard.rag.store import KnowledgeStore
from medboard.workflow_service import WorkflowService


@dataclass(slots=True)
class AppRuntime:
    settings: Settings
    checkpoint: WorkflowCheckpoint
    case_memory: CaseMemoryRepository
    knowledge_store: KnowledgeStore
    service: WorkflowService


def get_runtime() -> AppRuntime:
    """Build process-wide stores and the checkpointed graph once per server."""
    settings = get_settings()
    if not settings.demo_mode:
        raise RuntimeError("The Streamlit workflow currently supports DEMO_MODE=true")
    settings.ensure_runtime_directories()
    knowledge_store = KnowledgeStore(settings.chroma_persist_directory)
    knowledge_store.ingest_directory(settings.knowledge_directory)
    checkpoint = WorkflowCheckpoint(settings.workflow_checkpoint_path)
    case_memory = CaseMemoryRepository(Database(settings.database_path))
    graph = build_reviewable_workflow(
        DemoModelProvider(),
        knowledge_store,
        max_revisions=settings.max_revisions,
        checkpointer=checkpoint.saver,
    )
    return AppRuntime(
        settings=settings,
        checkpoint=checkpoint,
        case_memory=case_memory,
        knowledge_store=knowledge_store,
        service=WorkflowService(graph, case_memory),
    )
