"""Cached Streamlit runtime resources."""

from __future__ import annotations

from dataclasses import dataclass

from medboard.config import Settings, get_settings
from medboard.graph.workflow import build_reviewable_workflow
from medboard.memory import CaseMemoryRepository, Database, WorkflowCheckpoint
from medboard.observability import setup_logging
from medboard.providers import build_model_provider
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
    settings.ensure_runtime_directories()
    setup_logging(settings)
    knowledge_store = KnowledgeStore(settings.chroma_persist_directory)
    knowledge_store.ingest_directory(settings.knowledge_directory)
    checkpoint = WorkflowCheckpoint(settings.workflow_checkpoint_path)
    case_memory = CaseMemoryRepository(Database(settings.database_path))
    graph = build_reviewable_workflow(
        build_model_provider(settings),
        knowledge_store,
        max_revisions=settings.max_revisions,
        max_agent_retries=settings.max_agent_retries,
        rag_top_k=settings.rag_top_k,
        checkpointer=checkpoint.saver,
    )
    return AppRuntime(
        settings=settings,
        checkpoint=checkpoint,
        case_memory=case_memory,
        knowledge_store=knowledge_store,
        service=WorkflowService(graph, case_memory, checkpoint),
    )
