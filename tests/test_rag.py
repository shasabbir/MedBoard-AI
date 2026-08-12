"""Tests for knowledge ingestion, vector retrieval, and graph integration."""

from pathlib import Path

from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import MedicalCaseInput
from medboard.providers import DemoModelProvider
from medboard.rag.ingestion import chunk_document, load_document
from medboard.rag.store import KnowledgeStore


def test_markdown_ingestion_preserves_source_metadata() -> None:
    document = load_document(Path("data/knowledge/stroke.md"))
    chunks = chunk_document(document, chunk_size=300, overlap=40)

    assert document.organization == "World Health Organization"
    assert document.source_url.startswith("https://www.who.int/")
    assert chunks
    assert all(chunk.document == document.title for chunk in chunks)
    assert {chunk.section for chunk in chunks} >= {"Warning features", "Time and assessment"}


def test_chroma_retrieval_returns_relevant_source_attributed_chunk(
    tmp_path: Path,
) -> None:
    store = KnowledgeStore(tmp_path / "chroma")
    ingested = store.ingest_directory(Path("data/knowledge"))

    results = store.search(
        "sudden unilateral weakness stroke urgent assessment",
        question_id="Q-TEST-001",
        top_k=2,
    )

    assert ingested >= 8
    assert store.count == ingested
    assert results
    assert results[0].document == "Stroke fact sheet educational summary"
    assert results[0].source == "World Health Organization"
    assert results[0].source_url == "https://www.who.int/news-room/fact-sheets/detail/stroke"
    assert results[0].retrieved_text
    assert 0 <= results[0].similarity_score <= 1


def test_rag_agent_answers_differential_and_selected_specialist_questions(
    tmp_path: Path,
) -> None:
    case = MedicalCaseInput(
        case_id="CASE-RAG-001",
        chief_complaint="Sudden confusion and unilateral weakness",
        symptoms=["confusion", "unilateral weakness", "headache"],
    )
    store = KnowledgeStore(tmp_path / "chroma")
    store.ingest_directory(Path("data/knowledge"))
    workflow = build_collaboration_workflow(DemoModelProvider(), store)

    result = workflow.invoke(create_initial_state(case, run_id="RUN-RAG"))
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == ["neurology"]
    assert any(question.asked_by == "differential" for question in snapshot.evidence_questions)
    assert any(question.asked_by == "neurology" for question in snapshot.evidence_questions)
    assert snapshot.retrieved_evidence
    assert snapshot.evidence_retrieval_analysis is not None
    assert snapshot.evidence_retrieval_analysis.results == snapshot.retrieved_evidence
    question_ids = {question.question_id for question in snapshot.evidence_questions}
    assert all(item.question_id in question_ids for item in snapshot.retrieved_evidence)
    assert all(
        item.document and item.source and item.section and item.retrieved_text
        for item in snapshot.retrieved_evidence
    )
    assert any(
        message.sender == "evidence_retrieval" and message.retrieval_ids
        for message in snapshot.agent_messages
    )


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "chroma")
    first_count = store.ingest_directory(Path("data/knowledge"))
    second_count = store.ingest_directory(Path("data/knowledge"))

    assert second_count == first_count
    assert store.count == first_count
