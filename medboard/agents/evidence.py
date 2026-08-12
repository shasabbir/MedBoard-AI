"""Evidence-retrieval agent answering explicit clinical questions with RAG."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentOutput,
    AgentStatus,
    EvidenceRetrievalAnalysis,
    MessageType,
)
from medboard.providers import StructuredModelProvider
from medboard.rag.store import KnowledgeStore


class EvidenceRetrievalAgent(BaseAgent):
    name = "evidence_retrieval"

    def __init__(
        self, provider: StructuredModelProvider, store: KnowledgeStore, top_k: int = 3
    ) -> None:
        super().__init__(provider)
        self.store = store
        self.top_k = top_k

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        questions = state["evidence_questions"]
        results = [
            result
            for question in questions
            for result in self.store.search(
                question.question,
                question_id=question.question_id,
                top_k=self.top_k,
            )
        ]
        prompt = (
            "Interpret the retrieved source chunks for the explicit clinical questions. "
            "Retain source metadata and do not invent evidence."
        )
        provider_result = self.provider.generate(
            agent=self.name,
            prompt=prompt,
            response_model=EvidenceRetrievalAnalysis,
            demo_factory=lambda: EvidenceRetrievalAnalysis(
                results=results,
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary=(
                        f"Retrieved {len(results)} source-attributed chunks for "
                        f"{len(questions)} questions."
                    ),
                    warnings=(
                        ["No knowledge chunks were available for retrieval."]
                        if not results
                        else []
                    ),
                ),
            ),
        )
        return {
            "retrieved_evidence": provider_result.output.results,
            "evidence_retrieval_analysis": provider_result.output,
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=(MessageType.RESPONSE if results else MessageType.WARNING),
                    content=provider_result.output.output.summary,
                    retrieval_ids=[item.retrieval_id for item in results],
                )
            ],
            "token_usage": [provider_result.usage],
        }
