"""Load, validate, and chunk Markdown, text, and PDF knowledge documents."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field
from medboard.models import ContractModel

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
REQUIRED_METADATA = {"title", "organization", "year", "source_url", "document_type"}


class KnowledgeDocument(ContractModel):
    title: str
    organization: str
    year: str
    source_url: str
    document_type: str
    text: str = Field(min_length=1)
    source_path: str


class KnowledgeChunk(ContractModel):
    chunk_id: str
    document: str
    organization: str
    year: str
    source_url: str
    document_type: str
    section: str
    text: str = Field(min_length=1)


def load_document(path: Path) -> KnowledgeDocument:
    """Load a supported document; text formats require explicit front matter."""
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported knowledge document type: {path.suffix}")
    if path.suffix.casefold() == ".pdf":
        return _load_pdf(path)

    raw = path.read_text(encoding="utf-8")
    metadata, text = _parse_front_matter(raw)
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"knowledge metadata missing: {sorted(missing)}")
    return KnowledgeDocument(
        **{key: metadata[key] for key in REQUIRED_METADATA},
        text=text,
        source_path=str(path),
    )


def load_directory(directory: Path) -> list[KnowledgeDocument]:
    """Load supported files in deterministic path order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"knowledge directory not found: {directory}")
    return [
        load_document(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    ]


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[KnowledgeChunk]:
    """Create heading-aware chunks with bounded character overlap."""
    if chunk_size < 100 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be >= 100 and overlap smaller than chunk_size")
    sections = _split_sections(document.text)
    chunks: list[KnowledgeChunk] = []
    for section, text in sections:
        start = 0
        part = 1
        while start < len(text):
            end = min(len(text), start + chunk_size)
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{_slug(document.title)}-{_slug(section)}-{part:03d}",
                        document=document.title,
                        organization=document.organization,
                        year=document.year,
                        source_url=document.source_url,
                        document_type=document.document_type,
                        section=section,
                        text=excerpt,
                    )
                )
            if end == len(text):
                break
            start = end - overlap
            part += 1
    return chunks


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("Markdown and text knowledge files require YAML-style front matter")
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and value.strip():
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, match.group(2).strip()


def _load_pdf(path: Path) -> KnowledgeDocument:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    metadata = reader.metadata or {}
    return KnowledgeDocument(
        title=str(metadata.get("/Title") or path.stem),
        organization=str(metadata.get("/Author") or "Unknown organization"),
        year=str(metadata.get("/CreationDate") or "Unknown year")[:4],
        source_url=str(metadata.get("/Subject") or path.resolve()),
        document_type="PDF",
        text=text,
        source_path=str(path),
    )


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if "\n".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line.lstrip("#").strip() or "Overview"
            buffer = []
        else:
            buffer.append(line)
    if "\n".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    return sections or [("Overview", text.strip())]


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:64] or "chunk"
