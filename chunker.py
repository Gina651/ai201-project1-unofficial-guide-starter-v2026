"""
Stage 2 of the pipeline: splitting documents into chunks.

`split_documents` is my chunker for the advice_threads corpus: one chunk per
reply, with the thread title prepended. `fallback_split` is the starter's
original fixed-window chunker, kept for comparison.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


# One reply starts at a line like: --- reply 2 (27 votes) ---
REPLY_MARKER = re.compile(r"^--- reply \d+ \(\d+ votes?\) ---\s*$", re.MULTILINE)
MIN_CHARS = 30  # anything shorter than this is a fragment, not a thought


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    One chunk per reply, with the THREAD title prepended to each.

    Replies in these threads are short, self-contained, and often disagree
    with each other, so a reply is the natural unit. The title goes on every
    chunk because a reply like "Counterpoint, I sold mine" means nothing
    without the question it answers.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        parts = REPLY_MARKER.split(doc.text.strip())
        title = parts[0].strip()      # the "THREAD: ..." line
        replies = parts[1:]

        # No reply markers found: keep the whole document as one chunk
        # rather than silently losing it.
        if not replies:
            replies = [title]
            title = ""

        index = 0
        for reply in replies:
            body = reply.strip()
            if len(body) < MIN_CHARS:
                continue
            text = f"{title}\n{body}" if title else body
            chunks.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    index=index,
                    produced_by="chunker.py::split_documents",
                )
            )
            index += 1
    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))