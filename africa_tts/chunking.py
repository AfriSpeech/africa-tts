"""Paragraph and sentence preparation for Africa TTS.

Keeps natural sentence and clause intonation intact while ensuring long passages
(articles, paragraphs, and run-on sentences) are structured cleanly without token
overflow or skipped words.
"""

import re


def chunk_paragraphs(text: str, max_words_per_clause: int = 22) -> list[str]:
    """Split text into natural paragraphs, ensuring run-on sentences without punctuation
    have comma pauses inserted so the model's attention window is never exceeded.

    Leaves standard sentences and comma clauses completely intact to preserve
    melody, rhythm, and natural prosody.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Split on paragraph boundaries (two or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    out = []
    for p in paragraphs:
        # Split into sentences on . ! ? …
        sentences = [s.strip() for s in re.split(r"(?<=[.!?\u2026])\s+", " ".join(p.split())) if s.strip()]
        p_clean = []
        for s in sentences:
            # Check for overly long clauses without any punctuation
            clauses = [c.strip() for c in re.split(r"(?<=[,;:—–\(\)\[\]])\s+", s) if c.strip()]
            s_clauses = []
            for c in clauses:
                words = c.split()
                if len(words) > max_words_per_clause:
                    # Insert comma pause every max_words_per_clause words
                    for i in range(0, len(words), max_words_per_clause):
                        piece = " ".join(words[i : i + max_words_per_clause])
                        if i + max_words_per_clause < len(words):
                            piece += ","
                        s_clauses.append(piece)
                else:
                    s_clauses.append(c)
            s_clean = " ".join(s_clauses)
            if not s_clean.endswith((".", "!", "?", "…")):
                s_clean += "."
            p_clean.append(s_clean)
        out.append(" ".join(p_clean))

    return out
