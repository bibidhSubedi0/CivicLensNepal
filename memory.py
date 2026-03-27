from collections import deque


class ConversationMemory:
    """Rolling buffer of recent turns. That's it."""

    def __init__(self, max_turns: int = 6):
        self._turns = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self._turns.append({"role": role, "content": content})

    def get_context_string(self, max_chars: int = 1200) -> str:
        """Return recent turns as a plain string, truncated from the oldest end."""
        lines = []
        for t in self._turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {t['content']}")
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def clear(self):
        self._turns.clear()

    def __len__(self):
        return len(self._turns)


def expand_query(query: str, memory: ConversationMemory) -> str:
    """
    Prepend recent conversation context to the query before embedding.
    e5 models do better when the query string carries full context — 
    "passage: ..." is for indexing, "query: ..." is for retrieval,
    so we only expand the query side here.
    """
    context = memory.get_context_string()
    if not context:
        return query
    return f"{context}\nUser: {query}"