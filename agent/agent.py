from backend.schemas import AgentResult, SemanticResult


class AgentPolicy:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def decide(self, result: SemanticResult) -> AgentResult:
        if result.language == "unknown" or result.confidence < self.threshold or not result.text.strip():
            return AgentResult(
                action="unknown",
                language="unknown",
                text="",
                arguments={},
                requiresConfirmation=False,
            )
        if "?" in result.text or "吗" in result.text:
            return AgentResult(
                action="confirm",
                language=result.language,
                text=result.text,
                arguments={},
                requiresConfirmation=True,
            )
        return AgentResult(
            action="respond",
            language=result.language,
            text=result.text,
            arguments={},
            requiresConfirmation=False,
        )
