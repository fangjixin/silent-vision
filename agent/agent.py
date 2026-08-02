from backend.schemas import AgentResult, CommandDecision, SemanticResult


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

    def decide_command(self, decision: CommandDecision) -> AgentResult:
        if not decision.accepted:
            return AgentResult(
                action="reject",
                language="unknown",
                text="",
                arguments={"intent": decision.intent, "reason": decision.reason},
                requiresConfirmation=False,
            )
        if not decision.executable:
            return AgentResult(
                action="ignore",
                language="unknown",
                text="",
                arguments={"intent": decision.intent, "reason": decision.reason},
                requiresConfirmation=False,
            )
        return AgentResult(
            action="execute",
            language="unknown",
            text=decision.intent,
            arguments={"intent": decision.intent, "confidence": decision.confidence},
            requiresConfirmation=False,
        )
