from backend.schemas import AgentResult, CommandDecision


class AgentPolicy:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

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
