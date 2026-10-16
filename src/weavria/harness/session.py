"""Connect task understanding to the persistent SessionEngine preparation seam."""

from weavria.core import Message, MessageRole
from weavria.sessions import SessionTaskContext, SessionTaskPreparation

from .understanding import TaskUnderstandingEngine


class TaskAwareSessionPreparer:
    """Build one initial Task Frame and least-capability view per accepted input."""

    version = "task-aware-session/1"

    def __init__(self, understanding: TaskUnderstandingEngine) -> None:
        self._understanding = understanding

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation:
        history: list[Message] = []
        for turn in context.history:
            history.append(Message(role=MessageRole.USER, content=turn.user))
            history.extend(
                Message(role=MessageRole.USER, content=update.directive.text)
                for update in turn.updates
            )
            if turn.assistant is not None:
                history.append(Message(role=MessageRole.ASSISTANT, content=turn.assistant))
        preparation = await self._understanding.prepare(
            input_id=context.input_id,
            task=context.task,
            history=tuple(history[-64:]),
            tools=context.tools,
        )
        return SessionTaskPreparation(
            preparation_version=self.version,
            visible_tool_names=preparation.record.selected_capabilities,
            observation=preparation.observation,
        )
