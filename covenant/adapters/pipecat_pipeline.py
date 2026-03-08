"""
Pipecat testing pipeline infrastructure for CovenantAI.
"""
import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipecat.frames.frames import (  # type: ignore
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TranscriptionFrame,
    ErrorFrame,
    EndFrame,
    TTSAudioRawFrame,
    InterruptionFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection  # type: ignore

class CovenantFrameObserver(FrameProcessor):  # type: ignore
    """
    Passively observes Pipecat frames flowing downstream to aggregate LLM calls and text.
    """
    def __init__(self) -> None:
        super().__init__()
        self._function_calls: List[Dict[str, Any]] = []
        self._response_chunks: List[str] = []
        self._response_complete: bool = False
        self._response_started: bool = False
        self._agent_interrupted_user: bool = False
        self._errors: List[str] = []
        self._completion_event = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallInProgressFrame):
                self._function_calls.append({
                    "name": frame.function_name,
                    "tool_call_id": frame.tool_call_id,
                    "arguments": frame.arguments,
                    "result": None,
                    "timestamp_ms": time.perf_counter() * 1000
                })
            elif isinstance(frame, FunctionCallResultFrame):
                for fc in self._function_calls:
                    if fc["tool_call_id"] == frame.tool_call_id:
                        fc["result"] = frame.result
                        break
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._response_started = True
            elif isinstance(frame, LLMTextFrame):
                if self._response_started:
                    self._response_chunks.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._response_complete = True
                self._completion_event.set()
            elif isinstance(frame, InterruptionFrame):
                if self._response_started and not self._response_complete:
                    self._agent_interrupted_user = True
            elif isinstance(frame, ErrorFrame):
                self._errors.append(str(frame.error))

        await self.push_frame(frame, direction)

    def get_trace_data(self) -> Dict[str, Any]:
        return {
            "function_calls": self._function_calls,
            "final_response": "".join(self._response_chunks),
            "errors": self._errors,
            "asked_for_confirmation": self._detect_confirmation(),
            "interrupted_user": self._agent_interrupted_user
        }

    def _detect_confirmation(self) -> bool:
        keywords = {"confirm", "approve", "human_input", "ask_user", "request_approval"}
        for fc in self._function_calls:
            name = str(fc.get("name", "")).lower()
            if any(k in name for k in keywords):
                return True
        return False

    async def wait_for_completion(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

class CovenantInputProcessor(FrameProcessor):  # type: ignore
    def __init__(self, input_queue: "asyncio.Queue[Any]") -> None:
        super().__init__()
        self._input_queue = input_queue
        self._drain_task: "Optional[asyncio.Task[Any]]" = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)
        
        from pipecat.frames.frames import StartFrame
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, StartFrame):
                self._drain_task = asyncio.create_task(self._drain_queue())
            elif isinstance(frame, EndFrame):
                if self._drain_task and not self._drain_task.done():
                    self._drain_task.cancel()

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain_queue())

    async def _drain_queue(self) -> None:
        try:
            while True:
                frame = await self._input_queue.get()
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

class CovenantOutputProcessor(FrameProcessor):  # type: ignore
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction == FrameDirection.DOWNSTREAM and hasattr(frame, "audio"):
            # The prompt requested "discards TTSAudioRawFrame silently". Checking exactly for that class
            pass
        if isinstance(frame, TTSAudioRawFrame):
            return
            
        await self.push_frame(frame, FrameDirection.UPSTREAM)

class CovenantTestTransport:
    def __init__(self) -> None:
        self._input_queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._started = False

    def input(self) -> CovenantInputProcessor:
        return CovenantInputProcessor(self._input_queue)

    def output(self) -> CovenantOutputProcessor:
        return CovenantOutputProcessor()

    async def inject_transcription(self, text: str, user_id: str = "test_user") -> None:
        frame = TranscriptionFrame(text=text, user_id=user_id, timestamp=datetime.now().isoformat())
        await self._input_queue.put(frame)
