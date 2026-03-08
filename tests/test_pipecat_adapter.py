import pytest
from covenant.adapters.pipecat import PipecatAdapter
from covenant.exceptions import CovenantTimeoutError, CovenantRunError
from pipecat.frames.frames import (
    TranscriptionFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame,
    LLMTextFrame, FunctionCallInProgressFrame, FunctionCallResultFrame, ErrorFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from covenant.adapters.pipecat_pipeline import CovenantTestTransport

class MockLLMProcessor(FrameProcessor):
    def __init__(self, simulate_tool=False, simulate_confirm=False, timeout=False, error=False):
        super().__init__()
        self.simulate_tool = simulate_tool
        self.simulate_confirm = simulate_confirm
        self.timeout = timeout
        self.error = error

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            if self.error:
                await self.push_frame(ErrorFrame(error="LLM failed"), FrameDirection.DOWNSTREAM)
                return
            if self.simulate_tool:
                await self.push_frame(FunctionCallInProgressFrame(
                    tool_call_id="call1", function_name="get_weather", arguments="{}"
                ), FrameDirection.DOWNSTREAM)
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="call1", function_name="get_weather", arguments="{}", result="Sunny"
                ), FrameDirection.DOWNSTREAM)
            if self.simulate_confirm:
                await self.push_frame(FunctionCallInProgressFrame(
                    tool_call_id="call2", function_name="confirm_action", arguments="{}"
                ), FrameDirection.DOWNSTREAM)
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="call2", function_name="confirm_action", arguments="{}", result="Yes"
                ), FrameDirection.DOWNSTREAM)
            
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMTextFrame("test response"), FrameDirection.DOWNSTREAM)
            if not self.timeout:
                await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

def build_mock_pipeline(**kwargs):
    transport = CovenantTestTransport()
    return Pipeline([
        transport.input(),
        MockLLMProcessor(**kwargs),
        transport.output()
    ])

@pytest.mark.asyncio
async def test_basic_transcription_to_response():
    pipeline = build_mock_pipeline()
    adapter = PipecatAdapter(pipeline)
    trace = await adapter.run("Hello", 2)
    assert trace.final_response == "test response"
    assert len(trace.tool_calls) == 0

@pytest.mark.asyncio
async def test_tool_call_captured():
    pipeline = build_mock_pipeline(simulate_tool=True)
    adapter = PipecatAdapter(pipeline)
    trace = await adapter.run("Hello", 2)
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "get_weather"
    assert trace.tool_calls[0].result == "Sunny"

@pytest.mark.asyncio
async def test_confirmation_detection():
    pipeline = build_mock_pipeline(simulate_confirm=True)
    adapter = PipecatAdapter(pipeline)
    trace = await adapter.run("Hello", 2)
    assert trace.asked_for_confirmation is True

@pytest.mark.asyncio
async def test_timeout_raises_covenant_error():
    pipeline = build_mock_pipeline(timeout=True)
    adapter = PipecatAdapter(pipeline)
    with pytest.raises(CovenantTimeoutError):
        await adapter.run("Hello", 1)

@pytest.mark.asyncio
async def test_can_handle_detection():
    pipeline = build_mock_pipeline()
    assert PipecatAdapter.can_handle(pipeline) is True
    
    async def dummy_factory(transport):
        return pipeline
    assert PipecatAdapter.can_handle(dummy_factory) is True

    class DummyLangchain:
        def invoke(self): pass
    assert PipecatAdapter.can_handle(DummyLangchain()) is False

@pytest.mark.asyncio
async def test_error_frame_raises_covenant_error():
    pipeline = build_mock_pipeline(error=True)
    adapter = PipecatAdapter(pipeline)
    with pytest.raises(CovenantRunError) as exc:
        await adapter.run("Hello", 2)
    assert "LLM failed" in str(exc.value)

@pytest.mark.asyncio
async def test_pattern_c_agent():
    pipeline = build_mock_pipeline()
    class PatternCAgent:
        def __init__(self):
            self.pipeline = pipeline
        async def run(self, transport):
            pass
            
    adapter = PipecatAdapter(PatternCAgent())
    trace = await adapter.run("hello", 2)
    assert trace.final_response == "test response"

@pytest.mark.asyncio
async def test_pattern_c_multi_turn():
    pipeline = build_mock_pipeline()
    class PatternCAgent:
        def __init__(self):
            self.pipeline = pipeline
        async def run(self, transport):
            pass
            
    adapter = PipecatAdapter(PatternCAgent())
    traces = await adapter.run_multi_turn(["hello"], 2)
    assert len(traces) == 1
