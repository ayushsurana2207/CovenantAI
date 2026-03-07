import asyncio
import pytest
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TranscriptionFrame,
    ErrorFrame,
    EndFrame,
    StartFrame,
    TTSAudioRawFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from covenant.adapters.pipecat_pipeline import (
    CovenantFrameObserver,
    CovenantTestTransport,
    CovenantInputProcessor,
    CovenantOutputProcessor
)

@pytest.mark.asyncio
async def test_frame_observer_records_function_call_in_progress():
    observer = CovenantFrameObserver()
    frame = FunctionCallInProgressFrame(
        tool_call_id="call_123",
        function_name="get_weather",
        arguments='{"location": "Tokyo"}'
    )
    await observer.process_frame(frame, FrameDirection.DOWNSTREAM)
    
    trace_data = observer.get_trace_data()
    assert len(trace_data["function_calls"]) == 1
    call = trace_data["function_calls"][0]
    assert call["name"] == "get_weather"
    assert call["tool_call_id"] == "call_123"
    assert call["arguments"] == '{"location": "Tokyo"}'
    assert call["result"] is None

@pytest.mark.asyncio
async def test_frame_observer_records_function_call_result():
    observer = CovenantFrameObserver()
    in_progress = FunctionCallInProgressFrame(
        tool_call_id="call_123",
        function_name="get_weather",
        arguments='{"location": "Tokyo"}'
    )
    await observer.process_frame(in_progress, FrameDirection.DOWNSTREAM)
    
    result_frame = FunctionCallResultFrame(
        tool_call_id="call_123",
        function_name="get_weather",
        result="Sunny",
        arguments='{"location": "Tokyo"}'
    )
    await observer.process_frame(result_frame, FrameDirection.DOWNSTREAM)
    
    trace_data = observer.get_trace_data()
    call = trace_data["function_calls"][0]
    assert call["result"] == "Sunny"

@pytest.mark.asyncio
async def test_frame_observer_assembles_llm_text_chunks():
    observer = CovenantFrameObserver()
    await observer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await observer.process_frame(LLMTextFrame("Hello"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(LLMTextFrame(" "), FrameDirection.DOWNSTREAM)
    await observer.process_frame(LLMTextFrame("World"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    
    trace_data = observer.get_trace_data()
    assert trace_data["final_response"] == "Hello World"

@pytest.mark.asyncio
async def test_wait_for_completion_returns_true():
    observer = CovenantFrameObserver()
    
    async def push_end_frame():
        await asyncio.sleep(0.01)
        await observer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        
    task = asyncio.create_task(push_end_frame())
    result = await observer.wait_for_completion(timeout=1.0)
    assert result is True
    await task

@pytest.mark.asyncio
async def test_wait_for_completion_returns_false_on_timeout():
    observer = CovenantFrameObserver()
    result = await observer.wait_for_completion(timeout=0.01)
    assert result is False

@pytest.mark.asyncio
async def test_detect_confirmation_tool_calls():
    observer = CovenantFrameObserver()
    
    await observer.process_frame(FunctionCallInProgressFrame(
        tool_call_id="1",
        function_name="get_weather",
        arguments="{}"
    ), FrameDirection.DOWNSTREAM)
    
    assert observer.get_trace_data()["asked_for_confirmation"] is False
    
    await observer.process_frame(FunctionCallInProgressFrame(
        tool_call_id="2",
        function_name="ask_user_for_approval",
        arguments="{}"
    ), FrameDirection.DOWNSTREAM)
    
    assert observer.get_trace_data()["asked_for_confirmation"] is True

@pytest.mark.asyncio
async def test_input_processor_emits_transcription():
    transport = CovenantTestTransport()
    input_proc = transport.input()
    
    class DummyProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.caught_frames = []
            
        async def process_frame(self, frame, direction):
            self.caught_frames.append(frame)
            
    dummy = DummyProcessor()
    input_proc.link(dummy)
    
    await transport.inject_transcription("Hello test", "user1")
    
    await input_proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)
    
    await input_proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.01)
    
    assert any(isinstance(f, TranscriptionFrame) and f.text == "Hello test" for f in dummy.caught_frames)

@pytest.mark.asyncio
async def test_output_processor_discards_tts():
    out_proc = CovenantOutputProcessor()
    
    class DummyProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.caught_frames = []
            
        async def process_frame(self, frame, direction):
            self.caught_frames.append((frame, direction))
            
    dummy = DummyProcessor()
    out_proc.link(dummy)
    out_proc._prev = dummy 
    
    await out_proc.process_frame(TTSAudioRawFrame(b"fake_audio", 16000, 1), FrameDirection.DOWNSTREAM)
    await out_proc.process_frame(TranscriptionFrame(text="pass_me_up", user_id="1", timestamp=""), FrameDirection.DOWNSTREAM)
    
    assert len(dummy.caught_frames) == 1
    assert isinstance(dummy.caught_frames[0][0], TranscriptionFrame)
    assert dummy.caught_frames[0][1] == FrameDirection.UPSTREAM
