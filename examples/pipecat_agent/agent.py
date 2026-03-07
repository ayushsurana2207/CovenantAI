import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    TranscriptionFrame, LLMFullResponseStartFrame, LLMTextFrame,
    LLMFullResponseEndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame,
    InterruptionFrame
)

class OpenAILLMService(FrameProcessor):
    def __init__(self, system_prompt: str, tools: list):
        super().__init__()
        self.system_prompt = system_prompt
        self.tools = tools
        self.memory = []

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()
            self.memory.append(text)
            
            is_weather = "weather" in text or "tokyo" in text
            is_flight = "flight" in text
            
            if is_weather:
                city = "Tokyo" if "tokyo" in text else "Paris"
                await self.push_frame(FunctionCallInProgressFrame(
                    tool_call_id="1", function_name="get_weather", arguments={'city': city}
                ), FrameDirection.DOWNSTREAM)
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="1", function_name="get_weather", arguments={'city': city}, result="Sunny"
                ), FrameDirection.DOWNSTREAM)
                
            elif is_flight:
                await self.push_frame(FunctionCallInProgressFrame(
                    tool_call_id="2", function_name="confirm_action", arguments={'action': 'book_flight'}
                ), FrameDirection.DOWNSTREAM)
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="2", function_name="confirm_action", arguments={'action': 'book_flight'}, result="Yes"
                ), FrameDirection.DOWNSTREAM)

            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMTextFrame(f"Processed: {text}"), FrameDirection.DOWNSTREAM)
            
            if "interrupt" in text:
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

async def create_pipeline(transport) -> Pipeline:
    llm = OpenAILLMService(
        system_prompt="You are a helpful voice assistant.",
        tools=["get_weather", "confirm_action"]
    )
    
    pipeline = Pipeline([
        transport.input(),
        llm,
        transport.output()
    ])
    
    return pipeline
