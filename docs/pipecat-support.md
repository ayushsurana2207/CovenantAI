# Pipecat Support in CovenantAI

## Introduction

Pipecat is fundamentally different from request/response frameworks like LangChain or the OpenAI Agents SDK. Instead of sending a single text payload and waiting for a single text response, Pipecat applications are real-time, streaming frame pipelines. They handle duplex audio, continuous transcription, interruption events, and asynchronous chunked text generation over an active WebRTC session.

CovenantAI handles this architecture transparently. When testing a Pipecat agent, Covenant dynamically injects a `CovenantTestTransport` to replace physical audio interfaces. It captures `TranscriptionFrame`s sent downstream, passively aggregates tool usage and spoken text via the `CovenantFrameObserver`, and manages asynchronous session termination when the LLM finishes speaking. This allows your LLM behavioral pipeline to be tested natively purely through text representations without standing up a live Daily room or rendering expensive TTS audio.

---

## Supported Pipeline Patterns

Covenant automatically detects and mounts your `pipecat-ai` agent if it conforms to one of these 3 common architectural patterns:

### Pattern A: Factory Function Returning a Pipeline
This is the recommended pattern. Your agent is a function that accepts an injected transport.

**Agent Definition (`agent.py`)**
```python
from pipecat.pipeline.pipeline import Pipeline

async def create_pipeline(transport) -> Pipeline:
    # transport.input() replaces your DailyTransport microphone
    # transport.output() replaces your DailyTransport speaker
    return Pipeline([
        transport.input(),
        llm_service,
        transport.output()
    ])
```

**YAML Suite**
```yaml
framework: auto
agent: my_module.agent.create_pipeline
scenarios:
  - name: "Simple text push"
    input: "Hello world"
    # ...
```

### Pattern B: Instantiated Pipeline Object
If you export a raw `Pipeline`, Covenant will automatically traverse its components and dynamically substitute your Daily/WebSocket transports with the Covenant mocking processors.

**Agent Definition (`agent.py`)**
```python
pipeline = Pipeline([
    DailyTransport().input(), 
    # ...
    DailyTransport().output()
])
```

**YAML Suite**
```yaml
framework: auto
agent: my_module.agent.pipeline
```

### Pattern C: Object with a `.run()` Method
If you wrap your pipeline in an application class, export the instance. Covenant will call `app.run(transport)` and inspect `app.pipeline`.

**Agent Definition (`agent.py`)**
```python
class VoiceApp:
    async def run(self, transport):
        self.pipeline = Pipeline([transport.input(), OpenAILLMService(), transport.output()])
        
    # (runner logic happens here)

app = VoiceApp()
```

**YAML Suite**
```yaml
framework: auto
agent: my_module.agent.app
```

---

## Pipecat-Specific Assertions

Because Pipecat agents stream asynchronously and maintain long-running state buffers, Covenant ships with 4 custom-built voice assertions:

### `response_within_ms`
Measures the end-to-end latency of a response, from the moment the user transcription finishes to the moment the `LLMFullResponseEndFrame` is emitted.
```yaml
assert:
  - type: response_within_ms
    max_ms: 1500
```

### `conversation_flow_followed`
Validates that the conversational states (often represented as function calls like `get_weather` -> `confirm_action`) occurred in a specific sequence.
```yaml
assert:
  - type: conversation_flow_followed
    states: ["fetch_context", "confirm_booking"]
    strict: true # Fails if ANY other tool is called
```

### `never_interrupted`
Listens for `InterruptionFrame`s arriving downstream *while* the LLM is actively streaming response text. Ensures the agent doesn't interrupt the user accidentally, or alternatively, that the AI *did* interrupt them when prompted.
```yaml
assert:
  - type: never_interrupted
    expected: false # We expect it DID NOT interrupt the user
```

### `multi_turn`
(See section below)

---

## Multi-Turn Testing

Voice agents operate fundamentally over persistent sessions. A single-turn test often isn't enough to validate if context from a previous sentence was retained.

The `multi_turn` assertion lets you simulate a contiguous back-and-forth session with the agent, checking assertions across each specific turn, all while keeping the Pipecat Task Runner spinning.

**Explanation:** Multi-turn matters because the LLM memory context buffer is retained across these text inputs. If you asked "What's the weather in Paris?" and then "What about Tokyo?", the second turn should inherently understand we are still talking about the weather. 

```yaml
scenarios:
  - name: "Maintains conversational context"
    # Note: 'input' is omitted at the scenario level constraint
    Multi-turn:
      - turn: "What is the weather in Paris?"
        assert:
          - type: tool_calls_include
            tools: ["get_weather"]
      - turn: "...what about Tokyo?"
        assert:
          - type: tool_calls_include
            tools: ["get_weather"]
          - type: tool_call_arg_contains
            tool: get_weather
            arg: city
            value: "Tokyo"
    assert:
      - type: multi_turn
        turns: [] # Handled structurally under the hood
```

---

## Limitations

Be honest about what Covenant is testing. Covenant is a *behavioral logic* validation harness, not a hardware emulator.

1. **No Real Audio**: We do not test the actual microphone latency, voice activation detection (VAD) accuracy, or the resulting text-to-speech (TTS) intelligibility. `TTSAudioRawFrame`s are discarded silently. We test the LLM behavioral routing layer inside the pipecat stream.
2. **Network Drops & WebRTC Jitter**: Transport layer stability is outside Covenant's scope. Packet drops will not be simulated. 
3. **Pipeline Startup Thresholds**: The `response_within_ms` assertion measures the entire duration of a turn. This naturally includes Pipecat's asyncio event-loop pipeline initialization latency (typically ~10-100ms) on the first turn. Pad your thresholds slightly to account for compute overhead. 

---

## Testing Without Real API Keys

Running high-volume probabilistic suites against OpenAI can drain credits rapidly. For unit testing, implement a mock Pipecat processor that mimics `OpenAILLMService` without reaching the network.

```python
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import TranscriptionFrame, LLMFullResponseEndFrame, LLMTextFrame, FunctionCallResultFrame

class MockLLMService(FrameProcessor):
    """A fully localized determinisitic frame emitter"""
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            if "hello" in frame.text.lower():
                await self.push_frame(LLMTextFrame("Hi there!"), FrameDirection.DOWNSTREAM)
            else:
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="1", function_name="lookup", arguments={}, result="value"
                ), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
```

By binding `MockLLMService` in your standard test fixtures, you validate your pipeline wiring and conditional branching natively before ever touching an API key!
