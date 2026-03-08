"""
Pipecat adapter for CovenantAI.
"""
import asyncio
import inspect
import time
import json
from typing import Any

from covenant.models import AgentTrace, ToolCallTrace
from covenant.exceptions import CovenantRunError, CovenantTimeoutError
from covenant.adapters.base import BaseAdapter
from covenant.adapters.pipecat_pipeline import CovenantFrameObserver, CovenantTestTransport

class PipecatAdapter(BaseAdapter):
    """Adapter for testing Pipecat pipelines."""

    def __init__(self, agent: Any):
        self.agent = agent

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        try:
            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.pipeline.task import PipelineTask
            
            if isinstance(agent, Pipeline) or isinstance(agent, PipelineTask):
                return True
                
            if getattr(agent, "__module__", "").startswith("pipecat"):
                return True

            if inspect.isroutine(agent):
                sig = inspect.signature(agent)
                if "transport" in sig.parameters:
                    return True
                    
            if hasattr(agent, "run") and inspect.isroutine(agent.run):
                sig = inspect.signature(agent.run)
                if "transport" in sig.parameters:
                    return True
                    
        except ImportError:
            pass
            
        return False

    async def run(self, user_input: str, timeout_seconds: int) -> AgentTrace:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.frames.frames import EndFrame
        
        transport = CovenantTestTransport()
        observer = CovenantFrameObserver()
        
        pipeline = None
        
        if inspect.isroutine(self.agent):
            pipeline = await self.agent(transport)
        elif isinstance(self.agent, Pipeline):
            pipeline = self.agent
            self._replace_transport(pipeline, transport)
        elif isinstance(self.agent, PipelineTask):
            pipeline = self.agent.pipeline
            self._replace_transport(pipeline, transport)
        elif hasattr(self.agent, "run"):
            # Pattern C
            asyncio.create_task(self.agent.run(transport))
            await asyncio.sleep(0.1)
            pipeline = getattr(self.agent, "pipeline", None)
            if not pipeline:
                raise CovenantRunError("Agent must expose a .pipeline attribute after initialization", None)
            self._replace_transport(pipeline, transport)
            
        if not pipeline:
            raise CovenantRunError("Could not extract a Pipecat Pipeline from the agent.", None)

        # Inject observer
        self._inject_observer(pipeline, observer)
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False
            )
        )
        
        runner = PipelineRunner(handle_sigint=False)
        start_time_ms = time.perf_counter() * 1000
        
        async def inject_input() -> None:
            await asyncio.sleep(0.1)
            await transport.inject_transcription(user_input)
            
        async def wait_completion() -> None:
            success = await observer.wait_for_completion(timeout_seconds)
            await task.queue_frame(EndFrame())
            if not success:
                if observer._errors:
                    raise CovenantRunError(f"Pipecat pipeline errors: {observer._errors}", None)
                raise CovenantTimeoutError(f"Pipecat pipeline did not complete within {timeout_seconds}s")

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    runner.run(task),
                    inject_input(),
                    wait_completion()
                ),
                timeout=timeout_seconds + 2
            )
        except asyncio.TimeoutError:
            raise CovenantTimeoutError(f"Pipecat pipeline wait_for timed out after {timeout_seconds+2}s")
            
        if observer._errors:
            raise CovenantRunError(f"Pipecat pipeline errors: {observer._errors}", None)
            
        trace_data = observer.get_trace_data()
        
        tool_calls = []
        for fc in trace_data["function_calls"]:
            args = fc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            elif not isinstance(args, dict):
                args = {"raw": str(args)}
            tool_calls.append(ToolCallTrace(
                tool_name=fc["name"],
                arguments=args,
                result=str(fc["result"]) if fc["result"] is not None else "",
                timestamp_ms=fc["timestamp_ms"]
            ))
            
        return AgentTrace(
            tool_calls=tool_calls,
            final_response=trace_data["final_response"],
            asked_for_confirmation=trace_data["asked_for_confirmation"],
            duration_ms=(time.perf_counter() * 1000) - start_time_ms,
            interrupted_user=trace_data.get("interrupted_user", False),
        )

    async def run_multi_turn(self, turns: list[str], timeout_seconds: int) -> list[AgentTrace]:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.frames.frames import EndFrame
        
        transport = CovenantTestTransport()
        observer = CovenantFrameObserver()
        
        pipeline = None
        
        if inspect.isroutine(self.agent):
            pipeline = await self.agent(transport)
        elif isinstance(self.agent, Pipeline):
            pipeline = self.agent
            self._replace_transport(pipeline, transport)
        elif isinstance(self.agent, PipelineTask):
            pipeline = self.agent.pipeline
            self._replace_transport(pipeline, transport)
        elif hasattr(self.agent, "run"):
            asyncio.create_task(self.agent.run(transport))
            await asyncio.sleep(0.1)
            pipeline = getattr(self.agent, "pipeline", None)
            if not pipeline:
                raise CovenantRunError("Agent must expose a .pipeline attribute after initialization", None)
            self._replace_transport(pipeline, transport)
            
        if not pipeline:
            raise CovenantRunError("Could not extract a Pipecat Pipeline from the agent.", None)

        self._inject_observer(pipeline, observer)
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False
            )
        )
        
        runner = PipelineRunner(handle_sigint=False)
        runner_task = asyncio.create_task(runner.run(task))
        await asyncio.sleep(0.1)
        
        traces = []
        for turn_input in turns:
            start_time_ms = time.perf_counter() * 1000
            
            # Reset observer state manually since it's the same instance
            observer._function_calls.clear()
            observer._response_chunks.clear()
            observer._errors.clear()
            observer._response_complete = False
            observer._response_started = False
            observer._agent_interrupted_user = False
            observer._completion_event.clear()
            
            await transport.inject_transcription(turn_input)
            
            success = await observer.wait_for_completion(timeout_seconds)
            if not success:
                # Cleanup
                await task.queue_frame(EndFrame())
                runner_task.cancel()
                if observer._errors:
                    raise CovenantRunError(f"Pipecat pipeline errors: {observer._errors}", None)
                raise CovenantTimeoutError(f"Pipecat pipeline did not complete turn within {timeout_seconds}s")
                
            if observer._errors:
                await task.queue_frame(EndFrame())
                runner_task.cancel()
                raise CovenantRunError(f"Pipecat pipeline errors: {observer._errors}", None)
                
            trace_data = observer.get_trace_data()
            
            tool_calls = []
            for fc in trace_data["function_calls"]:
                args = fc["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                elif not isinstance(args, dict):
                    args = {"raw": str(args)}
                tool_calls.append(ToolCallTrace(
                    tool_name=fc["name"],
                    arguments=args,
                    result=str(fc["result"]) if fc["result"] is not None else "",
                    timestamp_ms=fc["timestamp_ms"]
                ))
                
            traces.append(AgentTrace(
                tool_calls=tool_calls,
                final_response=trace_data["final_response"],
                asked_for_confirmation=trace_data["asked_for_confirmation"],
                duration_ms=(time.perf_counter() * 1000) - start_time_ms,
                interrupted_user=trace_data.get("interrupted_user", False),
            ))
            
        await task.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(runner_task, timeout=2.0)
        except asyncio.TimeoutError:
            pass
            
        return traces

    def _replace_transport(self, pipeline: Any, test_transport: CovenantTestTransport) -> None:
        if not hasattr(pipeline, "processors") or not pipeline.processors:
            return
            
        first = pipeline.processors[0]
        last = pipeline.processors[-1]
        
        def is_transport(p: Any) -> bool:
            name = p.__class__.__name__.lower()
            return "transport" in name or "input" in name or "output" in name
            
        if is_transport(first):
            pipeline.processors[0] = test_transport.input()
        if is_transport(last):
            pipeline.processors[-1] = test_transport.output()

    def _inject_observer(self, pipeline: Any, observer: CovenantFrameObserver) -> None:
        if not hasattr(pipeline, "processors"):
            return
        target_idx = -1
        for i, p in enumerate(pipeline.processors):
            name = p.__class__.__name__.lower()
            if "llm" in name or "service" in name:
                target_idx = i
                break
                
        if target_idx != -1:
            pipeline.processors.insert(target_idx + 1, observer)
        else:
            pipeline.processors.insert(1, observer)
