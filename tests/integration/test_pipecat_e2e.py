import os
import pytest
from click.testing import CliRunner
from pathlib import Path

# Skip the whole module if pipecat isn't installed
pytest.importorskip("pipecat")

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    TranscriptionFrame, LLMFullResponseStartFrame, LLMTextFrame,
    LLMFullResponseEndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame
)

from covenant.models import (
    ScenarioModel, SuiteModel, ResponseContains, ToolCallsInclude, 
    MultiTurnAssertion, SingleTurnAssertion
)
from covenant.runner import ScenarioRunner, SuiteRunner
from covenant.adapters.pipecat import PipecatAdapter
from covenant.cli import cli

class MockScriptedLLMService(FrameProcessor):
    def __init__(self, script):
        super().__init__()
        self.script = script

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = frame.text
            calls, response = self.script.get(text, ([], "I don't know."))
            
            for call in calls:
                await self.push_frame(FunctionCallInProgressFrame(
                    tool_call_id="1",
                    function_name=call["tool"],
                    arguments=call.get("args", {})
                ), FrameDirection.DOWNSTREAM)
                await self.push_frame(FunctionCallResultFrame(
                    tool_call_id="1",
                    function_name=call["tool"],
                    arguments=call.get("args", {}),
                    result="Ok"
                ), FrameDirection.DOWNSTREAM)
            
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMTextFrame(response), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

def create_pipeline_factory(script):
    from covenant.adapters.pipecat_pipeline import CovenantTestTransport
    async def factory(transport: CovenantTestTransport):
        return Pipeline([
            transport.input(),
            MockScriptedLLMService(script),
            transport.output()
        ])
    return factory


# Global factories for the CLI to import
PASSING_SCRIPT = {
    "hello": ([], "hi there"),
    "search": ([{"tool": "google"}], "results found"),
    "bye": ([], "goodbye"),
}
def get_passing_agent(transport):
    return create_pipeline_factory(PASSING_SCRIPT)(transport)

MIXED_SCRIPT = {
    "pass1": ([], "pass 1"),
    "pass2": ([], "pass 2"),
    "fail1": ([], "fail"),
}
def get_mixed_agent(transport):
    return create_pipeline_factory(MIXED_SCRIPT)(transport)


@pytest.mark.asyncio
async def test_full_suite_all_passing():
    script = {
        "one": ([], "response one"),
        "two": ([{"tool": "search"}], "response two"),
        "three": ([], "response three"),
    }
    adapter = PipecatAdapter(create_pipeline_factory(script))
    runner = ScenarioRunner(adapter)
    
    scenarios = [
        ScenarioModel(name="s1", input="one", runs=5, confidence_threshold=1.0, assert_=[ResponseContains(text="response one")]),
        ScenarioModel(name="s2", input="two", runs=5, confidence_threshold=1.0, assert_=[ToolCallsInclude(tools=["search"])]),
        ScenarioModel(name="s3", input="three", runs=5, confidence_threshold=1.0, assert_=[ResponseContains(text="three")]),
    ]
    
    # We test them via the ScenarioRunner directly first
    results = []
    for s in scenarios:
        res = await runner.run_scenario(s)
        results.append(res)
        
    assert all(r.passed for r in results)
    assert all(r.pass_rate >= s.confidence_threshold for r, s in zip(results, scenarios))


@pytest.mark.asyncio
async def test_full_suite_with_failure(monkeypatch):
    # We mock import_agent to return our factory
    import covenant.runner
    monkeypatch.setattr(covenant.runner, "import_agent", lambda path: get_mixed_agent)

    scenarios = [
        ScenarioModel(name="s1", input="pass1", runs=1, confidence_threshold=1.0, assert_=[ResponseContains(text="pass 1")]),
        ScenarioModel(name="s2", input="pass2", runs=1, confidence_threshold=1.0, assert_=[ResponseContains(text="pass 2")]),
        ScenarioModel(name="s3", input="fail1", runs=1, confidence_threshold=1.0, assert_=[ResponseContains(text="wrong text")]),
    ]
    suite = SuiteModel(name="suite", agent="dummy", scenarios=scenarios)
    
    runner = SuiteRunner()
    result = await runner.run_suite(suite)
    
    assert result.passed is False
    assert result.total_scenarios == 3
    assert result.passed_scenarios == 2
    
    assert result.scenario_results[0].passed is True
    assert result.scenario_results[1].passed is True
    assert result.scenario_results[2].passed is False


@pytest.mark.asyncio
async def test_multi_turn_e2e():
    script = {
        "What is the weather in Paris?": ([{"tool": "get_weather", "args": {"city": "Paris"}}], "Sunny in Paris"),
        "What about Tokyo?": ([{"tool": "get_weather", "args": {"city": "Tokyo"}}], "Rainy in Tokyo"),
    }
    adapter = PipecatAdapter(create_pipeline_factory(script))
    runner = ScenarioRunner(adapter)
    
    multi_turn = MultiTurnAssertion(turns=[
        SingleTurnAssertion(turn="What is the weather in Paris?", assert_=[ToolCallsInclude(tools=["get_weather"])]),
        SingleTurnAssertion(turn="What about Tokyo?", assert_=[ToolCallsInclude(tools=["get_weather"])]),
    ])
    
    scenario = ScenarioModel(
        name="multi-turn scenario",
        input=None,
        runs=1,
        assert_=[multi_turn]
    )
    
    res = await runner.run_scenario(scenario)
    assert res.passed is True
    
    # Check trace content
    trace1_tools = res.run_results[0].tool_calls # aggregated tool calls for all turns
    # In run_single, tool calls are extended
    assert list(trace1_tools) == ["get_weather", "get_weather"]


def test_pipecat_yaml_loads_and_runs():
    runner = CliRunner()
    # Runs the very first scenario of the existing example (runs=3 for speed)
    # The example YAML has scenarios. We'll run the CLI with a custom YAML that points to it
    
    yaml_path = Path("examples/pipecat_agent/tests/voice_assistant.yaml")
    assert yaml_path.exists(), "voice_assistant.yaml should exist"
    
    # We will invoke covenant run --suite on a temp yaml that pulls just 1 scenario
    # Alternatively we can just run the whole thing, but we want it fast and isolated
    # Since the example agent OpenAILLMService in agent.py uses Mock-like logic internally
    # it doesn't need API keys!
    result = runner.invoke(cli, ["run", "--suite", str(yaml_path)])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_cli_run_with_pipecat_suite(tmp_path):
    runner = CliRunner()
    
    yaml_content = """
name: CLI Pipecat E2E
agent: tests.integration.test_pipecat_e2e.get_passing_agent
framework: pipecat
scenarios:
  - name: "Simple text push"
    input: "hello"
    runs: 2
    assert:
      - type: response_contains
        text: "hi there"
"""
    suite_file = tmp_path / "test_cli.yaml"
    suite_file.write_text(yaml_content)
    
    # Ensure current dir is in pythonpath
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    
    result = runner.invoke(cli, ["run", "--suite", str(suite_file)], env=env)
    
    assert result.exit_code == 0
    assert "1 passed" in result.output or "Passed" in result.output
    assert "PASS" in result.output
