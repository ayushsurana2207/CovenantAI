"""
Tests for the probabilistic execution engine.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from covenant.runner import import_agent, ScenarioRunner, SuiteRunner
from covenant.exceptions import CovenantImportError, CovenantRunError, CovenantTimeoutError
from covenant.models import ScenarioModel, SuiteModel, ToolCallsInclude, AgentTrace
from tests.fixtures.mock_adapter import MockAdapter

def test_import_agent_valid():
    from covenant.models import SuiteModel
    agent = import_agent("covenant.models.SuiteModel")
    assert agent is SuiteModel

def test_import_agent_invalid_path():
    with pytest.raises(CovenantImportError, match="Path must be a dotted module path"):
        import_agent("foobar")
        
def test_import_agent_missing_module():
    with pytest.raises(CovenantImportError, match="Failed to import"):
        import_agent("covenant.not_a_module.NotAClass")

@pytest.mark.asyncio
async def test_scenario_runner_pass_rate_and_threshold():
    adapter = MockAdapter("mock")
    # 5 total runs
    # 4 succeed with expected tool call
    for _ in range(4):
        adapter.traces.append(AgentTrace(tool_calls=[{"tool_name": "search", "arguments": {}, "result": "ok", "timestamp_ms": 0}], final_response=""))
        
    # 1 fails (missing tool call)
    adapter.traces.append(AgentTrace(tool_calls=[], final_response=""))
    
    runner = ScenarioRunner(adapter)
    scenario = ScenarioModel(
        name="Test",
        input="Search query",
        runs=5,
        confidence_threshold=0.8,
        assert_=[ToolCallsInclude(tools=["search"])]
    )
    
    res = await runner.run_scenario(scenario)
    assert len(res.run_results) == 5
    passed_runs = [r for r in res.run_results if r.passed]
    assert len(passed_runs) == 4
    
    # 4/5 = 0.8, which meets the 0.8 threshold
    assert res.pass_rate == 0.8
    assert res.passed is True
    assert res.duration_ms > 0

@pytest.mark.asyncio
async def test_scenario_runner_aborts_consecutive_errors():
    adapter = MockAdapter("mock")
    # Enqueue 3 crashes
    for _ in range(3):
        adapter.exceptions_to_raise.append(ValueError("API overloaded"))
        
    # We ask for 10 runs, it should abort after 3
    runner = ScenarioRunner(adapter)
    scenario = ScenarioModel(
        name="Crash Test",
        input="Crash query",
        runs=10,
        confidence_threshold=0.8,
        assert_=[]
    )
    
    res = await runner.run_scenario(scenario)
    
    # Needs exactly 10 runs in output, 3 real runs + 7 aborted runs padding
    assert len(res.run_results) == 10
    
    # First 3 should be actual errors
    for i in range(3):
        assert "API overloaded" in res.run_results[i].error
        
    # Remaining 7 should be aborted padding
    for i in range(3, 10):
        assert "Run aborted due to 3 consecutive upstream agent errors" in res.run_results[i].error

    assert res.pass_rate == 0.0
    assert not res.passed

@pytest.mark.asyncio
async def test_scenario_runner_counts_timeouts_gracefully():
    adapter = MockAdapter("mock")
    # Don't abort on consecutive timeouts
    # Make adapter just time out internally without raising
    async def slow_mock(*args, **kwargs):
        raise CovenantTimeoutError("Timed out")
        
    adapter.run = slow_mock
    
    runner = ScenarioRunner(adapter)
    scenario = ScenarioModel(
        name="Slow Test",
        input="query",
        runs=5,
        confidence_threshold=0.8,
        assert_=[]
    )
    
    res = await runner.run_scenario(scenario)
    assert len(res.run_results) == 5
    for r in res.run_results:
        assert not r.passed
        assert "Timed out" in r.error

@pytest.mark.asyncio
@patch('covenant.runner.get_adapter')
@patch('covenant.runner.import_agent')
async def test_suite_runner_execution(mock_import_agent, mock_get_adapter):
    # Mocking dependency resolution
    mock_agent = MagicMock()
    mock_import_agent.return_value = mock_agent
    
    adapter = MockAdapter("mock")
    adapter.traces.append(AgentTrace(final_response="Success"))
    adapter.traces.append(AgentTrace(final_response="Success"))
    mock_get_adapter.return_value = adapter
    
    suite = SuiteModel(
        name="Integration Suite",
        agent="tests.fixtures.mock_agent",
        scenarios=[
            ScenarioModel(name="S1", input="I1", runs=1),
            ScenarioModel(name="S2", input="I2", runs=1)
        ]
    )
    
    runner = SuiteRunner()
    
    cb_calls = []
    def progress(msg):
        cb_calls.append(msg)
        
    res = await runner.run_suite(suite, progress_callback=progress)
    
    assert res.passed
    assert len(res.scenario_results) == 2
    assert cb_calls # Should receive status updates
    assert any("S1" in c for c in cb_calls)
    assert any("S2" in c for c in cb_calls)
