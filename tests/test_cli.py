"""
Tests for the command-line interface.
"""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from covenant.cli import cli
from covenant.models import SuiteResult, ScenarioResult

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Behavioral regression testing for AI agents." in result.output
    
def test_init_scaffolding(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init", "test-agent"])
        assert result.exit_code == 0
        assert "Successfully initialized 'test-agent' project!" in result.output
        
        tests_dir = Path("tests")
        assert tests_dir.exists()
        
        yaml_file = tests_dir / "test-agent.yaml"
        agent_file = Path("test-agent_agent.py")
        
        assert yaml_file.exists()
        assert agent_file.exists()
        
        yaml_content = yaml_file.read_text()
        assert "name: test-agent Suite" in yaml_content
        assert "framework: langchain" in yaml_content # default

def test_run_command_successful_ci(runner, tmp_path, monkeypatch):
    from covenant.models import SuiteModel, ScenarioModel
    
    # Needs absolute path for isolated filesystem to resolve correctly or mock SuiteModel
    test_yaml = tmp_path / "test.yaml"
    test_yaml.write_text("""name: Test Suite
agent: dummy.agent
scenarios:
  - name: S1
    input: "test\"""")

    # Mock the internal SuiteModel parsing and execution
    def mock_execute(*args, **kwargs):
        return SuiteResult(
            suite_name="Test Suite",
            total_scenarios=1,
            passed_scenarios=1,
            scenario_results=[
                ScenarioResult(
                    scenario_name="S1",
                    total_runs=1,
                    passed_runs=1,
                    pass_rate=1.0,
                    passed=True,
                    confidence_threshold=0.8,
                    duration_ms=10.0,
                    run_results=[]
                )
            ],
            duration_ms=10.0,
            passed=True
        )
        
    import covenant.cli
    monkeypatch.setattr(covenant.cli.asyncio, "run", mock_execute)
    
    # We must patch from_yaml because isolated filesystem path won't have dummy.agent
    monkeypatch.setattr(SuiteModel, "from_yaml", lambda p: SuiteModel(name="test", agent="test", scenarios=[]))

    with runner.isolated_filesystem(temp_dir=tmp_path):
        out_file = Path("out.json")
        # Must catch output directly from monkeypatched commands
        result = runner.invoke(cli, ["run", "--suite", str(test_yaml), "--ci", "--output", str(out_file)], catch_exceptions=False)
        
        assert result.exit_code == 0
        assert out_file.exists()
        # Ensure it outputs JUnit schema on standard out directly
        assert "<testsuites>" in result.output
        assert "failures=\"0\"" in result.output
        assert "tests=\"1\"" in result.output

def test_diff_command_regressions(runner, tmp_path):
    base_file = tmp_path / "base.json"
    curr_file = tmp_path / "curr.json"
    
    base_res = SuiteResult(
        suite_name="Test",
        total_scenarios=1,
        passed_scenarios=1,
        passed=True,
        duration_ms=0,
        scenario_results=[
            ScenarioResult(
                scenario_name="S1",
                total_runs=10, passed_runs=9, pass_rate=0.9,
                passed=True, confidence_threshold=0.8, duration_ms=0, run_results=[]
            )
        ]
    )
    
    # Drop pass rate to 60% (30 point drop = failure)
    curr_res = base_res.model_copy(deep=True)
    curr_res.scenario_results[0].pass_rate = 0.6
    
    base_file.write_text(base_res.model_dump_json())
    curr_file.write_text(curr_res.model_dump_json())
    
    result = runner.invoke(cli, ["diff", "--baseline", str(base_file), "--current", str(curr_file)], catch_exceptions=False)
    
    assert result.exit_code == 1 # Regressions throw 1
    assert "Regression detected" in result.output

def test_diff_command_succesful(runner, tmp_path):
    base_file = tmp_path / "base.json"
    curr_file = tmp_path / "curr.json"
    
    base_res = SuiteResult(
        suite_name="Test",
        total_scenarios=1,
        passed_scenarios=1,
        passed=True,
        duration_ms=0,
        scenario_results=[
            ScenarioResult(
                scenario_name="S1",
                total_runs=10, passed_runs=9, pass_rate=0.9,
                passed=True, confidence_threshold=0.8, duration_ms=0, run_results=[]
            )
        ]
    )
    
    # Elevate pass rate to 100% (+10 points = success)
    curr_res = base_res.model_copy(deep=True)
    curr_res.scenario_results[0].pass_rate = 1.0
    
    base_file.write_text(base_res.model_dump_json())
    curr_file.write_text(curr_res.model_dump_json())
    
    result = runner.invoke(cli, ["diff", "--baseline", str(base_file), "--current", str(curr_file)], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "No significant regressions detected" in result.output
