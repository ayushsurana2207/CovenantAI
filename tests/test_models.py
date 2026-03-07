"""
Tests for the CovenantAI data models.
"""
import pytest
from pydantic import ValidationError
from covenant.models import (
    ScenarioModel,
    SuiteModel,
    ToolCallsInclude,
    MaxToolCalls,
    ResponseContains
)

@pytest.fixture
def valid_yaml_path(tmp_path):
    """Fixture to create a temporary valid YAML file."""
    yaml_content = """
    name: "Valid Test Suite"
    description: "A suite to test yaml parsing."
    agent: "dummy.agent.path"
    framework: "langchain"
    default_runs: 5
    scenarios:
      - name: "Simple Scenario"
        input: "Hello"
        runs: 10
        confidence_threshold: 0.9
        assert:
          - type: "response_contains"
            text: "Hi"
          - type: "max_tool_calls"
            limit: 3
    """
    file_path = tmp_path / "valid_suite.yaml"
    file_path.write_text(yaml_content)
    return str(file_path)

@pytest.fixture
def invalid_yaml_path(tmp_path):
    """Fixture to create a temporary invalid YAML file (missing required fields)."""
    yaml_content = """
    name: "Invalid Suite"
    # Missing 'agent' field which is required
    scenarios:
      - name: "Scenario without input"
    """
    file_path = tmp_path / "invalid_suite.yaml"
    file_path.write_text(yaml_content)
    return str(file_path)

@pytest.fixture
def malformed_yaml_path(tmp_path):
    """Fixture to create a malformed YAML file."""
    yaml_content = "this is not valid dictionary yaml list: [\n"
    file_path = tmp_path / "malformed.yaml"
    file_path.write_text(yaml_content)
    return str(file_path)

def test_suite_from_valid_yaml(valid_yaml_path):
    """Test loading a suite from a valid YAML file."""
    suite = SuiteModel.from_yaml(valid_yaml_path)
    
    assert suite.name == "Valid Test Suite"
    assert suite.agent == "dummy.agent.path"
    assert suite.framework == "langchain"
    assert len(suite.scenarios) == 1
    
    scenario = suite.scenarios[0]
    assert scenario.name == "Simple Scenario"
    assert scenario.input == "Hello"
    assert scenario.runs == 10
    assert scenario.confidence_threshold == 0.9
    
    assert len(scenario.assert_) == 2
    
    assert isinstance(scenario.assert_[0], ResponseContains)
    assert scenario.assert_[0].text == "Hi"
    
    assert isinstance(scenario.assert_[1], MaxToolCalls)
    assert scenario.assert_[1].limit == 3

def test_suite_from_invalid_yaml(invalid_yaml_path):
    """Test loading a suite from YAML missing required fields."""
    with pytest.raises(ValueError, match="Validation failed for YAML file"):
        SuiteModel.from_yaml(invalid_yaml_path)

def test_suite_from_malformed_yaml(malformed_yaml_path):
    """Test loading a suite from malformed YAML strings."""
    with pytest.raises(ValueError, match="(must contain a dictionary at the top level|Failed to read or parse YAML file)"):
        SuiteModel.from_yaml(malformed_yaml_path)

def test_scenario_validation_boundaries():
    """Test boundary validation for runs and confidence_threshold in ScenarioModel."""
    # Valid boundaries
    ScenarioModel(name="valid1", input="test", runs=1, confidence_threshold=0.0)
    ScenarioModel(name="valid2", input="test", runs=100, confidence_threshold=1.0)
    
    # Invalid runs (too low)
    with pytest.raises(ValidationError, match="runs"):
        ScenarioModel(name="invalid", input="test", runs=0)
        
    # Invalid runs (too high)
    with pytest.raises(ValidationError, match="runs"):
        ScenarioModel(name="invalid", input="test", runs=101)
        
    # Invalid confidence (too low)
    with pytest.raises(ValidationError, match="confidence_threshold"):
        ScenarioModel(name="invalid", input="test", confidence_threshold=-0.1)
        
    # Invalid confidence (too high)
    with pytest.raises(ValidationError, match="confidence_threshold"):
        ScenarioModel(name="invalid", input="test", confidence_threshold=1.5)

def test_discriminated_union_parsing():
    """Test that all assertion types parse correctly from dicts."""
    data = {"type": "tool_calls_include", "tools": ["search"]}
    # ScenarioModel parses the assert list which uses the Union
    scenario = ScenarioModel(name="test", input="test", assert_=[data])
    
    assert isinstance(scenario.assert_[0], ToolCallsInclude)
    assert scenario.assert_[0].tools == ["search"]
