"""
Data models for CovenantAI.
"""
from typing import Any, Dict, List, Literal, Optional, Union, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
import yaml # type: ignore

# --- 5. AGENT TRACE MODELS ---

class ToolCallTrace(BaseModel):
    """A trace of a single tool invocation made by an agent."""
    model_config = ConfigDict(populate_by_name=True)
    
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    timestamp_ms: float

class AgentTrace(BaseModel):
    """The full trace of an agent's execution."""
    model_config = ConfigDict(populate_by_name=True)
    
    tool_calls: List[ToolCallTrace] = Field(default_factory=list)
    final_response: str
    asked_for_confirmation: bool = False
    duration_ms: float = 0.0
    interrupted_user: bool = False

# --- 1. ASSERTION MODELS ---

class BaseAssertion(BaseModel):
    """Base class for all assertions."""
    model_config = ConfigDict(populate_by_name=True)
    type: str

class ToolCallsInclude(BaseAssertion):
    type: Literal["tool_calls_include"] = "tool_calls_include"
    tools: List[str]

class ToolCallsExclude(BaseAssertion):
    type: Literal["tool_calls_exclude"] = "tool_calls_exclude"
    tools: List[str]

class ToolCallsSequence(BaseAssertion):
    type: Literal["tool_calls_sequence"] = "tool_calls_sequence"
    tools: List[str]
    strict: bool = False

class ResponseContains(BaseAssertion):
    type: Literal["response_contains"] = "response_contains"
    text: str
    case_sensitive: bool = False

class ResponseNotContains(BaseAssertion):
    type: Literal["response_not_contains"] = "response_not_contains"
    text: str
    case_sensitive: bool = False

class ResponseMatchesRegex(BaseAssertion):
    type: Literal["response_matches_regex"] = "response_matches_regex"
    pattern: str

class RequiresConfirmation(BaseAssertion):
    type: Literal["requires_confirmation"] = "requires_confirmation"
    expected: bool = True

class MaxToolCalls(BaseAssertion):
    type: Literal["max_tool_calls"] = "max_tool_calls"
    limit: int

class ToolCallArgContains(BaseAssertion):
    type: Literal["tool_call_arg_contains"] = "tool_call_arg_contains"
    tool: str
    arg: str
    value: str

class ResponseWithinMs(BaseAssertion):
    type: Literal["response_within_ms"] = "response_within_ms"
    max_ms: int

class ConversationFlowFollowed(BaseAssertion):
    type: Literal["conversation_flow_followed"] = "conversation_flow_followed"
    states: List[str]
    strict: bool = False

class NeverInterrupted(BaseAssertion):
    type: Literal["never_interrupted"] = "never_interrupted"
    expected: bool = False

class SingleTurnAssertion(BaseModel):
    """A single turn within a multi-turn conversation."""
    model_config = ConfigDict(populate_by_name=True)
    turn: str
    assert_: List['AssertionType'] = Field(default_factory=list, alias="assert")

class MultiTurnAssertion(BaseAssertion):
    type: Literal["multi_turn"] = "multi_turn"
    turns: List[SingleTurnAssertion]

AssertionType = Annotated[
    Union[
        ToolCallsInclude,
        ToolCallsExclude,
        ToolCallsSequence,
        ResponseContains,
        ResponseNotContains,
        ResponseMatchesRegex,
        RequiresConfirmation,
        MaxToolCalls,
        ToolCallArgContains,
        ResponseWithinMs,
        ConversationFlowFollowed,
        NeverInterrupted,
        MultiTurnAssertion,
    ],
    Field(discriminator="type"),
]

# --- 2. SCENARIO MODEL ---

class ScenarioModel(BaseModel):
    """A single test scenario definition."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str
    description: Optional[str] = None
    input: Optional[str] = None
    turns: Optional[List[SingleTurnAssertion]] = Field(default=None, alias="Multi-turn") # Keep compatible with yaml structure, though the prompt asked for MultiTurnAssertion as an assertion type
    runs: int = 10
    confidence_threshold: float = 0.80
    timeout_seconds: int = 30
    assert_: List[AssertionType] = Field(default_factory=list, alias="assert")
    
    @field_validator("runs")
    @classmethod
    def validate_runs(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError(f"runs must be between 1 and 100, got {v}")
        return v
        
    @model_validator(mode="after")
    def validate_confidence(self) -> "ScenarioModel":
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        return self

# --- 3. SUITE MODEL ---

class SuiteModel(BaseModel):
    """A complete suite of tests."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str
    description: Optional[str] = None
    agent: str
    framework: Literal["langchain", "openai-agents", "pipecat", "auto"] = "auto"
    default_runs: int = 10
    default_confidence_threshold: float = 0.80
    scenarios: List[ScenarioModel] = Field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SuiteModel":
        """Load and validate a suite from a YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to read or parse YAML file at '{path}': {e}")
            
        if not isinstance(data, dict):
            raise ValueError(f"YAML file at '{path}' must contain a dictionary at the top level.")
            
        try:
            return cls.model_validate(data)
        except Exception as e:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                raise ValueError(f"Validation failed for YAML file at '{path}':\n{e}")
            raise ValueError(f"Error validating YAML file at '{path}': {e}")

# --- 4. RESULT MODELS ---

class AssertionResult(BaseModel):
    """Result of a single assertion evaluation."""
    model_config = ConfigDict(populate_by_name=True)
    
    assertion_type: str
    passed: bool
    message: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class RunResult(BaseModel):
    """Result of a single attempt for a scenario."""
    model_config = ConfigDict(populate_by_name=True)
    
    run_index: int
    tool_calls: List[str]
    response: str
    assertion_results: List[AssertionResult] = Field(default_factory=list)
    passed: bool
    error: Optional[str] = None
    duration_ms: float

class ScenarioResult(BaseModel):
    """Aggregated results for a scenario across multiple runs."""
    model_config = ConfigDict(populate_by_name=True)
    
    scenario_name: str
    total_runs: int
    passed_runs: int
    pass_rate: float
    passed: bool
    confidence_threshold: float
    run_results: List[RunResult] = Field(default_factory=list)
    duration_ms: float

class SuiteResult(BaseModel):
    """Aggregated results for the entire suite."""
    model_config = ConfigDict(populate_by_name=True)
    
    suite_name: str
    total_scenarios: int
    passed_scenarios: int
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    duration_ms: float
    passed: bool

__all__ = [
    "ToolCallTrace",
    "AgentTrace",
    "BaseAssertion",
    "ToolCallsInclude",
    "ToolCallsExclude",
    "ToolCallsSequence",
    "ResponseContains",
    "ResponseNotContains",
    "ResponseMatchesRegex",
    "RequiresConfirmation",
    "MaxToolCalls",
    "ToolCallArgContains",
    "AssertionType",
    "ScenarioModel",
    "SuiteModel",
    "AssertionResult",
    "RunResult",
    "ScenarioResult",
    "SuiteResult",
]
