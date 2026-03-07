"""
Core execution engine for CovenantAI.
"""
import asyncio
import importlib
import time
from typing import Any, Callable, Optional, List

from covenant.models import (
    ScenarioModel,
    SuiteModel,
    RunResult,
    ScenarioResult,
    SuiteResult,
    AgentTrace,
    MultiTurnAssertion
)
from covenant.adapters import BaseAdapter, get_adapter
from covenant.assertions import evaluate
from covenant.exceptions import CovenantTimeoutError, CovenantRunError, CovenantImportError

def import_agent(dotted_path: str) -> Any:
    """Import an agent object from a dotted module path."""
    try:
        if "." not in dotted_path:
            raise ValueError("Path must be a dotted module path (e.g. 'myproject.agent.app')")
        module_path, attr_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except Exception as e:
        raise CovenantImportError(f"Failed to import agent from '{dotted_path}': {str(e)}") from e

class ScenarioRunner:
    """Runs a scenario N times concurrently and aggregates probabilistic results."""
    
    def __init__(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter
        self.semaphore = asyncio.Semaphore(5)

    async def _run_single(self, scenario: ScenarioModel, run_index: int) -> RunResult:
        async with self.semaphore:
            start_time = time.perf_counter()
            trace: Optional[AgentTrace] = None
            error: Optional[str] = None
            
            try:
                multi_turn = None
                for a in scenario.assert_:
                    if isinstance(a, MultiTurnAssertion):
                        multi_turn = a
                        break
                
                assertion_results = []
                passed = False
                tool_calls = []
                response = ""
                
                if multi_turn:
                    turns_input = [t.turn for t in multi_turn.turns]
                    traces = await self.adapter.run_multi_turn(
                        turns=turns_input,
                        timeout_seconds=scenario.timeout_seconds
                    )
                    
                    for i, turn in enumerate(multi_turn.turns):
                        trace = traces[i]
                        tool_calls.extend(trace.tool_calls)
                        response += f"Turn {i+1}: {trace.final_response}\n"
                        for assertion in turn.assert_:
                            assertion_results.append(evaluate(assertion, trace))
                            
                    # We still evaluate scenario-level assertions if any (like the MultiTurnAssertion itself which passes passively)
                    for assertion in scenario.assert_:
                        # We evaluate against the last trace generally, though MultiTurn Assertion just returns true
                        if traces:
                            assertion_results.append(evaluate(assertion, traces[-1]))
                        else:
                            # if somehow no traces, we should probably fail or skip
                            pass
                        
                    passed = all(ar.passed for ar in assertion_results) if assertion_results else True
                    
                else:
                    trace = await self.adapter.run(
                        user_input=scenario.input or "",
                        timeout_seconds=scenario.timeout_seconds
                    )
                    
                    if trace:
                        tool_calls = trace.tool_calls
                        response = trace.final_response
                        for assertion in scenario.assert_:
                            assertion_results.append(evaluate(assertion, trace))
                        passed = all(ar.passed for ar in assertion_results) if assertion_results else True
                        
            except CovenantTimeoutError as e:
                error = str(e) or f"Run timed out after {scenario.timeout_seconds}s"
            except CovenantRunError as e:
                error = str(e)
            except Exception as e:
                error = f"Unexpected error during run: {str(e)}"
                
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return RunResult(
                run_index=run_index,
                tool_calls=[tc.tool_name for tc in tool_calls],
                response=response.strip(),
                assertion_results=assertion_results,
                passed=passed,
                error=error,
                duration_ms=duration_ms
            )

    async def run_scenario(self, scenario: ScenarioModel) -> ScenarioResult:
        """Run a scenario N times and aggregate probabilistic results."""
        start_time = time.perf_counter()
        
        runs: List[RunResult] = []
        consecutive_errors = 0
        
        # Determine number of batches needed. We map directly to gather lists to allow early termination checks.
        total_runs = scenario.runs
        
        # We run batches of 5 (since our semaphore is 5) to check for consecutive errors
        tasks = []
        for i in range(total_runs):
            tasks.append(self._run_single(scenario, i + 1))
            
        # Due to early termination requirement on consecutive errors, running strictly concurrently isn't safe.
        # However, asyncio.gather on all tasks will ignore the consecutive error rule during flight.
        # Instead, we process them as they complete or process in chunks.
        # For simplicity and to meet the requirement strictly, we will chunk by semaphore size.
        
        chunk_size = 5
        chunked_tasks = [tasks[x:x+chunk_size] for x in range(0, len(tasks), chunk_size)]
        
        for chunk in chunked_tasks:
            results = await asyncio.gather(*chunk)
            
            for res in results:
                runs.append(res)
                # Check for critical non-timeout errors
                if res.error and "timed out" not in res.error.lower():
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                    
                if consecutive_errors >= 3:
                    break
                    
            if consecutive_errors >= 3:
                # Pad the remaining runs as aborted errors
                remaining = total_runs - len(runs)
                for i in range(remaining):
                    err_res = RunResult(
                        run_index=len(runs) + 1,
                        tool_calls=[],
                        response="",
                        assertion_results=[],
                        passed=False,
                        error="Run aborted due to 3 consecutive upstream agent errors.",
                        duration_ms=0.0
                    )
                    runs.append(err_res)
                break

        passed_runs = sum(1 for r in runs if r.passed)
        pass_rate = passed_runs / total_runs if total_runs > 0 else 0.0
        passed = pass_rate >= scenario.confidence_threshold
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return ScenarioResult(
            scenario_name=scenario.name,
            total_runs=total_runs,
            passed_runs=passed_runs,
            pass_rate=pass_rate,
            passed=passed,
            confidence_threshold=scenario.confidence_threshold,
            run_results=runs,
            duration_ms=duration_ms
        )

class SuiteRunner:
    """Runs all scenarios in a given suite."""
    
    async def run_suite(self, suite: SuiteModel, progress_callback: Optional[Callable[[str], None]] = None) -> SuiteResult:
        """Run all scenarios in a suite sequentially and return aggregated results."""
        start_time = time.perf_counter()
        
        if progress_callback:
            progress_callback(f"Importing agent from {suite.agent}...")
            
        agent = import_agent(suite.agent)
        adapter = get_adapter(agent)
        
        if progress_callback:
            progress_callback(f"Resolved adapter: {type(adapter).__name__}")
            
        scenario_runner = ScenarioRunner(adapter)
        scenario_results = []
        
        for idx, scenario in enumerate(suite.scenarios, 1):
            if progress_callback:
                progress_callback(f"Running scenario {idx}/{len(suite.scenarios)}: '{scenario.name}' ({scenario.runs} runs)")
                
            result = await scenario_runner.run_scenario(scenario)
            scenario_results.append(result)
            
            if progress_callback:
                status_str = "PASSED" if result.passed else "FAILED"
                progress_callback(f"  -> {status_str} (Rate: {result.pass_rate*100:.1f}%, Threshold: {scenario.confidence_threshold*100:.1f}%)")
                
        passed = all(r.passed for r in scenario_results) if scenario_results else True
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if progress_callback:
             progress_callback(f"Suite run complete. Total scenarios passed: {sum(1 for r in scenario_results if r.passed)}/{len(scenario_results)}")
             
        return SuiteResult(
            suite_name=suite.name,
            total_scenarios=len(suite.scenarios),
            passed_scenarios=sum(1 for r in scenario_results if r.passed),
            scenario_results=scenario_results,
            passed=passed,
            duration_ms=duration_ms
        )

__all__ = ["import_agent", "ScenarioRunner", "SuiteRunner"]
