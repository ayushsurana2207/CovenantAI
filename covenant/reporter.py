"""
Test execution reporting.
"""
from typing import Any, Dict, Optional, DefaultDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.progress import TaskID
from rich.table import Table
from rich.panel import Panel
from collections import defaultdict

from covenant.models import ScenarioResult, SuiteResult

class RichReporter:
    """Rich console reporter for test outcomes."""
    
    def __init__(self) -> None:
        import sys
        # Disable colors if not running in a TTY
        no_color = not sys.stdout.isatty()
        self.console = Console(no_color=no_color)
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None
        
    def suite_start(self, suite_name: str, total_scenarios: int) -> None:
        """Prints the start of the suite execution."""
        content = f"Running [bold cyan]{total_scenarios}[/bold cyan] scenarios"
        panel = Panel(
            content,
            title=f"[covenant] {suite_name}",
            expand=False,
            border_style="cyan"
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()
        
    def scenario_start(self, scenario_name: str, runs: int, threshold: float) -> None:
        """Starts the progress tracking for a new scenario."""
        self.console.print(f"  [bold]*[/bold] {scenario_name}  [dim][{runs} runs @ {threshold*100:.0f}% threshold][/dim]")
        
        # We start a fresh progress context per scenario to keep the output clean
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} runs"),
            console=self.console,
            transient=True # Disappears when done
        )
        
        self.progress.start()
        self.task_id = self.progress.add_task("Evaluating agent...", total=runs)
        
    def run_complete(self, run_index: int, passed: bool, duration_ms: float) -> None:
        """Advances the scenario progress bar."""
        if self.progress and self.task_id is not None:
            self.progress.advance(self.task_id)
            
    def scenario_complete(self, result: ScenarioResult) -> None:
        """Prints the final outcome of the scenario and any granular failure details."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None
            
        color = "green" if result.passed else "red"
        status_text = "PASS" if result.passed else "FAIL"
        
        summary = (
            f"[{color}][bold]{status_text}[/bold][/]  {result.scenario_name}  "
            f"{result.passed_runs}/{result.total_runs} runs passed ({result.pass_rate*100:.0f}%)  "
            f"[dim][{result.duration_ms/1000:.1f}s][/dim]"
        )
        self.console.print(summary)
        
        if not result.passed:
            self._print_failure_table(result)
            
    def _print_failure_table(self, result: ScenarioResult) -> None:
        """Compiles and prints a structured table of assertion failures."""
        if result.total_runs == 0:
            return
            
        # Map: AssertionType -> (count, example message)
        failures: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "example": ""})
        sys_errors = 0
        sys_error_example = ""
        
        for run in result.run_results:
            if run.error:
                sys_errors += 1
                if not sys_error_example:
                    sys_error_example = run.error
                continue
                
            for assertion in run.assertion_results:
                if not assertion.passed:
                    data = failures[assertion.assertion_type]
                    data["count"] += 1
                    if not data["example"]:
                        data["example"] = assertion.error or str(assertion.details)
                        
        if failures or sys_errors > 0:
            table = Table(show_header=True, header_style="bold red", padding=(0, 2))
            table.add_column("Assertion / Source")
            table.add_column("Failed in")
            table.add_column("Example")
            
            # Print assertion failures
            for a_type, data in failures.items():
                table.add_row(
                    a_type,
                    f"{data['count']}/{result.total_runs} runs",
                    str(data['example']).replace("\n", " ").strip()[:80] + "..." if len(str(data['example'])) > 80 else str(data['example'])
                )
                
            # Print execution/system errors
            if sys_errors > 0:
                 table.add_row(
                    "[bold]System / Agent Error[/bold]",
                    f"{sys_errors}/{result.total_runs} runs",
                    str(sys_error_example).replace("\n", " ").strip()[:80] + "..." if len(str(sys_error_example)) > 80 else str(sys_error_example)
                 )
                 
            self.console.print(table)
            self.console.print()

    def suite_complete(self, result: SuiteResult) -> None:
        """Prints the final summary panel for the suite."""
        color = "green" if result.passed else "red"
        
        content = (
            f"[bold]{result.total_scenarios}[/bold] scenarios   "
            f"[green]{result.passed_scenarios} passed[/green]  "
            f"[{color}]{result.total_scenarios - result.passed_scenarios} failed[/{color}]\n"
            f"Total time: [dim]{result.duration_ms/1000:.1f}s[/dim]"
        )
        
        panel = Panel(
            content,
            expand=False,
            border_style=color
        )
        
        self.console.print()
        self.console.print(panel)
        
        if not result.passed:
            self.console.print("\n[dim]Run with --verbose for full trace details[/dim]")

__all__ = ["RichReporter"]
