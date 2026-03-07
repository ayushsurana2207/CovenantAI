# Contributing to CovenantAI

We welcome contributions! Please follow these guidelines to make the process smooth and effective.

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/covenant-ai.git
   cd covenant-ai
   ```
3. Create a clean virtual environment and install the package in editable mode with development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Coding Standards

Code quality is enforced using standard Python tools. Your code must adhere to:

- **Formatting & Linting**: We use `ruff` to ensure everything is PEP-8 compliant. Run `ruff check covenant/ tests/`.
- **Typing**: We enforce strict type checking with `mypy`. Run `mypy covenant/ --strict` to verify type safety.
- **Coverage**: Your PR must maintain the minimum **80% testing coverage**.

## Running Tests

We use `pytest` as our testing runner. You can execute the test suite explicitly checking for coverage as follows:

```bash
pytest tests/ -v --cov=covenant --cov-report=term-missing
```

## How to Add a New Assertion Type

CovenantAI provides an extensible engine allowing new evaluation criteria. Here is how to add a custom assertion type:

1. **Define the Data Model (`covenant/models.py`)**:
   Create a new class subclassing `BaseAssertion`. Provide literal string properties for `type` to distinguish your assertion from others in the `AssertionType` union!
   ```python
   class MyNewAssertion(BaseAssertion):
       type: Literal["my_new_assertion"] = "my_new_assertion"
       # Define your parameters here
       target_value: int
   ```
   *Remember to add your new assertion to the `AssertionType` discriminated union and update `__all__`.*

2. **Implement Evaluation Logic (`covenant/assertions.py`)**:
   Create a private evaluation method intercepting your model and the generated `AgentTrace`. The method *must* return an `AssertionResult`.
   ```python
   def _eval_my_new_assertion(assertion: MyNewAssertion, trace: AgentTrace) -> AssertionResult:
       # Write check logic here...
       passed = True
       message = "Target hit exactly!"
       
       return AssertionResult(
           assertion_type=assertion.type,
           passed=passed,
           message=message,
           details={
               "assertion_type": assertion.type,
               "expected": assertion.target_value,
               "actual": 0 # Extract actual state from `trace`
           }
       )
   ```

3. **Register the Evaluator (`covenant/assertions.py`)**:
   Add your new private evaluation method into the O(1) `EVALUATORS` dispatch dictionary mapping exactly your model's literal `type` string.
   ```python
   EVALUATORS = {
       # ... existing evaluators,
       "my_new_assertion": _eval_my_new_assertion,
   }
   ```

4. **Write Tests (`tests/test_assertions.py`)**:
   Create unit tests proving correct pass boundaries, fail bounds, and edge cases natively testing `evaluate(MyNewAssertion(...), trace)`.

## PR Checklist

Before submitting your Pull Request, ensure:
- [ ] Added or updated models and functionality.
- [ ] Successfully passed `ruff check covenant/ tests/`.
- [ ] Successfully compiled `mypy covenant/ --strict` successfully.
- [ ] Passed Pytest `pytest tests/ -v` and met coverage boundaries.
- [ ] Updated corresponding documentation in `/docs` and added PR explanations!
