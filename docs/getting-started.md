# Getting Started with CovenantAI

## Installation
```bash
pip install covenant-ai
```

## Basic Usage
Create a test file `tests.yaml`:
```yaml
test_cases:
  - name: Greeting
    input: "Hello!"
    expected_output: "Hi there!"
```

Run the tests:
```bash
covenant tests.yaml
```
