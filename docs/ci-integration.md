# CI/CD Integration

You can easily integrate CovenantAI into your GitHub Actions workflow:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install covenant-ai
      - run: covenant tests.yaml
```
