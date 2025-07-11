# AMT Tests

This directory contains tests for the AMT (Automated Music Transcription) project.

## Structure

- `unit/`: Unit tests for individual components
  - `test_config.py`: Tests for configuration system
  - `test_settings.py`: Tests for settings loading
  - `test_cli.py`: Tests for CLI functionality
  - `test_simple.py`: Simple configuration tests
  - `test_midi_processor.py`: Tests for MIDI processing
- `integration/`: Integration tests for testing multiple components together
  - `test_processing_pipeline.py`: Tests for the entire processing pipeline
  - `test_fast.py`: Tests for music generation
- `fixtures/`: Common test fixtures and data

## Running Tests

To run all tests:

```bash
pytest
```

To run only unit tests:

```bash
pytest tests/unit
```

To run only integration tests:

```bash
pytest tests/integration
```

To run a specific test file:

```bash
pytest tests/unit/test_midi_processor.py
```

To run a specific test:

```bash
pytest tests/unit/test_midi_processor.py::TestMIDIProcessor::test_load_midi
```

## Coverage Report

To generate a coverage report:

```bash
pytest --cov=amt --cov-report=html
```

This will create an HTML coverage report in the `htmlcov` directory.

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for test files, `Test*` for test classes, and `test_*` for test functions.
2. Use appropriate fixtures from `conftest.py` when possible.
3. Keep unit tests focused on testing a single component.
4. Use integration tests to test the interaction between multiple components.
5. Add appropriate markers to categorize tests (e.g., `@pytest.mark.slow` for slow tests).

## Test Dependencies

All test dependencies are listed in the `pyproject.toml` file under the `dev` optional dependencies. To install them:

```bash
pip install -e ".[dev]"
``` 