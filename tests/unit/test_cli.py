#!/usr/bin/env python3
"""
Test script for AMT CLI
"""

import pytest
from click.testing import CliRunner
from amt.cli import cli

def test_cli_import():
    """Test that CLI can be imported successfully."""
    # If import fails, the test will fail
    assert cli is not None, "CLI import failed"

def test_cli_commands():
    """Test that CLI has expected commands."""
    expected_commands = {"process", "train", "generate", "test", "collect"}
    actual_commands = set(cli.commands.keys())
    
    # Check that all expected commands are present
    for cmd in expected_commands:
        assert cmd in actual_commands, f"Command '{cmd}' not found in CLI"

def test_cli_help():
    """Test that CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, "CLI help command failed"
    assert "Usage:" in result.output, "CLI help output is incorrect" 