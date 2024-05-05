"""Test the goob_ai CLI."""
from __future__ import annotations

from typer.testing import CliRunner

from goob_ai.cli import APP

runner = CliRunner()


class TestApp:
    """Test the goob_ai CLI."""

    def test_version(self) -> None:
        """Test the version command."""
        result = runner.invoke(APP, ["--version"])
        assert result.exit_code == 0
        assert "goob_ai version:" in result.stdout

    def test_help(self) -> None:
        """Test the help command."""
        result = runner.invoke(APP, ["--help"])
        assert result.exit_code == 0
        assert "goob_ai: A Python package for the Transport for London (TFL) API." in result.stdout


    def test_about(self) -> None:
        """Test the help command."""
        result = runner.invoke(APP, ["about"])
        assert result.exit_code == 0
        assert "This is GoobBot CLI" in result.stdout
