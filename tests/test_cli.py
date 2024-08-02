"""Test the goob_ai CLI."""

from __future__ import annotations

from goob_ai.cli import APP
from typer.testing import CliRunner


runner = CliRunner()


class TestApp:
    """Test the goob_ai CLI."""

    def test_version(self) -> None:
        """Test the version command."""
        result = runner.invoke(APP, ["version"])
        assert result.exit_code == 0
        assert "goob_ai version:" in result.stdout

    def test_help(self) -> None:
        """Test the help command."""
        result = runner.invoke(APP, ["--help"])
        assert result.exit_code == 0
        assert "About command" in result.stdout
        # assert "Interact w/ chroma local vectorstore" in result.stdout
        # assert "Create a pinecone index" in result.stdout
        # assert "Delete a pinecone index" in result.stdout
        # assert "Deps command" in result.stdout
        # assert "Main entry point for GoobAI" in result.stdout
        # assert "Smoketest for querying readthedocs pdfs against vectorstore." in result.stdout
        # assert "Manually run screencrop's download_and_predict service and get bounding boxes" in result.stdout
        # assert "Manually run screencrop's download_and_predict service and get bounding boxes" in result.stdout
        # assert "Manually run screencrop's download_and_predict service and get bounding boxes" in result.stdout
        # assert "Generate typestubs GoobAI" in result.stdout
        # assert "Manually run screncrop service and get bounding boxes" in result.stdout
        # assert "Show command" in result.stdout
        # assert "Version command" in result.stdout

    def test_about(self) -> None:
        """Test the help command."""
        result = runner.invoke(APP, ["about"])
        assert result.exit_code == 0
        assert "This is GoobBot CLI" in result.stdout
