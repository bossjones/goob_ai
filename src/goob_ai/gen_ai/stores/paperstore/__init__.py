from __future__ import annotations

from pickledb import PickleDB  # pyright: ignore[reportMissingTypeStubs]


class PaperStore:
    """Rudimentary persistent storage for paper titles, abstracts, and generated summaries."""

    def __init__(self, filepath: str) -> None:
        """Initialize the PaperStore with a given file path.

        Args:
            filepath (str): The path to the database file.
        """
        self.db = PickleDB(filepath, auto_dump=False, sig=False)

    def save_summary(self, paper_id: str, summary_type: str, summary: str) -> None:
        """
        Save a summary for a given paper and summary type.

        This method stores the summary text for a specific paper identified by its unique
        identifier and the type of summary. The summary is saved in the database.

        Args:
            paper_id (str): The unique identifier for the paper.
            summary_type (str): The type of summary (e.g., 'short', 'detailed').
            summary (str): The summary text.
        """
        self.db.set(f"{paper_id}-{summary_type}", summary)

    def get_summary(self, paper_id: str, summary_type: str) -> str | None:
        """
        Retrieve a summary for a given paper and summary type.

        This method fetches the summary of a paper from the database using the paper's unique identifier
        and the type of summary. If the summary does not exist, it returns None.

        Args:
            paper_id (str): The unique identifier for the paper.
            summary_type (str): The type of summary (e.g., 'short', 'detailed').

        Returns:
            str | None: The summary text if it exists, otherwise None.

        Args:
            paper_id (str): The unique identifier for the paper.
            summary_type (str): The type of summary (e.g., 'short', 'detailed').

        Returns:
            str | None: The summary text if it exists, otherwise None.
        """
        return self.db.get(f"{paper_id}-{summary_type}")

    def save_title_abstract(self, paper_id: str, title: str, abstract: str) -> None:
        """
        Save the title and abstract for a given paper.

        This method stores the title and abstract of a paper identified by its unique
        identifier in the database. The title and abstract are saved as separate entries
        in the database.

        Args:
            paper_id (str): The unique identifier for the paper.
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.
        """
        self.db.set(f"{paper_id}-title", title)
        self.db.set(f"{paper_id}-abstract", abstract)

    def get_title(self, paper_id: str) -> str | None:
        """
        Retrieve the title for a given paper.

        This method fetches the title of a paper from the database using the paper's unique identifier.
        If the title does not exist, it returns None.

        Args:
            paper_id (str): The unique identifier for the paper.

        Returns:
            str | None: The title if it exists, otherwise None.
        """
        return self.db.get(f"{paper_id}-title")

    def get_abstract(self, paper_id: str) -> str | None:
        """
        Retrieve the abstract for a given paper.

        This method fetches the abstract of a paper from the database using the paper's unique identifier.
        If the abstract does not exist, it returns None.

        Args:
            paper_id (str): The unique identifier for the paper.

        Returns:
            str | None: The abstract text if it exists, otherwise None.
        """
        return self.db.get(f"{paper_id}-abstract")

    def save(self) -> None:
        """
        Persist the current state of the database to disk.

        This method writes the in-memory state of the database to the file specified
        during the initialization of the PaperStore instance. It ensures that all
        changes made to the database are saved to disk.

        Returns:
            None
        """
        self.db.dump()

    def add_mentioned_paper(self, paper_id: str, chat_id: str) -> None:
        """Add a mentioned paper to a chat's list of mentioned papers.

        Args:
            paper_id (str): The unique identifier for the paper.
            chat_id (str): The unique identifier for the chat.
        """
        papers = self.db.get(chat_id) if self.db.exists(chat_id) else []
        papers.append(paper_id)
        self.db.set(chat_id, papers)

    def get_papers(self, chat_id: str) -> str:
        """
        Retrieve the list of mentioned papers for a given chat.

        This method fetches the list of paper IDs mentioned in a specific chat
        and returns a formatted string containing each paper's ID and title.
        If no papers are mentioned in the chat, it returns "NONE".

        Args:
            chat_id (str): The unique identifier for the chat.

        Returns:
            str: A formatted string of mentioned papers with their titles, or "NONE" if no papers are mentioned.
        """
        if not self.db.exists(chat_id):
            return "NONE"
        papers = self.db.get(chat_id)
        return "\n".join([f"[`{paper_id}`] {self.get_title(paper_id)}" for paper_id in papers])
