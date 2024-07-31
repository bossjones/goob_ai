# pdf

```
DEBUG:urllib3.connectionpool:http://localhost:8010 "POST /api/v1/collections/1dc341a3-6c3e-4af5-b391-5bd6c99c14d7/query HTTP/1.1" 200 7715
2024-07-08 21:29:35.207 | DEBUG    | urllib3.connectionpool:_make_request:549 - http://localhost:8010 "POST /api/v1/collections/1dc341a3-6c3e-4af5-b391-5bd6c99c14d7/query HTTP/1.1" 200 7715
Answer:
[
    Document(
        page_content='CHAPTER\nONE\nINTRODUCTION\nRich is a Python library for writing rich text (with color and style) to the terminal, and for displaying advanced content\nsuch as tables, markdown, and syntax
highlighted code.\nUse Rich to make your command line applications visually appealing and present data in a more readable way. Rich\ncan also be a useful debugging aid by pretty printing and syntax highlighting
data structures.\n1.1 Requirements\nRich works with macOS, Linux and Windows.\nOn Windows both the (ancient) cmd.exe terminal is supported and the new Windows Terminal. The latter has much\nimproved support for
color and style.\nRich requires Python 3.7.0 and above.\nNote: PyCharm users will need to enable "emulate terminal" in output console option in run/debug configuration to\nsee styled output.\n1.2
Installation\nYou can install Rich from PyPI with pip or your favorite package manager:\npip install rich\nAdd the -U switch to update to the current version, if Rich is already installed.\nIf you intend to use
Rich with Jupyter then there are some additional dependencies which you can install with the\nfollowing command:\npip install "rich[jupyter]"\n1\n',
        metadata={
            'author': 'Will McGugan',
            'creationDate': 'D:20230930141230Z',
            'creator': 'LaTeX with hyperref package',
            'file_path': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'format': 'PDF 1.5',
            'keywords': '',
            'modDate': 'D:20230930141230Z',
            'page': 6,
            'producer': 'pdfTeX-1.40.18',
            'source': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'subject': '',
            'title': 'Rich',
            'total_pages': 204,
            'trapped': ''
        }
    ),
    Document(
        page_content='CHAPTER\nONE\nINTRODUCTION\nRich is a Python library for writing rich text (with color and style) to the terminal, and for displaying advanced content\nsuch as tables, markdown, and syntax
highlighted code.\nUse Rich to make your command line applications visually appealing and present data in a more readable way. Rich\ncan also be a useful debugging aid by pretty printing and syntax highlighting
data structures.\n1.1 Requirements\nRich works with macOS, Linux and Windows.\nOn Windows both the (ancient) cmd.exe terminal is supported and the new Windows Terminal. The latter has much\nimproved support for
color and style.\nRich requires Python 3.7.0 and above.\nNote: PyCharm users will need to enable "emulate terminal" in output console option in run/debug configuration to\nsee styled output.\n1.2
Installation\nYou can install Rich from PyPI with pip or your favorite package manager:\npip install rich\nAdd the -U switch to update to the current version, if Rich is already installed.\nIf you intend to use
Rich with Jupyter then there are some additional dependencies which you can install with the\nfollowing command:\npip install "rich[jupyter]"\n1\n',
        metadata={
            'author': 'Will McGugan',
            'creationDate': 'D:20230930141230Z',
            'creator': 'LaTeX with hyperref package',
            'file_path': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'format': 'PDF 1.5',
            'keywords': '',
            'modDate': 'D:20230930141230Z',
            'page': 6,
            'producer': 'pdfTeX-1.40.18',
            'source': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'subject': '',
            'title': 'Rich',
            'total_pages': 204,
            'trapped': ''
        }
    ),
    Document(
        page_content='CHAPTER\nONE\nINTRODUCTION\nRich is a Python library for writing rich text (with color and style) to the terminal, and for displaying advanced content\nsuch as tables, markdown, and syntax
highlighted code.\nUse Rich to make your command line applications visually appealing and present data in a more readable way. Rich\ncan also be a useful debugging aid by pretty printing and syntax highlighting
data structures.\n1.1 Requirements\nRich works with macOS, Linux and Windows.\nOn Windows both the (ancient) cmd.exe terminal is supported and the new Windows Terminal. The latter has much\nimproved support for
color and style.\nRich requires Python 3.7.0 and above.\nNote: PyCharm users will need to enable "emulate terminal" in output console option in run/debug configuration to\nsee styled output.\n1.2
Installation\nYou can install Rich from PyPI with pip or your favorite package manager:\npip install rich\nAdd the -U switch to update to the current version, if Rich is already installed.\nIf you intend to use
Rich with Jupyter then there are some additional dependencies which you can install with the\nfollowing command:\npip install "rich[jupyter]"\n1\n',
        metadata={
            'author': 'Will McGugan',
            'creationDate': 'D:20230930141230Z',
            'creator': 'LaTeX with hyperref package',
            'file_path': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'format': 'PDF 1.5',
            'keywords': '',
            'modDate': 'D:20230930141230Z',
            'page': 6,
            'producer': 'pdfTeX-1.40.18',
            'source': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'subject': '',
            'title': 'Rich',
            'total_pages': 204,
            'trapped': ''
        }
    ),
    Document(
        page_content='Rich, Release 13.6.0\n1.3 Quick Start\nThe quickest way to get up and running with Rich is to import the alternative print function which takes the same\narguments as the built-in print
and may be used as a drop-in replacement. Here's how you would do that:\nfrom rich import print\nYou can then print strings or objects to the terminal in the usual way. Rich will do some basic syntax
highlighting and\nformat data structures to make them easier to read.\nStrings may contain Console Markup which can be used to insert color and styles in to the output.\nThe following demonstrates both console
markup and pretty formatting of Python objects:\n>>> print("[italic red]Hello[/italic red] World!", locals())\nThis writes the following output to the terminal (including all the colors and styles):\nIf you
would rather not shadow Python's built-in print, you can import rich.print as rprint (for example):\nfrom rich import print as rprint\nContinue reading to learn about the more advanced features of Rich.\n1.4
Rich in the REPL\nRich may be installed in the REPL so that Python data structures are automatically pretty printed with syntax high-\nlighting. Here's how:\n>>> from rich import pretty\n>>>
pretty.install()\n>>> ["Rich and pretty", True]\nYou can also use this feature to try out Rich renderables. Here's an example:\n>>> from rich.panel import Panel\n>>> Panel.fit("[bold yellow]Hi, I\'m a Panel",
border_style="red")\nRead on to learn more about Rich renderables.\n1.4.1 IPython Extension\nRich also includes an IPython extension that will do this same pretty install + pretty tracebacks. Here's how to load
it:\nIn [1]: %load_ext rich\nYou can also have it load by default by adding "rich" to the c.InteractiveShellApp.extension variable in IPython\nConfiguration.\n2\nChapter 1. Introduction\n',
        metadata={
            'author': 'Will McGugan',
            'creationDate': 'D:20230930141230Z',
            'creator': 'LaTeX with hyperref package',
            'file_path': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'format': 'PDF 1.5',
            'keywords': '',
            'modDate': 'D:20230930141230Z',
            'page': 7,
            'producer': 'pdfTeX-1.40.18',
            'source': '/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf',
            'subject': '',
            'title': 'Rich',
            'total_pages': 204,
            'trapped': ''
        }
    )
]
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete

~/dev/bossjones/goob_ai feature-custom-vector-store-tool* 19s
❯
```
