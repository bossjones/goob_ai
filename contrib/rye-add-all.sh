#!/usr/bin/env bash

rye add fastapi
rye add boto3
rye add beautifulsoup4
rye add chardet
rye add langchain-community
rye add langchain
rye add langchain-openai
rye add openai
rye add pydantic-settings
rye add pydantic\[dotenv,email\]
rye add pypdf
rye add python-docx
rye add python-dotenv
rye add striprtf
rye add tenacity
rye add requests
rye add uvicorn
rye add aiocache
rye add aiodebug
rye add aiodns
rye add aiofile
rye add aiofiles
rye add aiohttp
rye add aiomonitor
rye add aioprometheus\[starlette\]
rye add aiosql
rye add aiosqlite
rye add attrs
rye add better-exceptions
rye add click-spinner
rye add codetiming
rye add discord-py
rye add factory-boy
rye add faker
rye add fonttools\[woff\]
rye add gallery-dl
rye add google-auth
rye add google-auth-oauthlib
rye add gutter
rye add imageio
rye add imutils
rye add logging-tree
rye add loguru
rye add lxml
rye add markdown
rye add matplotlib
rye add md2pdf
rye add memory-profiler
rye add motor
rye add multiprocess
rye add mutagen
rye add numpy
rye add passlib\[bcrypt\]
rye add Pillow
rye add prettytable
rye add pycryptodome
rye add pygments
rye add pyinspect
rye add PyPDF2
rye add pypi-command-line\[speedups\]
rye add pytablewriter\[html\]
rye add python-Levenshtein
rye add python-slugify
rye add pytz
rye add redis
rye add scenedetect\[opencv\]
rye add sentencepiece
rye add simpletransformers
rye add soupsieve
rye add streamlit
rye add telnetlib3
rye add tqdm
rye add transformers
rye add typer
rye add uritemplate
rye add uritools
rye add validators
rye add watchdog\[watchmedo\]
rye add webcolors
rye add websockets
rye add docutils
rye add pyinvoke
rye add pathlib-mate
rye add lazy-object-proxy
rye add httpx\[http2\]
rye add python-json-logger
rye add torch
rye add chromadb
rye add duckduckgo-search
rye add wikipedia
rye add pandas
rye add Babel
rye add dask


rye add --dev black
rye add --dev boto3-stubs\[essential\]
rye add --dev bpython
rye add --dev flake8
rye add --dev isort
rye add --dev mypy
rye add --dev mypy-boto3
rye add --dev pre-commit
rye add --dev pydocstyle
rye add --dev pytest-cov
rye add --dev pytest-mock
rye add --dev pytest-sugar
rye add --dev pytest
rye add --dev pyupgrade
rye add --dev requests-mock
rye add --dev rich
rye add --dev ruff
rye add --dev types-beautifulsoup4
rye add --dev types-boto
rye add --dev types-mock
rye add --dev types-PyYAML
rye add --dev types-aiofiles
rye add --dev types-click
rye add --dev types-colorama
rye add --dev types-dataclasses
rye add --dev types-freezegun
rye add --dev types-pytz
rye add --dev types-setuptools
rye add --dev types-six
rye add --dev MonkeyType
rye add --dev hunter
rye add --dev sourcery
rye add --dev types-html5lib
rye add --dev types-pillow
rye add --dev types-pyasn1
rye add --dev types-python-jose
rye add --dev typing-extensions
rye add --dev pyright
rye add --dev pytest-rerunfailures
rye add --dev pytest-asyncio
rye add --dev validate-pyproject\[all,store\]
rye add --dev pylint-per-file-ignores


rye add --dev types-ujson && \
rye add --dev types-tqdm && \
rye add --dev types-toml && \
rye add --dev types-six && \
rye add --dev types-regex && \
rye add --dev types-redis && \
rye add --dev types-psutil && \
rye add --dev types-mypy-extensions && \
rye add --dev types-jsonschema && \
rye add --dev types-html5lib && \
rye add --dev types-colorama && \
rye add --dev types-click-spinner && \
rye add --dev types-cffi && \
rye add --dev pandas-stubs && \
rye add --dev types-urllib3 && \
rye add --dev types-protobuf  && \
rye add --dev grpc-stubs && \
rye add --dev types-contextvars && \
rye add --dev types-dataclasses && \
rye add --dev types-beautifulsoup4
