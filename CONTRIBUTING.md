# history

```

pdm add boto3==1.34.41
pdm add fastapi==0.110.0


pdm add -dG test black==24.3.0 && \
pdm add -dG test boto3-stubs\[essential\]==1.34.79 && \
pdm add -dG test bpython==0.24 && \
pdm add -dG test flake8 && \
pdm add -dG test isort==5.13.2 && \

pdm add -dG test mypy==1.8.0 && \
pdm add -dG test mypy-boto3==1.34.79 && \
pdm add -dG test pre-commit==3.2.2 && \
pdm add -dG test pydocstyle==6.3.0 && \
pdm add -dG test pytest-cov==4.1.0 && \
pdm add -dG test pytest-mock==3.12.0 && \
pdm add -dG test pytest-sugar==1.0.0 && \
pdm add -dG test pytest==8.0.0 && \
pdm add -dG test pyupgrade==3.15.2 && \
pdm add -dG test requests_mock==1.11.0 && \
pdm add -dG test rich==13.7.1 && \
pdm add -dG test ruff==0.3.7 && \
pdm add -dG test types-beautifulsoup4==4.12.0.20240229 && \
pdm add -dG test types-boto==2.49.18.20240205 && \
pdm add -dG test types-mock==5.1.0.20240311 && \
pdm add -dG test "types-requests<2.31.0.7"


eval $(pdm venv activate)



pdm add -dG testvalidate-pyproject\[all,store\]==0.16 && \
pdm add beautifulsoup4==4.10.0 && \
pdm add chardet==5.2.0 && \
pdm add langchain_community==0.0.33 && \
pdm add langchain_openai==0.0.8 && \
pdm add langchain==0.1.16 && \
pdm add openai==1.10.0 && \
pdm add pydantic-settings==2.1.0 && \
pdm add pydantic==2.6.1 && \
pdm add pypdf==4.0.1 && \
pdm add python-docx==1.1.0 && \
pdm add python-dotenv==1.0.1 && \
pdm add striprtf==0.0.26 && \
pdm add tenacity==8.1.0 && \
pdm add requests==v2.31.0 && \
pdm add uvicorn==0.28.0
```
