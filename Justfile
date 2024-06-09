set shell := ["zsh", "-cu"]
LOCATION_PYTHON := `python -c "import sys;print(sys.executable)"`

# just manual: https://github.com/casey/just/#readme

# Ignore the .env file that is only used by the web service
set dotenv-load := false

K3D_VERSION := `k3d version`
CURRENT_DIR := "$(pwd)"
PATH_TO_TRAEFIK_CONFIG := CURRENT_DIR / "mounts/var/lib/rancer/k3s/server/manifests/traefik-config.yaml"

# base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -b 0 -i cert.pem -o ca.pem" }
base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -w 0 -i cert.pem > ca.pem" }
grep_cmd := if "{{os()}}" =~ "macos" { "ggrep" } else { "grep" }
conntrack_fix := if "{{os()}}" =~ "linux" { "--k3s-arg '--kube-proxy-arg=conntrack-max-per-core=0@server:*' --k3s-arg '--kube-proxy-arg=conntrack-max-per-core=0@agent:*'" } else { "" }

en0_ip := `ifconfig en0 | grep inet | cut -d' ' -f2 | grep -v ":"`


_default:
		@just --list

info:
		print "Python location: {{LOCATION_PYTHON}}"
		print "PATH_TO_TRAEFIK_CONFIG: {{PATH_TO_TRAEFIK_CONFIG}}"
		print "OS: {{os()}}"

# verify python is running under pyenv
which-python:
		python -c "import sys;print(sys.executable)"

# when developing, you can use this to watch for changes and restart the server
autoreload-code:
	rye run watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM rye run goobctl go

local-open-coverage:
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

open-coverage: local-open-coverage

rye-get-pythons:
	rye fetch 3.8.19
	rye fetch 3.9.19
	rye fetch 3.10.14
	rye fetch 3.11.4
	rye fetch 3.12.3

rye-add-all:
	./contrib/rye-add-all.sh

pre-commit-run-all:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

pipdep-tree:
	pipdeptree --python .venv/bin/python3

# install rye tools globally
rye-tool-install:
	rye install invoke
	rye install pipdeptree
	rye install click

lint-github-actions:
	actionlint

# check that taplo is installed to lint/format TOML
check-taplo-installed:
	@command -v taplo >/dev/null 2>&1 || { echo >&2 "taplo is required but it's not installed. run 'brew install taplo'"; exit 1; }

fmt-python:
	git ls-files '*.py' | xargs rye run pre-commit run --files

# format pyproject.toml using taplo
fmt-toml:
	pre-commit run taplo-format --all-files

# format all code using pre-commit config
fmt: fmt-python fmt-toml

# lint python files using ruff
lint-python:
	pre-commit run ruff --all-files

# lint TOML files using taplo
lint-toml: check-taplo-installed
	pre-commit run taplo-lint --all-files

# Lint all files in the current directory (and any subdirectories).
lint: lint-python lint-toml

# SOURCE: https://github.com/RobertCraigie/prisma-client-py/blob/da53c4280756f1a9bddc3407aa3b5f296aa8cc10/Makefile#L77
clean:
	rm -rf .cache
	rm -rf `find . -name __pycache__`
	rm -rf .tests_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -f coverage.xml

# generate type stubs for the project
createstubs:
	./scripts/createstubs.sh

# sweep init
sweep-init:
	rye run sweep init

# TODO: We should try out trunk
# By default, we use the following config that runs Trunk, an opinionated super-linter that installs all the common formatters and linters for your codebase. You can set up and configure Trunk for yourself by following https://docs.trunk.io/get-started.
# sandbox:
#   install:
#     - trunk init
#   check:
#     - trunk fmt {file_path}
#     - trunk check {file_path}


download-models:
	curl -L 'https://www.dropbox.com/s/im6ytahqgbpyjvw/ScreenNetV1.pth?dl=1' > src/goob_ai/data/ScreenNetV1.pth
