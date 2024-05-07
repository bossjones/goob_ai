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
	watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM python src/bot.py

# # via ada
# taplo-dry-run:
# 	taplo format --check --config taplo.toml --verbose --diff pyproject.toml

# taplo:
# 	taplo format --config taplo.toml pyproject.toml

# # Try fomatting existing code using ruff but simply display the diff
# ruff-fmt-dry-run:
# 	@echo "fix pyproject.toml first"
# 	taplo format --check pyproject.toml
# 	@echo "isort fixes first"
# 	ruff check . --select I --diff --config=pyproject.toml
# 	@echo "format everything else"
# 	ruff format --check --diff --config=pyproject.toml .

# ruff-config:
# 	ruff check --config=pyproject.toml --show-settings

# # Show config we would use to lint with ruff
# lint-show-settings:
# 	ruff check --config=pyproject.toml --show-settings | pbcopy

# # Lint all files in the current directory (and any subdirectories).
# lint:
# 	# TODO: Figure out if we want to use tapo or validate-pyproject instead.
# 	taplo lint --schema=file:///$(PWD)/hack/jsonschema/pyproject.json pyproject.toml
# 	ruff check --config=pyproject.toml --show-settings

# # run pre-commit on all files
# run-pre-commit-all:
# 	git ls-files | xargs pre-commit run --files

local-open-coverage:
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

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

fmt-python:
	git ls-files '*.py' | xargs rye run pre-commit run --files

fmt-pyproject:
	rye run pre-commit run taplo-format --all-files

# format all code using pre-commit config
fmt: fmt-python fmt-pyproject

# Lint all files in the current directory (and any subdirectories).
lint:
	rye run pre-commit run ruff --all-files
	rye run pre-commit run taplo-lint --all-files
	taplo check --schema=file:///$(PWD)/hack/jsonschema/pyproject.json pyproject.toml
