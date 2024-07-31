"""
db tasks
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import traceback

from urllib.parse import urlparse

# from sqlalchemy.engine.url import make_url
import click

from invoke import call, task
from invoke.exceptions import Failure

from tasks.utils import get_compose_env

from .utils import COLOR_CAUTION, COLOR_DANGER, COLOR_SUCCESS


# from tasks.core import clean, execute_sql

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


@task(incrementable=["verbose"])
def get_env(ctx, loc="local", verbose=0):
    """
    Get environment vars necessary to run fastapi
    Usage: inv db.get-env
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = False

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    for key in env:
        print(f"{key}={env[key]}")


@task(incrementable=["verbose"])
def get_python_path(ctx, loc="local", verbose=0):
    """
    Get environment vars necessary to run fastapi
    Usage: inv db.get-python-path
    """
    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = 'python -c "import sys; print(sys.executable)"'
    ctx.run(_cmd)


@task(incrementable=["verbose"])
def detect_os(ctx, loc="local", verbose=0):
    """
    detect what type of os we are using
    Usage: inv db.detect-os
    """
    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    res_os = ctx.run("uname -s")
    ctx.config["run"]["env"]["OS"] = f"{res_os.stdout}"

    if ctx.config["run"]["env"]["OS"] == "Windows_NT":
        ctx.config["run"]["env"]["DETECTED_OS"] = "Windows"
    else:
        ctx.config["run"]["env"]["DETECTED_OS"] = ctx.config["run"]["env"]["OS"]

    if verbose >= 1:
        msg = f'[detect-os] Detected: {ctx.config["run"]["env"]["DETECTED_OS"]}'
        click.secho(msg, fg=COLOR_SUCCESS)

    if ctx.config["run"]["env"]["DETECTED_OS"] == "Darwin":
        ctx.config["run"]["env"]["ARCHFLAGS"] = "-arch x86_64"
        ctx.config["run"]["env"]["PKG_CONFIG_PATH"] = "/usr/local/opt/libffi/lib/pkgconfig"
        ctx.config["run"]["env"]["LDFLAGS"] = "-L/usr/local/opt/openssl/lib"
        ctx.config["run"]["env"]["CFLAGS"] = "-I/usr/local/opt/openssl/include"


@task(pre=[call(detect_os, loc="local")], incrementable=["verbose"])
def autogen(ctx, loc="local", verbose=0, clean=False, dry_run=True, comment=""):
    """
    Autogenerate alembic migration scripts
    Usage: inv db.autogen
    """
    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    if verbose >= 1:
        msg = "[autogen] override env vars 'SERVER_NAME' and 'SERVER_HOST' - We don't want to mess w/ '.env.dist' for this situation"
        click.secho(msg, fg=COLOR_SUCCESS)

    if clean:
        if dry_run:
            msg = "[autogen] dry-run mode enabled"
            click.secho(msg, fg=COLOR_CAUTION)
        if verbose >= 1:
            msg = "[autogen] removing existing db first along with items in migration folder"
            click.secho(msg, fg=COLOR_SUCCESS)

        # TODO: 2/10/2020 # Add code to remove both the .db file and the files inside of goob_ai/migrations/*.py
        if ctx.config["run"]["env"]["TESTING"] and ctx.config["run"]["env"]["TEST_DATABASE_URL"]:
            DATABASE_URL = ctx.config["run"]["env"]["TEST_DATABASE_URL"]
        else:
            DATABASE_URL = ctx.config["run"]["env"]["DATABASE_URL"]

        # get file name, eg dev.db
        dbfile = urlparse(DATABASE_URL).path.replace("/", "")

        # if it exists, nuke it
        if os.path.isfile(dbfile):
            if verbose >= 1:
                _msg = f"[autogen] deleting file '{dbfile}'"
                click.secho(_msg, fg=COLOR_SUCCESS)

            if not dry_run:
                ctx.run(f"rm -fv {dbfile}")
            else:
                _msg = f"[autogen] (dry-run) would run rm -fv {dbfile}"
                click.secho(_msg, fg=COLOR_CAUTION)

        if versions := glob.glob("goob_ai/migrations/versions/*.py"):
            for i in versions:
                if verbose >= 1:
                    _msg = f"[autogen] git deleting file '{i}'"
                    click.secho(_msg, fg=COLOR_SUCCESS)
                if not dry_run:
                    ctx.run(f"git rm --force {i}")
                else:
                    _msg = f"[autogen] (dry-run) would run git rm -v {i}"
                    click.secho(_msg, fg=COLOR_CAUTION)

    if verbose >= 2:
        _msg = f'ctx.config.run.env.TESTING: {ctx.config["run"]["env"]["TESTING"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)
        _msg = f'ctx.config.run.env.TEST_DATABASE_URL: {ctx.config["run"]["env"]["TEST_DATABASE_URL"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)
        _msg = f'ctx.config.run.env.DATABASE_URL: {ctx.config["run"]["env"]["DATABASE_URL"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)

    # # override CI_IMAGE value
    # ctx.config["run"]["env"]["SERVER_NAME"] = "localhost:8000"
    # ctx.config["run"]["env"]["SERVER_HOST"] = "http://localhost:8000"
    # ctx.config["run"]["env"]["BETTER_EXCEPTIONS"] = "1"

    if not comment:
        _crud_models_cmd = r"""
grep "," goob_ai/api/crud/__init__.py | grep -v "^#" | tr ',' '\n' | xargs
"""
        res = ctx.run(_crud_models_cmd)
        # _cmd = "alembic revision --autogenerate -m 'Initial: {comment}'".format(comment=comment)
        _cmd = fr"alembic revision --autogenerate -m 'Initial: {res.stdout.rstrip()}'"
    else:
        _cmd = f"alembic revision --autogenerate -m 'Initial: {comment}'"

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    if not dry_run:
        ctx.run(_cmd)
    else:
        _msg = f"[autogen] (dry-run) would run -> {_cmd}"
        click.secho(_msg, fg=COLOR_CAUTION)


@task(pre=[call(detect_os, loc="local")], incrementable=["verbose"])
def alembic(ctx, loc="local", verbose=0, clean=False, dry_run=True, comment="", run=None):
    """
    Invoke alembic commands <OPTIONS> [upgrade, autogenerate, show, history, status, downgrade]

    Examples:
        # upgrade migrations
        $ invoke db.alembic --run=upgrade -v

        # autogenerate
        $ invoke db.alembic --run=autogenerate -v

        # show
        $ invoke db.alembic --run=show -v

        # history
        $ invoke db.alembic --run=history -v

        # history
        $ invoke db.alembic --run=history -v

        # status
        $ invoke db.alembic --run=status -v

        # downgrade
        $ invoke db.alembic --run=downgrade -v
    """

    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    if verbose >= 1:
        msg = """
[alembic] override env vars 'SERVER_NAME' and 'SERVER_HOST' - We don't want to mess w/ '.env.dist' for this situation
"""
        click.secho(msg, fg=COLOR_SUCCESS)

    if clean:
        if dry_run:
            msg = "[alembic] dry-run mode enabled"
            click.secho(msg, fg=COLOR_CAUTION)
        if verbose >= 1:
            msg = "[alembic] removing existing db first along with items in migration folder"
            click.secho(msg, fg=COLOR_SUCCESS)

        if ctx.config["run"]["env"]["TESTING"] and ctx.config["run"]["env"]["TEST_DATABASE_URL"]:
            DATABASE_URL = ctx.config["run"]["env"]["TEST_DATABASE_URL"]
        else:
            DATABASE_URL = ctx.config["run"]["env"]["DATABASE_URL"]

        # get file name, eg dev.db
        dbfile = urlparse(DATABASE_URL).path.replace("/", "")

        # if it exists, nuke it
        if os.path.isfile(dbfile):
            if verbose >= 1:
                _msg = f"[alembic] deleting file '{dbfile}'"
                click.secho(_msg, fg=COLOR_SUCCESS)

            if not dry_run:
                ctx.run(f"rm -fv {dbfile}")
            else:
                _msg = f"[alembic] (dry-run) would run rm -fv {dbfile}"
                click.secho(_msg, fg=COLOR_CAUTION)

        if versions := glob.glob("goob_ai/migrations/versions/*.py"):
            for i in versions:
                if verbose >= 1:
                    _msg = f"[alembic] git deleting file '{i}'"
                    click.secho(_msg, fg=COLOR_SUCCESS)
                if not dry_run:
                    ctx.run(f"git rm --force {i}")
                else:
                    _msg = f"[alembic] (dry-run) would run git rm -v {i}"
                    click.secho(_msg, fg=COLOR_CAUTION)

    if verbose >= 2:
        _msg = f'ctx.config.run.env.TESTING: {ctx.config["run"]["env"]["TESTING"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)
        _msg = f'ctx.config.run.env.TEST_DATABASE_URL: {ctx.config["run"]["env"]["TEST_DATABASE_URL"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)
        _msg = f'ctx.config.run.env.DATABASE_URL: {ctx.config["run"]["env"]["DATABASE_URL"]}'
        click.secho(_msg, fg=COLOR_SUCCESS)

    _cmd = ""

    # upgrade, autogenerate, show, history, status, downgrade
    try:
        if run == "upgrade":
            _cmd = "echo upgrade"
        elif run == "autogenerate":
            _crud_models_cmd = r"""grep "," goob_ai/api/crud/__init__.py | grep -v "^#" | tr ',' '\n' | xargs
"""
            res = ctx.run(_crud_models_cmd)
            _cmd = fr"alembic revision --autogenerate -m 'Initial: {res.stdout.rstrip()}'"
        elif run == "show":
            _cmd = "alembic show"
        elif run == "history":
            _cmd = "alembic history"
        elif run == "status":
            _cmd = "alembic status"
        elif run == "downgrade":
            _cmd = "alembic downgrade -1"
        else:
            _msg = "Invalid alemic command specified."
            click.secho(_msg, fg=COLOR_DANGER)
    except Failure:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(f"Ran {sys._getframe().f_code.co_name}| exc_type={exc_type}")
        logger.debug(f"Ran {sys._getframe().f_code.co_name}| exc_value={exc_value}")
        traceback.print_tb(exc_traceback)

    ####################################################
    # Now actually run the command
    ####################################################
    if verbose >= 1:
        msg = """
CMD: {}
""".format(
            _cmd
        )
        click.secho(msg, fg=COLOR_SUCCESS)

    if not dry_run:
        ctx.run(_cmd)
    else:
        _msg = f"[autogen] (dry-run) would run -> {_cmd}"
        click.secho(_msg, fg=COLOR_CAUTION)
