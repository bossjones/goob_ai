#!/usr/bin/env python
"""goob_ai.shell"""
# pylint: disable=consider-using-with

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
import time

from asyncio.subprocess import Process
from pathlib import Path
from typing import List, Tuple, Union

import uritools

from codetiming import Timer


HOME_PATH = os.environ.get("HOME")


async def _aio_run_process_and_communicate(cmd: List[str], cwd: Union[str, None] = None):
    """_summary_

    Args:
        cmd (List[str]): _description_

    Returns:
        _type_: _description_
    """
    program = cmd
    process: Process = await asyncio.create_subprocess_exec(*program, stdout=asyncio.subprocess.PIPE, cwd=cwd)
    print(f"Process pid is: {process.pid}")
    stdout, stderr = await process.communicate()
    return stdout.decode("utf-8").strip()


def _stat_y_file(fname: str, env: dict = None, cwd: Union[str, None] = None) -> str:
    # """Get the timestamp of a file."""
    if env is None:
        env = {}
    cmd_arg_without_str_fmt = """stat -c %y {fname}""".format(fname=fname)
    print(f"cmd_arg_without_str_fmt={cmd_arg_without_str_fmt}")

    cmd_arg = rf"""{cmd_arg_without_str_fmt}"""
    print(f"cmd_arg={cmd_arg}")
    try:
        result = subprocess.run(
            [cmd_arg],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            shell=True,
        )
        timestamp = result.stdout.replace("\n", "")
        print(f"timestamp={timestamp}")
    except subprocess.CalledProcessError as e:
        print(e.output)
    return timestamp


def _popen(cmd_arg: Tuple, env: dict = None, cwd: Union[str, None] = None):
    """_summary_

    Args:
        cmd_arg (Tuple): _description_
        env (dict, optional): _description_. Defaults to None.
        cwd (Union[str, None], optional): _description_. Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    if env is None:
        env = {}
    with open("/dev/null") as devnull:
        cmd = subprocess.Popen(cmd_arg, stdout=subprocess.PIPE, stderr=devnull, env=env, cwd=cwd)
        retval = cmd.stdout.read().strip()
        err = cmd.wait()
        cmd.stdout.close()
    if err:
        raise RuntimeError(f"Failed to close {cmd_arg} stream")
    return retval


def _popen_communicate(cmd_arg: Tuple, env: dict = None, cwd: Union[str, None] = None):
    """_summary_

    Args:
        cmd_arg (Tuple): _description_
        env (dict, optional): _description_. Defaults to None.
        cwd (Union[str, None], optional): _description_. Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    if env is None:
        env = {}
    devnull = open("/dev/null")
    cmd = subprocess.Popen(cmd_arg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)

    try:
        time.sleep(0.2)
        retval = cmd.stdout.read().strip()
        err = cmd.wait()

    finally:
        cmd.terminate()
        try:
            outs, _ = cmd.communicate(timeout=0.2)
            print("== subprocess exited with rc =", cmd.returncode)
            print(outs.decode("utf-8"))
        except subprocess.TimeoutExpired:
            print("subprocess did not terminate in time")

    if err:
        raise RuntimeError(f"Failed to close {cmd_arg} stream")
    return retval


# SOURCE: https://github.com/ARMmbed/mbed-cli/blob/f168237fabd0e32edcb48e214fc6ce2250046ab3/test/util.py
# Process execution
class ProcessException(Exception):
    pass


class ShellConsole:  # pylint: disable=too-few-public-methods
    quiet = False

    @classmethod
    def message(cls, str_format, *args):
        """_summary_

        Args:
            str_format (_type_): _description_
        """
        if cls.quiet:
            return

        if args:
            print(str_format % args)
        else:
            print(str_format)

        # Flush so that messages are printed at the right time
        # as we use many subprocesses.
        sys.stdout.flush()


def pquery(command: Union[str, list], stdin: bool = None, **kwargs):
    """_summary_

    Args:
        command (Union[str, list]): _description_
        stdin (bool, optional): _description_. Defaults to None.

    Raises:
        ProcessException: _description_

    Returns:
        _type_: _description_
    """
    # SOURCE: https://github.com/ARMmbed/mbed-cli/blob/f168237fabd0e32edcb48e214fc6ce2250046ab3/test/util.py
    # Example:
    if type(command) == list:
        print(" ".join(command))
    elif type(command) == str:
        command = command.split(" ")
        print(f"cmd: {command}")
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        stdout, _ = proc.communicate(stdin)
    except Exception:
        print("[fail] cmd={command}, ret={proc.returncode}")
        raise

    if proc.returncode != 0:
        raise ProcessException(proc.returncode)

    return stdout.decode("utf-8")


def _popen_stdout(cmd_arg: str, cwd: Union[str, None] = None):
    """_summary_

    Args:
        cmd_arg (str): _description_
        cwd (Union[str, None], optional): _description_. Defaults to None.
    """
    # if passing a single string, either shell mut be True or else the string must simply name the program to be executed without specifying any arguments
    cmd = subprocess.Popen(
        cmd_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        bufsize=4096,
        shell=True,
    )
    ShellConsole.message(f"BEGIN: {cmd_arg}")
    # output, err = cmd.communicate()

    for line in iter(cmd.stdout.readline, b""):
        # Print line
        _line = line.rstrip()
        ShellConsole.message(f'>>> {_line.decode("utf-8")}')

    ShellConsole.message(f"END: {cmd_arg}")
    # subprocess.CompletedProcess(args=cmd_arg, returncode=0)


def _popen_stdout_lock(cmd_arg: str, cwd: Union[str, None] = None):
    """_summary_

    Args:
        cmd_arg (str): _description_
        cwd (Union[str, None], optional): _description_. Defaults to None.
    """
    # if passing a single string, either shell mut be True or else the string must simply name the program to be executed without specifying any arguments
    with subprocess.Popen(
        cmd_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        bufsize=4096,
        shell=True,
    ) as cmd:
        ShellConsole.message(f"BEGIN: {cmd_arg}")
        # output, err = cmd.communicate()

        for line in iter(cmd.stdout.readline, b""):
            # Print line
            _line = line.rstrip()
            ShellConsole.message(f'>>> {_line.decode("utf-8")}')

        ShellConsole.message(f"END: {cmd_arg}")
        subprocess.CompletedProcess(args=cmd_arg, returncode=0)


async def run_coroutine_subprocess(cmd: str, uri: str, working_dir: str = f"{pathlib.Path('./').absolute()}"):
    """_summary_

    Args:
        cmd (str): _description_
        uri (str): _description_
        working_dir (str, optional): _description_. Defaults to f"{pathlib.Path('./').absolute()}".

    Returns:
        _type_: _description_
    """
    await asyncio.sleep(0.05)

    timer = Timer(text=f"Task {__name__} elapsed time: {{:.1f}}")

    env = {}
    env |= os.environ

    dl_uri = uritools.urisplit(uri)

    result = "0"
    cmd = f"{cmd}"

    timer.start()
    process = await asyncio.create_subprocess_shell(
        cmd,
        env=env,
        cwd=working_dir,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
    )
    stdout, stderr = await process.communicate()
    result = stdout.decode("utf-8").strip()
    timer.stop()
    return result
