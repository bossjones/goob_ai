"""goob_ai.settings_validator"""
# type: ignore
from __future__ import annotations

from collections import namedtuple
import netrc
import pathlib
import platform
import sys
from typing import Dict, Union

from rich.console import Console

from goob_ai.aio_settings import aiosettings
from goob_ai.utils.file_functions import is_a_symlink, is_directory, is_file

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "python_version",
        "python_platform",
        "token",
        "prefix",
        "discord_admin_user_id",
        "discord_server_id",
        "discord_token",
        "dropbox_cerebro_app_key",
        "dropbox_cerebro_app_secret",
        "dropbox_cerebro_token",
        "tweetpik_authorization",
        "gallery_dl_config_dir",
        "yt_dl_config_dir",
        "gallery_dl_homedir_config",
        "netrc_config_filepath",
        "aiomonitor_monitor_port",
        "aiomonitor_monitor_host",
        "global_config_file",
        "global_config_dir",
        "global_models_dir",
        "global_autoscan_dir",
        "global_converted_ckpts_dir",
        "gallery_dl_config_dot_json",
        "yt_dlp_config",
        "netrc_config",
        "gallery_dl_cookie_files",
    ],
)


def get_python_platform() -> str:
    """_summary_

    Returns:
        _type_: _description_
    """

    return platform.platform()


def get_gallery_dl_config_dir() -> str:
    """_summary_

    Returns:
        _type_: _description_
    """
    state_is_dir = is_directory(aiosettings.gallery_dl_config_dir)
    state_exists = pathlib.Path(aiosettings.gallery_dl_config_dir).exists()
    return f"{aiosettings.gallery_dl_config_dir} | exists: {state_exists} | is_dir: {state_is_dir}"


def get_is_dir_and_exists(path: Union[str, pathlib.PosixPath]) -> str:
    """_summary_

    Returns:
        _type_: _description_
    """
    if isinstance(path, pathlib.PosixPath):
        path = f"{path}"

    state_is_dir = is_directory(path)
    state_exists = pathlib.Path(path).exists()
    return f"{path} | exists: {state_exists} | is_dir: {state_is_dir}"


def get_is_file_and_exists(path: Union[str, pathlib.PosixPath]) -> str:
    """_summary_

    Returns:
        _type_: _description_
    """
    if isinstance(path, pathlib.PosixPath):
        path = f"{path}"
    state_is_file = is_file(path)
    state_exists = pathlib.Path(path).exists()
    return f"{path} | exists: {state_exists} | is_file: {state_is_file}"


def get_is_symlink_and_exists(path: Union[str, pathlib.PosixPath], is_symlink_to: Union[str, pathlib.PosixPath]) -> str:
    """_summary_

    Returns:
        _type_: _description_
    """
    if isinstance(path, pathlib.PosixPath):
        path = f"{path}"
    state_is_symlink = is_a_symlink(path)
    state_exists = pathlib.Path(path).exists()
    try:
        state_symlink_to = pathlib.Path(path).resolve().parent.samefile(pathlib.Path(is_symlink_to).resolve().parent)
    except FileNotFoundError:
        state_symlink_to = False

    # state_symlink_to = bool(pathlib.Path(path).readlink() == pathlib.Path(is_symlink_to))
    return (
        f"{path} | exists: {state_exists} | is_symlink: {state_is_symlink} | is_symlinked_correctly: {state_symlink_to}"
    )


def get_netrc_is_valid(path: Union[str, pathlib.PosixPath]) -> bool:
    """_summary_

    Args:
        path (Union[str, pathlib.PosixPath]): _description_

    Returns:
        bool: _description_
    """
    netrc_local = netrc.netrc(path)
    return bool(netrc_local.hosts)


def get_gallery_dl_cookie_files_exists(cookie_file_data: Dict[str, pathlib.PosixPath]) -> str:
    lines = [
        f"    Cookie Name: {cookie_name} | Cookie Path: {cookie_path} | Cookie exists: {cookie_path.exists()}"
        for cookie_name, cookie_path in cookie_file_data.items()
    ]
    return "\n".join(lines)


# from collect_env.py in pytorch
def get_env_info():
    """_summary_

    Returns:
        _type_: _description_
    """

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        python_version=f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        python_platform=get_python_platform(),
        token=aiosettings.token,
        prefix=aiosettings.prefix,
        discord_admin_user_id=aiosettings.discord_admin_user_id,
        discord_server_id=aiosettings.discord_server_id,
        discord_token=aiosettings.discord_token,
        dropbox_cerebro_app_key=aiosettings.dropbox_cerebro_app_key,
        dropbox_cerebro_app_secret=aiosettings.dropbox_cerebro_app_secret,
        dropbox_cerebro_token=aiosettings.dropbox_cerebro_token,
        tweetpik_authorization=aiosettings.tweetpik_authorization,
        gallery_dl_config_dir=get_gallery_dl_config_dir(),
        yt_dl_config_dir=get_is_dir_and_exists(aiosettings.yt_dl_config_dir),
        gallery_dl_homedir_config=get_is_file_and_exists(aiosettings.gallery_dl_homedir_config),
        netrc_config_filepath=get_is_file_and_exists(aiosettings.netrc_config_filepath),
        aiomonitor_monitor_port=aiosettings.monitor_port,
        aiomonitor_monitor_host=aiosettings.monitor_host,
        global_config_file=get_is_file_and_exists(aiosettings.global_config_file),
        global_config_dir=get_is_dir_and_exists(aiosettings.global_config_dir),
        global_models_dir=get_is_dir_and_exists(aiosettings.global_models_dir),
        global_autoscan_dir=get_is_dir_and_exists(aiosettings.global_autoscan_dir),
        global_converted_ckpts_dir=get_is_dir_and_exists(aiosettings.global_converted_ckpts_dir),
        gallery_dl_config_dot_json=get_is_symlink_and_exists(
            aiosettings.gallery_dl_config_dot_json, aiosettings.gallery_dl_homedir_config
        ),
        yt_dlp_config=get_is_dir_and_exists(aiosettings.yt_dlp_config),
        netrc_config=get_netrc_is_valid(aiosettings.netrc_config),
        gallery_dl_cookie_files=get_gallery_dl_cookie_files_exists(aiosettings.gallery_dl_cookie_files),
    )


env_info_fmt = """
Python version: {python_version}
Python platform: {python_platform}

Discord Token: {token}
Discord Chatbot Prefix: {prefix}
Discord Admin User ID: {discord_admin_user_id}
Discord Server Id: {discord_server_id}
Discord Token: {discord_token}


Dropbox App Key: {dropbox_cerebro_app_key}
Dropbox App Secret: {dropbox_cerebro_app_secret}
Dropbox App Token: {dropbox_cerebro_token}

Tweetpik Authorization: {tweetpik_authorization}

Yt-dlp Config Dir: {yt_dl_config_dir}
Yt-dlp Config File Path: {yt_dlp_config}

Gallery-dl config dir: {gallery_dl_config_dir}
Gallery-dl Homedir Config: {gallery_dl_homedir_config}
Gallery-dl json config exists : {gallery_dl_config_dot_json}
Gallery-dl cookie files:
{gallery_dl_cookie_files}

Valid Netrc: {netrc_config_filepath}
Netrc: {netrc_config}

AioMonitor port: {aiomonitor_monitor_port}
AioMonitor host: {aiomonitor_monitor_host}

GoobBot Global Config Dir: {global_config_dir}
GoobBot Global Config File: {global_config_file}
GoobBot Global Models Dir: {global_models_dir}
GoobBot Global Autoscan Dir: {global_autoscan_dir}
GoobBot Global Checkpoints Dir: {global_converted_ckpts_dir}
""".strip()


# TODO: enable this
def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        """_summary_

        Args:
            dct (_type_): _description_
            replacement (str, optional): _description_. Defaults to "Could not collect".

        Returns:
            _type_: _description_
        """
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        """_summary_

        Args:
            dct (_type_): _description_
            true (str, optional): _description_. Defaults to "Yes".
            false (str, optional): _description_. Defaults to "No".

        Returns:
            _type_: _description_
        """
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        """_summary_

        Args:
            text (_type_): _description_
            tag (str, optional): _description_. Defaults to "[prepend]".

        Returns:
            _type_: _description_
        """
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text, replacement="No relevant packages"):
        """_summary_

        Args:
            text (_type_): _description_
            replacement (str, optional): _description_. Defaults to "No relevant packages".

        Returns:
            _type_: _description_
        """
        return replacement if text is not None and len(text) == 0 else text

    def maybe_start_on_next_line(string):
        """_summary_

        Args:
            string (_type_): _description_

        Returns:
            _type_: _description_
        """
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # # If nvidia_gpu_models is multiline, start on the next line
    # mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(
    #     envinfo.nvidia_gpu_models,
    # )

    # # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    # dynamic_cuda_fields = [
    #     "cuda_runtime_version",
    #     "nvidia_gpu_models",
    #     "nvidia_driver_version",
    # ]
    # all_cuda_fields = dynamic_cuda_fields + ["cudnn_version"]
    # all_dynamic_cuda_fields_missing = all(
    #     mutable_dict[field] is None for field in dynamic_cuda_fields
    # )
    # if (
    #     TORCH_AVAILABLE
    #     and not torch.cuda.is_available()
    #     and all_dynamic_cuda_fields_missing
    # ):
    #     for field in all_cuda_fields:
    #         mutable_dict[field] = "No CUDA"
    #     if envinfo.cuda_compiled_version is None:
    #         mutable_dict["cuda_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # # If either of these are '', replace with 'No relevant packages'
    # mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    # mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # # Tag conda and pip packages with a prefix
    # # If they were previously None, they'll show up as ie '[conda] Could not collect'
    # if mutable_dict["pip_packages"]:
    #     mutable_dict["pip_packages"] = prepend(
    #         mutable_dict["pip_packages"],
    #         "[{}] ".format(envinfo.pip_version),
    #     )
    # if mutable_dict["conda_packages"]:
    #     mutable_dict["conda_packages"] = prepend(
    #         mutable_dict["conda_packages"],
    #         "[conda] ",
    #     )
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    return pretty_str(get_env_info())


def get_rich_pretty_env_info() -> None:
    """_summary_"""
    console = Console()
    env_data = get_pretty_env_info()
    console.print(env_data)
