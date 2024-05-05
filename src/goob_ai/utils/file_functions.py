"""goob_ai.file_functions"""
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalIterable=false

from __future__ import annotations

import glob
import json
import logging
import os
import pathlib
import string
import sys
from typing import List, Tuple, Union

import aiofiles
import pandas as pd
import rich
from rich.console import Console
from rich.table import Table

from goob_ai.bot_logger import get_logger
from goob_ai.constants import (
    FIFTY_THOUSAND,
    FIVE_HUNDRED_THOUSAND,
    MAX_BYTES_UPLOAD_DISCORD,
    ONE_HUNDRED_THOUSAND,
    ONE_MILLION,
    TEN_THOUSAND,
    THIRTY_THOUSAND,
    TWENTY_THOUSAND,
)

from loguru import logger as LOGGER

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
# python convert string to safe filename
VALID_FILENAME_CHARS = f"-_.() {string.ascii_letters}{string.digits}"
CHAR_LIMIT = 255

JSON_EXTENSIONS = [".json", ".JSON"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".MP4", ".MOV"]
AUDIO_EXTENSIONS = [".mp3", ".MP3"]
GIF_EXTENSIONS = [".gif", ".GIF"]
MKV_EXTENSIONS = [".mkv", ".MKV"]
M3U8_EXTENSIONS = [".m3u8", ".M3U8"]
WEBM_EXTENSIONS = [".webm", ".WEBM"]
IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif", ".PNG", ".JPEG", ".JPG", ".GIF"]
TORCH_MODEL_EXTENSIONS = [".pth", ".PTH"]


async def aio_read_jsonfile(jsonfile: str):
    print(f" [aio_read_jsonfile] jsonfile -> {jsonfile}")
    async with aiofiles.open(jsonfile, mode="r", encoding="utf-8") as f:
        contents = await f.read()
    json_data = json.loads(contents)
    print(f" [aio_read_jsonfile] json_data -> {json_data}")
    return json_data


async def aio_json_loads(uri: str):
    return json.loads(await (await aiofiles.open(uri, mode="r")).read())


async def run_aio_json_loads(uri: str):
    return await aio_json_loads(uri=uri)


# SOURCE: https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
def sort_dir_by_mtime(dirpath: str) -> List[pathlib.Path]:
    return sorted(pathlib.Path(dirpath).iterdir(), key=os.path.getmtime)


# SOURCE: https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
def sort_dir_by_ctime(dirpath: str) -> List[pathlib.Path]:
    return sorted(pathlib.Path(dirpath).iterdir(), key=os.path.getctime)


def get_all_media_files_to_upload(tmpdirname: str):
    # top level function that grabs all media files
    tree_list = tree(pathlib.Path(f"{tmpdirname}"))
    rich.print(tree_list)

    file_to_upload_list = [f"{p}" for p in tree_list]
    LOGGER.debug(f"get_all_media_files_to_upload -> file_to_upload_list = {file_to_upload_list}")
    rich.print(file_to_upload_list)

    return filter_media(file_to_upload_list)


def filter_pth(working_dir: List[str]) -> List[str]:
    return [
        f
        for f in working_dir
        if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in TORCH_MODEL_EXTENSIONS
    ]


def filter_json(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in JSON_EXTENSIONS
    ]


def rename_without_cachebuster(working_dir: List[str]) -> List[str]:
    working_dir_only = []
    for f in working_dir:
        if ("?updatedAt" in f"{f}") and (pathlib.Path(f"{f}").is_file()):
            orig = pathlib.Path(f"{f}").absolute()
            # tweetpik now adds cache buster, lets work around it
            without_cb = f"{orig}".split("?updatedAt")[0]
            orig.rename(f"{without_cb}")
            working_dir_only.append(f"{without_cb}")
    return working_dir_only


def filter_videos(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in VIDEO_EXTENSIONS
    ]


def filter_audio(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in AUDIO_EXTENSIONS
    ]


def filter_gif(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in GIF_EXTENSIONS
    ]


def filter_mkv(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in MKV_EXTENSIONS
    ]


def filter_m3u8(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in M3U8_EXTENSIONS
    ]


def filter_webm(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in WEBM_EXTENSIONS
    ]


def filter_images(working_dir: List[str]) -> List[str]:
    return [
        f for f in working_dir if (pathlib.Path(f"{f}").is_file()) and pathlib.Path(f"{f}").suffix in IMAGE_EXTENSIONS
    ]


def filter_pdfs(working_dir: List[str]) -> List[str]:
    return [
        f
        for f in working_dir
        if (pathlib.Path(f"{f}").is_file())
        and pathlib.Path(f"{f}").suffix
        in [
            ".pdf",
            ".PDF",
        ]
    ]


def filter_media(working_dir: List[str]) -> List[str]:
    imgs = filter_images(working_dir)
    videos = filter_videos(working_dir)
    return imgs + videos


def filter_pdf(working_dir: List[str]) -> List[str]:
    return filter_pdfs(working_dir)


def get_dataframe_from_csv(filename: str, return_parent_folder_name: bool = False) -> pd.core.frame.DataFrame:
    """Open csv files and return a dataframe from pandas

    Args:
        filename (str): path to file
    """
    src = pathlib.Path(f"{filename}").resolve()
    df = pd.read_csv(f"{src}")

    # import bpdb
    # bpdb.set_trace()

    return (df, f"{src.parent.stem}") if return_parent_folder_name else df


def sort_dataframe(df: pd.core.frame.DataFrame, columns: list = None, ascending: Tuple = ()) -> pd.core.frame.DataFrame:
    """Return dataframe sorted via columns

    Args:
        df (pd.core.frame.DataFrame): existing dataframe
        columns (list, optional): [description]. Defaults to []. Eg. ["Total Followers", "Total Likes", "Total Comments", "ERDay", "ERpost"]
        ascending (Tuple, optional): [description]. Defaults to (). Eg. (False, False, False, False, False)

    Returns:
        pd.core.frame.DataFrame: [description]
    """
    if columns is None:
        columns = []
    df = df.sort_values(columns, ascending=ascending)
    return df


def rich_format_followers(val: int) -> str:
    """Given a arbritary int, return a 'rich' string formatting

    Args:
        val (int): eg. followers = 4366347347457

    Returns:
        str: [description] eg. "[bold bright_yellow]4366347347457[/bold bright_yellow]"
    """

    if val > ONE_MILLION:
        return f"[bold bright_yellow]{val}[/bold bright_yellow]"
    elif FIVE_HUNDRED_THOUSAND < val < ONE_MILLION:
        return f"[bold dark_orange]{val}[/bold dark_orange]"
    elif ONE_HUNDRED_THOUSAND < val < FIVE_HUNDRED_THOUSAND:
        return f"[bold orange_red1]{val}[/bold orange_red1]"

    elif FIFTY_THOUSAND < val < ONE_HUNDRED_THOUSAND:
        return f"[bold dodger_blue2]{val}[/bold dodger_blue2]"
    elif THIRTY_THOUSAND < val < FIFTY_THOUSAND:
        return f"[bold purple3]{val}[/bold purple3]"
    elif TWENTY_THOUSAND < val < THIRTY_THOUSAND:
        return f"[bold rosy_brown]{val}[/bold rosy_brown]"
    elif TEN_THOUSAND < val < TWENTY_THOUSAND:
        return f"[bold green]{val}[/bold green]"
    else:
        return f"[bold bright_white]{val}[/bold bright_white]"


def rich_likes_or_comments(val: int) -> str:
    """Given a arbritary int, return a 'rich' string formatting

    Args:
        val (int): eg. followers = 4366347347457

    Returns:
        str: [description] eg. "[bold bright_yellow]4366347347457[/bold bright_yellow]"
    """

    if TEN_THOUSAND >= val:
        return f"[bold bright_yellow]{val}[/bold bright_yellow]"
    elif FIFTY_THOUSAND < val < ONE_HUNDRED_THOUSAND:
        return f"[bold dodger_blue2]{val}[/bold dodger_blue2]"
    elif THIRTY_THOUSAND < val < FIFTY_THOUSAND:
        return f"[bold purple3]{val}[/bold purple3]"
    elif TWENTY_THOUSAND < val < THIRTY_THOUSAND:
        return f"[bold rosy_brown]{val}[/bold rosy_brown]"
    elif TEN_THOUSAND < val < TWENTY_THOUSAND:
        return f"[bold green]{val}[/bold green]"
    else:
        return f"[bold bright_white]{val}[/bold bright_white]"


def rich_display_meme_pull_list(df: pd.core.frame.DataFrame):  # noqa
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Account")
    table.add_column("Social")
    table.add_column("Total Followers")
    table.add_column("Total Likes")
    table.add_column("Total Comments")
    table.add_column("Total Posts")
    table.add_column("Start Date")
    table.add_column("End Date")
    table.add_column("ERDay")
    table.add_column("ERpost")
    table.add_column("Average Likes")
    table.add_column("Average Comments")
    table.add_column("Links")

    for index, row in df.iterrows():
        account = f"[bold blue]{row['Account']}[/bold blue]"
        social = f"[bold]{row['Social']}[/bold]"
        total_followers = rich_format_followers(row["Total Followers"])
        total_likes = f"[bold]{row['Total Likes']}[/bold]"
        total_comments = f"[bold]{row['Total Comments']}[/bold]"
        total_posts = f"[bold]{row['Total Posts']}[/bold]"
        start_date = f"[bold]{row['Start Date']}[/bold]"
        end_date = f"[bold]{row['End Date']}[/bold]"
        erday = f"[bold]{row['ERDay']}[/bold]"
        erpost = f"[bold]{row['ERpost']}[/bold]"
        average_likes = f"[bold]{row['Average Likes']}[/bold]"
        average_comments = f"[bold]{row['Average Comments']}[/bold]"
        links = f"[bold]{row['Links']}[/bold]"

        table.add_row(
            account,
            social,
            total_followers,
            total_likes,
            total_comments,
            total_posts,
            start_date,
            end_date,
            erday,
            erpost,
            average_likes,
            average_comments,
            links,
        )

    console.print(table)


def rich_display_popstars_analytics(df: pd.core.frame.DataFrame):  # noqa
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Social")
    table.add_column("Author")
    table.add_column("Url")
    table.add_column("Likes")
    table.add_column("Comments")
    table.add_column("ER")
    table.add_column("Text")
    table.add_column("Date")
    table.add_column("Media 1")

    for index, row in df.iterrows():
        social = f"[bold]{row['Social']}[/bold]"
        author = f"[bold]{row['Author']}[/bold]"
        url = f"[bold]{row['Url']}[/bold]"
        likes = f"[bold]{rich_likes_or_comments(row['Likes'])}[/bold]"
        comments = f"[bold]{rich_likes_or_comments(row['Comments'])}[/bold]"
        er = f"[bold]{row['ER']}[/bold]"
        text = f"[bold]{row['Text']}[/bold]"
        date = f"[bold]{row['Date']}[/bold]"
        media = f"[bold]{row['Media 1']}[/bold]"

        table.add_row(social, author, url, likes, comments, er, text, date, media)

    console.print(table)


# # >>> import glob
# # >>> json_files = glob.glob(f"{metadata_path}/*.json")
# # >>> json_files


def glob_file_by_extension(working_dir: List[str], extension: str = "*.mp4", recursive: bool = False) -> List[str]:
    print(f"Searching dir -> {working_dir}/{extension}")

    if recursive:
        # NOTE: When recursive is set True "**"" followed by path separator('./**/') will match any files or directories.
        expression = f"{working_dir}/**/{extension}"
    else:
        expression = f"{working_dir}/{extension}"
    return glob.glob(expression, recursive=recursive)


def print_and_append(dir_listing: list, tree_str: str, silent: bool = False) -> None:
    if not silent:
        print(tree_str)
    dir_listing.append(tree_str)


def tree(directory: Union[pathlib.PosixPath, pathlib.Path], silent: bool = False) -> List[pathlib.PosixPath]:
    """"""
    # from ffmpeg_tools import fileobject
    file_system: List[pathlib.Path]
    file_system = []
    _tree = []
    print_and_append(_tree, f"+ {directory}", silent=silent)
    for path in sorted(directory.rglob("*")):
        file_system.append(pathlib.Path(f"{path.resolve()}"))
        depth = len(path.relative_to(directory).parts)
        spacer = "    " * depth
        print_and_append(_tree, f"{spacer}+ {path.name}", silent=silent)

    return sorted(file_system, key=os.path.getmtime)


# SOURCE: https://python.hotexamples.com/site/file?hash=0xda3708e60cd1ddb3012abd7dba205f48214aee7366f452e93807887c6a04db42&fullName=spring_cleaning.py&project=pambot/SpringCleaning
def format_size(a_file: str):
    if a_file > 1024**3:
        return "{:.0f} GB".format(a_file / float(2**30))
    elif a_file > 1024**2:
        return "{:.0f} MB".format(a_file / float(2**20))
    elif a_file > 1024:
        return "{:.0f} KB".format(a_file / float(2**10))
    else:
        return "{:.0f} B".format(a_file)


async def aiowrite_file(data: str, dl_dir: str = "./", fname: str = "", ext: str = ""):
    p_dl_dir = pathlib.Path(dl_dir)
    full_path_dl_dir = f"{p_dl_dir.absolute()}"
    p_new = pathlib.Path(f"{full_path_dl_dir}/{fname}.{ext}")
    LOGGER.debug(f"Writing to {p_new.absolute()}")
    async with aiofiles.open(f"{p_new.absolute()}", mode="w") as f:
        await f.write(data)


async def aioread_file(data: str, dl_dir: str = "./", fname: str = "", ext: str = ""):
    p_dl_dir = pathlib.Path(dl_dir)
    full_path_dl_dir = f"{p_dl_dir.absolute()}"
    p_new = pathlib.Path(f"{full_path_dl_dir}/{fname}.{ext}")
    LOGGER.debug(f"Writing to {p_new.absolute()}")
    async with aiofiles.open(f"{p_new.absolute()}", mode="r") as f:
        await f.read(data)


def check_file_size(a_file: str) -> Tuple[bool, str]:
    p = pathlib.Path(a_file)
    file_size = p.stat().st_size
    LOGGER.debug(f"File: {p} | Size(bytes): {file_size} | Size(type): {type(file_size)}")
    check = file_size > MAX_BYTES_UPLOAD_DISCORD
    msg = f"Is file size greater than {MAX_BYTES_UPLOAD_DISCORD}: {check}"
    LOGGER.debug(msg)
    return check, msg


# ------------------------------------------------------------
# NOTE: MOVE THIS TO A FILE UTILITIES LIBRARY
# ------------------------------------------------------------
# SOURCE: https://github.com/tgbugs/pyontutils/blob/05dc32b092b015233f4a6cefa6c157577d029a40/ilxutils/tools.py
def is_file(path: str):
    """Check if path contains a file

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return bool(pathlib.Path(path).is_file())


def is_directory(path: str):
    """Check if path contains a dir

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    return bool(pathlib.Path(path).is_dir())


def is_a_symlink(path: str):
    """Check if path contains a dir

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    return bool(pathlib.Path(path).is_symlink())


def expand_path_str(path: str) -> pathlib.PosixPath:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        pathlib.PosixPath: _description_
    """
    return pathlib.Path(tilda(path))


def tilda(obj):
    """wrapper for linux ~/ shell notation

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(obj, list):
        return [str(pathlib.Path(o).expanduser()) if isinstance(o, str) else o for o in obj]
    elif isinstance(obj, str):
        return str(pathlib.Path(obj).expanduser())
    else:
        return obj


def fix_path(path: str):
    """Automatically convert path to fully qualifies file uri.

    Args:
        path (_type_): _description_
    """

    def __fix_path(path):
        if not isinstance(path, str):
            return path
        elif "~" == path[0]:
            tilda_fixed_path = tilda(path)
            if is_file(tilda_fixed_path):
                return tilda_fixed_path
            else:
                exit(path, ": does not exit.")
        elif is_file(pathlib.Path.home() / path):
            return str(pathlib.Path().home() / path)
        elif is_directory(pathlib.Path.home() / path):
            return str(pathlib.Path().home() / path)
        else:
            return path

    if isinstance(path, str):
        return __fix_path(path)
    elif isinstance(path, list):
        return [__fix_path(p) for p in path]
    else:
        return path


# smoke tests
