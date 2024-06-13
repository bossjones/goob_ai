from __future__ import annotations

import json
import os
import pathlib

from os import PathLike
from typing import List

import aiofiles
import pandas as pd

from goob_ai.services.chroma_service import CHROMA_PATH, DATA_PATH
from goob_ai.utils.file_functions import (
    aio_json_loads,
    aio_read_jsonfile,
    aioread_file,
    aiowrite_file,
    check_file_size,
    expand_path_str,
    filter_audio,
    filter_gif,
    filter_images,
    filter_json,
    filter_m3u8,
    filter_media,
    filter_mkv,
    filter_pdf,
    filter_pdfs,
    filter_pth,
    filter_videos,
    filter_webm,
    fix_path,
    format_size,
    get_all_media_files_to_upload,
    get_dataframe_from_csv,
    glob_file_by_extension,
    is_a_symlink,
    is_directory,
    is_file,
    print_and_append,
    rename_without_cachebuster,
    rich_display_meme_pull_list,
    rich_display_popstars_analytics,
    rich_format_followers,
    rich_likes_or_comments,
    run_aio_json_loads,
    sort_dataframe,
    sort_dir_by_ctime,
    sort_dir_by_mtime,
    tilda,
    tree,
)

import pytest


HERE = os.path.dirname(__file__)

# DATA_PATH = os.path.join(HERE, "..", "..", "src", "goob_ai", "data", "chroma", "documents")
# CHROMA_PATH = os.path.join(HERE, "..", "..","src", "goob_ai", "data", "chroma", "vectorstorage")

# ~/dev/bossjones/goob_ai/tests/utils/../../src/goob_ai/data/chroma/documents


# @pytest.mark.asyncio
# async def test_aio_read_jsonfile(mocker):
#     mock_open = mocker.patch("aiofiles.open", new_callable=mocker.AsyncMock)
#     mock_open.return_value.__aenter__.return_value.read = mocker.AsyncMock(return_value='{"key": "value"}')
#     mock_open.return_value.__aenter__.return_value.__aexit__ = mocker.AsyncMock()
#     result = await aio_read_jsonfile("test.json")
#     assert result == {"key": "value"}
#     mock_open.assert_called_once_with("test.json", mode="r", encoding="utf-8")


# @pytest.mark.asyncio
# async def test_aio_json_loads(mocker):
#     mock_open = mocker.patch("aiofiles.open", new_callable=mocker.AsyncMock)
#     mock_open.return_value.__aenter__.return_value.read.return_value = '{"key": "value"}'
#     mock_open.return_value.__aenter__.return_value.__aexit__ = mocker.AsyncMock()
#     result = await aio_json_loads("test.json")
#     assert result == {"key": "value"}
#     mock_open.assert_called_once_with("test.json", mode="r")


# @pytest.mark.asyncio
# async def test_run_aio_json_loads(mocker):
#     mock_open = mocker.patch("aiofiles.open", new_callable=mocker.AsyncMock)
#     mock_open.return_value.__aenter__.return_value.read.return_value = '{"key": "value"}'
#     result = await run_aio_json_loads("test.json")
#     assert result == {"key": "value"}
#     mock_open.assert_called_once_with("test.json", mode="r")


def test_sort_dir_by_mtime(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("file1"), pathlib.Path("file2")])
    mocker.patch("os.path.getmtime", side_effect=[2, 1])
    result = sort_dir_by_mtime("test_dir")
    assert result == [pathlib.Path("file2"), pathlib.Path("file1")]


def test_sort_dir_by_ctime(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("file1"), pathlib.Path("file2")])
    mocker.patch("os.path.getctime", side_effect=[2, 1])
    result = sort_dir_by_ctime("test_dir")
    assert result == [pathlib.Path("file2"), pathlib.Path("file1")]


def test_get_all_media_files_to_upload(mocker):
    mock_tree = mocker.patch("goob_ai.utils.file_functions.tree", return_value=["file1", "file2"])
    mock_filter_media = mocker.patch("goob_ai.utils.file_functions.filter_media", return_value=["file1"])
    result = get_all_media_files_to_upload("test_dir")
    assert result == ["file1"]
    mock_tree.assert_called_once_with(pathlib.Path("test_dir"))
    mock_filter_media.assert_called_once_with(["file1", "file2"])


def test_filter_pdfs():
    d = tree(DATA_PATH)
    result = filter_pdfs(d)
    expected = [
        "opencv-tutorial-readthedocs-io-en-latest.pdf",
        "pillow-readthedocs-io-en-latest.pdf",
        "rich-readthedocs-io-en-latest.pdf",
    ]

    for i in result:
        assert i.name in expected


# def test_filter_pth():
#     result = filter_pth(["file1.pth", "file2.txt", "file3.PTH"])
#     assert result == ["file1.pth"]


# def test_filter_json():
#     result = filter_json(["file1.json", "file2.txt", "file3.JSON"])
#     assert result == ["file1.json"]


# def test_rename_without_cachebuster(mocker):
#     mock_rename = mocker.patch("pathlib.Path.rename")
#     result = rename_without_cachebuster(["file1?updatedAt=123", "file2", "file3?updatedAt=456"])
#     assert result == ["file1", "file2"]
#     mock_rename.assert_called_once()


# def test_filter_videos():
#     result = filter_videos(["file1.mp4", "file2.txt", "file3.MP4"])
#     assert result == ["file1.mp4"]


# def test_filter_audio():
#     result = filter_audio(["file1.mp3", "file2.txt", "file3.MP3"])
#     assert result == ["file1.mp3"]


# def test_filter_gif():
#     result = filter_gif(["file1.gif", "file2.txt", "file3.GIF"])
#     assert result == ["file1.gif"]


# def test_filter_mkv():
#     result = filter_mkv(["file1.mkv", "file2.txt", "file3.MKV"])
#     assert result == ["file1.mkv"]


# def test_filter_m3u8():
#     result = filter_m3u8(["file1.m3u8", "file2.txt", "file3.M3U8"])
#     assert result == ["file1.m3u8"]


# def test_filter_webm():
#     result = filter_webm(["file1.webm", "file2.txt", "file3.WEBM"])
#     assert result == ["file1.webm"]


# def test_filter_images():
#     result = filter_images(["file1.png", "file2.txt", "file3.PNG"])
#     assert result == ["file1.png"]


# def test_filter_pdfs():
#     result = filter_pdfs(["file1.pdf", "file2.txt", "file3.PDF"])
#     assert result == ["file1.pdf"]


def test_filter_media(mocker):
    mock_filter_images = mocker.patch("goob_ai.utils.file_functions.filter_images", return_value=["file1.png"])
    mock_filter_videos = mocker.patch("goob_ai.utils.file_functions.filter_videos", return_value=["file2.mp4"])
    result = filter_media(["file1.png", "file2.mp4", "file3.txt"])
    assert result == ["file1.png", "file2.mp4"]
    mock_filter_images.assert_called_once_with(["file1.png", "file2.mp4", "file3.txt"])
    mock_filter_videos.assert_called_once_with(["file1.png", "file2.mp4", "file3.txt"])


# def test_filter_pdf(mocker):
#     mock_filter_pdfs = mocker.patch("goob_ai.utils.file_functions.filter_pdfs", return_value=["file1.pdf"])
#     result = filter_pdf(["file1.pdf", "file2.txt"])
#     assert result == ["file1.pdf"]
#     mock_filter_pdfs.assert_called_once_with(["file1.pdf", "file2.txt"])


def test_get_dataframe_from_csv(mocker):
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame({"col1": [1], "col2": [2]}))
    result = get_dataframe_from_csv("/Users/malcolm/dev/bossjones/goob_ai/test.csv")
    assert result.equals(
        pd.DataFrame({"col1": [1], "col2": [2]})
    ), f"Expected DataFrame: {pd.DataFrame({'col1': [1], 'col2': [2]})}, but got: {result}"
    mock_read_csv.assert_called_once_with("/Users/malcolm/dev/bossjones/goob_ai/test.csv")


# def test_sort_dataframe():
#     df = pd.DataFrame({"col1": [2, 1], "col2": [1, 2]})
#     result = sort_dataframe(df, columns=["col1"], ascending=(True,))
#     expected_df = pd.DataFrame({"col1": [1, 2], "col2": [2, 1]})
#     assert result.equals(expected_df), f"Expected DataFrame: {expected_df}, but got: {result}"


def test_rich_format_followers():
    result = rich_format_followers(1000000)
    assert result == "[bold bright_white]1000000[/bold bright_white]"


def test_rich_likes_or_comments():
    result = rich_likes_or_comments(10000)
    assert result == "[bold bright_yellow]10000[/bold bright_yellow]"


def test_rich_display_meme_pull_list(mocker):
    mock_console = mocker.patch("goob_ai.utils.file_functions.Console.print")
    df = pd.DataFrame(
        {
            "Account": ["acc1"],
            "Social": ["soc1"],
            "Total Followers": [1000],
            "Total Likes": [100],
            "Total Comments": [10],
            "Total Posts": [5],
            "Start Date": ["2021-01-01"],
            "End Date": ["2021-01-02"],
            "ERDay": [0.1],
            "ERpost": [0.2],
            "Average Likes": [20],
            "Average Comments": [2],
            "Links": ["link1"],
        }
    )
    rich_display_meme_pull_list(df)
    mock_console.assert_called_once()


def test_rich_display_popstars_analytics(mocker):
    mock_console = mocker.patch("goob_ai.utils.file_functions.Console.print")
    df = pd.DataFrame(
        {
            "Social": ["soc1"],
            "Author": ["auth1"],
            "Url": ["url1"],
            "Likes": [100],
            "Comments": [10],
            "ER": [0.1],
            "Text": ["text1"],
            "Date": ["2021-01-01"],
            "Media 1": ["media1"],
        }
    )
    rich_display_popstars_analytics(df)
    mock_console.assert_called_once()


def test_glob_file_by_extension(mocker):
    mock_glob = mocker.patch("glob.glob", return_value=["file1.mp4"])
    result = glob_file_by_extension("test_dir", "*.mp4", recursive=False)
    assert result == ["file1.mp4"]
    mock_glob.assert_called_once_with("test_dir/*.mp4", recursive=False)


def test_print_and_append(mocker):
    mock_print = mocker.patch("builtins.print")
    dir_listing = []
    print_and_append(dir_listing, "test_str", silent=False)
    assert dir_listing == ["test_str"]
    mock_print.assert_called_once_with("test_str")


# def test_tree(mocker):
#     mock_rglob = mocker.patch("pathlib.Path.rglob", return_value=[pathlib.Path("file1"), pathlib.Path("file2")])
#     mock_getmtime = mocker.patch("os.path.getmtime", side_effect=[2, 1])
#     result = tree(pathlib.Path("/Users/malcolm/dev/bossjones/goob_ai/test_dir"), silent=True)
#     assert result == [pathlib.Path("file2"), pathlib.Path("file1")]


# def test_format_size():
#     result = format_size(1024)
#     expected_result = "1 KB"
#     assert result == expected_result, f"Expected: {expected_result}, but got: {result}"


# @pytest.mark.asyncio
# async def test_aiowrite_file(mocker):
#     mock_open = mocker.patch("aiofiles.open", new_callable=mocker.AsyncMock)
#     await aiowrite_file("data", dl_dir="test_dir", fname="test", ext="txt")
#     mock_open.assert_called_once_with(pathlib.Path("test_dir/test.txt").absolute(), mode="w")


# @pytest.mark.asyncio
# async def test_airead_file(mocker):
#     mock_open = mocker.patch("aiofiles.open", new_callable=mocker.AsyncMock)
#     mock_open.return_value.__aenter__.return_value.read = mocker.AsyncMock(return_value="data")
#     await aioread_file("data", dl_dir="test_dir", fname="test", ext="txt")
#     mock_open.assert_called_once_with(pathlib.Path("test_dir/test.txt").absolute(), mode="r")


def test_check_file_size(mocker):
    mock_stat = mocker.patch("pathlib.Path.stat", return_value=mocker.Mock(st_size=1024))
    result = check_file_size("/Users/malcolm/dev/bossjones/goob_ai/test_file")
    assert result == (False, "Is file size greater than 50000000: False")


def test_is_file(mocker):
    mock_is_file = mocker.patch("pathlib.Path.is_file", return_value=True)
    result = is_file("test_file")
    assert result is True
    mock_is_file.assert_called_once_with()


def test_is_directory(mocker):
    mock_is_dir = mocker.patch("pathlib.Path.is_dir", return_value=True)
    result = is_directory("test_dir")
    assert result is True
    mock_is_dir.assert_called_once_with()


def test_is_a_symlink(mocker):
    mock_is_symlink = mocker.patch("pathlib.Path.is_symlink", return_value=True)
    result = is_a_symlink("test_symlink")
    assert result is True
    mock_is_symlink.assert_called_once_with()


def test_expand_path_str(mocker):
    mock_expanduser = mocker.patch("pathlib.Path.expanduser", return_value=pathlib.Path("/home/user"))
    result = expand_path_str("~")
    assert result == pathlib.Path("/home/user")
    mock_expanduser.assert_called_once_with()


def test_tilda(mocker):
    mock_expanduser = mocker.patch("pathlib.Path.expanduser", return_value=pathlib.Path("/home/user"))
    result = tilda("~")
    assert result == "/home/user"
    mock_expanduser.assert_called_once_with()


def test_fix_path(mocker):
    mock_is_file = mocker.patch("goob_ai.utils.file_functions.is_file", return_value=True)
    mock_tilda = mocker.patch("goob_ai.utils.file_functions.tilda", return_value="/home/user/file")
    result = fix_path("~")
    assert result == "/home/user/file"
    mock_is_file.assert_called_once_with("/home/user/file")
    mock_tilda.assert_called_once_with("~")
