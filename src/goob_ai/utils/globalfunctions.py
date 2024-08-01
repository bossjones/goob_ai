# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/main/utility/globalfunctions.py
from __future__ import annotations

import os
import re
import site

from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Union

import discord

from discord import Webhook, ui
from discord.ext import commands
from discord.utils import escape_markdown
from PIL import Image, ImageDraw


def find_urls(text: str):
    """
    Summary:
    Find URLs within a given text.

    Explanation:
    This function uses a regular expression pattern to search for URLs within the input text 'text'. It then returns a list of URLs found in the text.

    Args:
    ----
    - text (str): The text in which to search for URLs.

    Returns:
    -------
    - List[str]: A list of URLs extracted from the input text.

    """
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.findall(url_pattern, text)


def split_and_cluster_strings(input_string: str, max_cluster_size: int, split_substring: str, length=len) -> list[str]:
    """
    Split up the input_string by the split_substring
    and group the resulting substrings into
    clusters of about max_cluster_size length.
    Return the list of clusters.

    Args:
    ----
    input_string (str): The string to be split and clustered.
    max_cluster_size (int): The preferred maximum length of each cluster.
    split_substring (str): The substring used to split the input_string.
    length(Callable):  function to determine string length with.

    Returns:
    -------
    list[str]: A list of clusters.

    """
    clusters = []
    # There's no reason to split if input is already less than max_cluster_size
    if length(input_string) < max_cluster_size:
        return [input_string]

    split_by = split_substring

    is_regex = isinstance(split_substring, re.Pattern)
    if is_regex:
        result = split_substring.split(input_string)
        substrings = [r for r in result if r]
    else:
        if "%s" not in split_substring:
            split_by = f"%s{split_by}"
        split_character = split_by.replace("%s", "")

        # Split the input string based on the specified substring
        substrings = input_string.split(split_character)

    # No reason to run the loop if there's less than two
    # strings within the substrings list.  That means
    # it couldn't find anything to split up.
    if len(substrings) < 2:
        return [input_string]

    current_cluster = substrings[0]
    for substring in substrings[1:]:
        new_string = substring if is_regex else split_by.replace("%s", substring, 1)
        sublength = length(new_string)
        if length(current_cluster) + sublength <= max_cluster_size:
            # Add the substring to the current cluster
            current_cluster += new_string
        else:
            # Adding to the current cluster will exceed the maximum size,
            # So start a new cluster.
            if current_cluster:
                # Don't add to clusters if current_cluster is empty.
                clusters.append(current_cluster)
            current_cluster = new_string if substring else ""
    if current_cluster:
        clusters.append(current_cluster)  # Remove the trailing split_substring

    return clusters


def prioritized_string_split(
    input_string: str,
    substring_split_order: list[Union[str, tuple[str, int]]],
    default_max_len: int = 1024,
    trim=False,
    length=len,
) -> list[str]:
    """
    Segment the input string based on the delimiters specified in `substring_split_order`.
    Then, concatenate these segments to form a sequence of grouped strings,
    ensuring that no cluster surpasses a specified maximum length.
    The maximum length for each cluster addition
    can be individually adjusted along with the list of delimiters.


    Args:
    ----
        input_string (str): The string to be split.
        substring_split_order (list[Union[str, tuple[str, int]]]):
            A list of strings or tuples containing
            the delimiters to split by and their max lengths.
            If an argument here is "%s\\n", then the input string will be split by "\\n" and will
            place the relevant substrings in the position given by %s.
        default_max_len (int): The maximum length a string in a cluster may be if not given
            within a specific tuple for that delimiter.
        trim (bool): If True, trim leading and trailing whitespaces in each cluster. Default is False.

    Returns:
    -------
        list[str]: A list of clusters containing the split substrings.

    """
    # Initalize new cluster
    current_clusters = [input_string]
    for e, arg in enumerate(substring_split_order):
        if isinstance(arg, (str, re.Pattern)):
            s, max_len = arg, None
        elif len(arg) == 1:
            s, max_len = arg[0], None
        else:
            s, max_len = arg

        max_len = max_len or default_max_len  # Use default if not specified
        split_substring = s
        new_splits = []

        for cluster in current_clusters:
            result_clusters = split_and_cluster_strings(cluster, max_len, split_substring, length=length)
            new_splits.extend(result_clusters)
        # for c_num, cluster in enumerate(new_splits):
        #    print(f"Pass {e},  Cluster {c_num + 1}: {len(cluster)}, {len(cluster)}")
        current_clusters = new_splits

    # Optional trimming of leading and trailing whitespaces
    if trim:
        current_clusters = [cluster.strip() for cluster in current_clusters]

    return current_clusters


def split_string_with_code_blocks(input_str: str, max_length: int, oncode=False):
    """
    Summary:
    Split a string into segments based on specified criteria.

    Explanation:
    This function takes an input string 'input_str' and splits it into segments based on various splitting criteria such as Markdown headings, horizontal lines, and code blocks. It returns a list of segmented strings based on the provided maximum length 'max_length' and the presence of code blocks.

    Args:
    ----
    - input_str (str): The input string to be split into segments.
    - max_length (int): The maximum length of each segment.
    - oncode (bool): Flag to indicate whether to split on code blocks. Default is False.

    Returns:
    -------
    - List[str]: A list of segmented strings based on the splitting criteria and maximum length.

    """
    tosplitby = [
        # First, try to split along Markdown headings (starting with level 2)
        "\n#{1,6} ",
        # Note the alternative syntax for headings (below) is not handled here
        # Heading level 2
        # ---------------
        # Horizontal lines
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        " #{1,6} ",
        # Note that this splitter doesn't handle horizontal lines defined
        # by *three or more* of ***, ---, or ___, but this is not handled
        "\n\n",
        "\n",
        " ",
        "",
    ]
    if len(input_str) <= max_length:
        return [input_str]
    symbol = re.escape("```")
    pattern = re.compile(f"({symbol}(?:(?!{symbol}).)+{symbol})", re.DOTALL)

    splitorder = [pattern, "\n### %s", "%s\n", " %s"]
    return prioritized_string_split(input_str, splitorder, default_max_len=max_length)


def replace_working_directory(text: str):
    """This function is for replacing the current working directory with a shorthand"""
    cwd = os.getcwd()  # Replace backslashes for regex

    parent_dir = os.path.dirname(cwd)
    replaced_string = text
    for rawsite in site.getsitepackages():
        sites = os.path.dirname(rawsite)
        replaced_string = re.sub(re.escape(sites), "site", text, flags=re.IGNORECASE)

    replaced_string = re.sub(re.escape(parent_dir), "..", replaced_string, flags=re.IGNORECASE)

    return escape_markdown(replaced_string)


# def filter_trace_stack(stack):
#     """This function is for filtering call stacks so that ONLY the trace related to code files is shown"""
#     cwd = os.getcwd()  # Replace backslashes for regex
#     newlines = []

#     parent_dir = os.path.dirname(cwd)
#     for line in stack:
#         if parent_dir.upper() in line.upper().strip() and not ".venv" in line.strip():
#             newlines.append(line)
#     replaced_string = "\n".join(newlines)

#     return escape_markdown(replaced_string)


"""Utility functions here to assist."""


def the_string_numerizer(num: int, thestring: str, comma: bool = False, force: bool = False, have_s: bool = True):
    """
    Summary:
    Format a numerical value and string into a readable format.

    Explanation:
    This function takes a numerical value 'num', a string 'thestring', and optional parameters to format the value and string together. It returns a formatted string with the numerical value, string, and optional pluralization and comma based on the provided parameters.

    Args:
    ----
    - num (int): The numerical value to be formatted.
    - thestring (str): The string to be combined with the numerical value.
    - comma (bool): Flag to include a comma in the formatted string. Default is False.
    - force (bool): Flag to force formatting even if the numerical value is 0. Default is False.
    - have_s (bool): Flag to pluralize the string if the numerical value is greater than 1. Default is True.

    Returns:
    -------
    - str: The formatted string combining the numerical value and string based on the provided parameters.

    """
    if num > 0 or force:
        retstr = f"{num:.2f} {thestring}"
        if num > 1 and have_s:
            retstr += "s"
        if comma:
            retstr += ", "
        return retstr
    return ""


def seconds_to_time_string(seconds_start):
    """Return string of days, hours, minutes, and seconds"""
    return_string = ""
    seconds = seconds_start % 60
    minutes_r = (seconds_start - seconds) // 60
    minutes = minutes_r % 60
    hours_r = (minutes_r - minutes) // 60
    hours = hours_r % 24
    days = (hours_r - hours) // 24

    return "{}{}{}{}".format(
        the_string_numerizer(days, "day", True),
        the_string_numerizer(hours, "hour", True),
        the_string_numerizer(minutes, "minute", True),
        the_string_numerizer(seconds, "second", force=True),
    )


def seconds_to_time_stamp(seconds_init: int | float):
    # """return string of d:h:m:s"""
    """
    Summary:
    Convert seconds to a time stamp string in the format 'd:h:m:s'.

    Explanation:
    This function takes an initial number of seconds 'seconds_init' and converts it into a time stamp string representing days, hours, minutes, and seconds. It calculates the corresponding values for days, hours, minutes, and seconds and returns the formatted time stamp string.

    Args:
    ----
    - seconds_init (int | float): The initial number of seconds to convert to a time stamp.

    Returns:
    -------
    - str: The time stamp string in the format 'd:h:m:s' representing the converted seconds.

    """
    return_string = ""
    seconds_start = int(round(seconds_init))
    seconds = seconds_start % 60
    minutes_r = (seconds_start - seconds) // 60
    minutes = minutes_r % 60
    hours_r = (minutes_r - minutes) // 60
    hours = hours_r % 24
    if hours > 1:
        return_string += f"{hours:02d}:"
    return_string += f"{minutes:02d}:{seconds:02d}"
    return return_string


async def get_server_icon_color(guild: discord.Guild) -> str | int:
    "Get the guild icon, and color."
    if not guild.icon:
        return 0xFFFFFF
    icon_bytes = await guild.icon.read()
    icon_image = Image.open(BytesIO(icon_bytes))

    # Resize the image to 1x1 and get the most visible average color
    icon_image = icon_image.resize((1, 1))
    icon_color = icon_image.getpixel((0, 0))

    # Convert the color to hex format
    hex_color = f"{icon_color[0]:02x}{icon_color[1]:02x}{icon_color[2]:02x}"
    return int(hex_color, 16)


def extract_timestamp(timestamp: str):
    """
    Summary:
    Extract a timestamp string and convert it to a datetime object.

    Explanation:
    This function takes a timestamp string 'timestamp' and adjusts it to include up to 6 digits of fractional seconds. It then converts the adjusted timestamp string to a datetime object with timezone information and returns the datetime object.

    Args:
    ----
    - timestamp (str): The timestamp string to be converted to a datetime object.

    Returns:
    -------
    - datetime: The datetime object representing the converted timestamp with timezone information.

    """
    # Define the format of the timestamp string (with 7-digit fractional seconds)
    format_string = "%Y-%m-%dT%H:%M:%S.%fZ"

    # Extract the fractional seconds (up to 6 digits) and Z separately
    timestamp_parts = timestamp.split(".")
    timestamp_adjusted = timestamp
    if len(timestamp_parts) >= 2:
        timestamp_adjusted = f"{timestamp_parts[0]}.{timestamp_parts[1][:6]}"
    else:
        format_string = "%Y-%m-%dT%H:%M:%SZ"
        # timestamp_adjusted=timestamp_adjusted
    if not timestamp_adjusted.endswith("Z"):
        timestamp_adjusted += "Z"
    return datetime.strptime(timestamp_adjusted, format_string).replace(tzinfo=timezone.utc)


def human_format(num):
    """Format a large number"""
    num = float(f"{num:.3g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    suffixes = ["", "K", "M", "B", "T", "Q", "Qi"]
    return "{}{}".format(f"{num:f}".rstrip("0").rstrip("."), suffixes[magnitude])
