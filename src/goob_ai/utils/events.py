# pylint: disable=possibly-used-before-assignment,used-before-assignment
"""goob_ai.utils.events"""

from __future__ import annotations

import logging
import pathlib
import sys
import traceback

from typing import TYPE_CHECKING, List, Optional, Union

import discord
import rich

from loguru import logger as LOGGER

from goob_ai import downloader
from goob_ai.bot_logger import get_logger
from goob_ai.factories import cmd_factory
from goob_ai.utils.file_functions import get_all_media_files_to_upload, glob_file_by_extension, run_aio_json_loads


if TYPE_CHECKING:
    from discord.ext.commands import Context
    from discord.types.message import Message


def css_syntax_highlight(text: str):
    return f"```css\n{text}\n```"


def ig_type(typename: str):
    """
    Take instagram metadata payload typenames and convert them into readable IG post type values, eg Albums, Reels, or Image Post.

    Args:
    ----
        typename (str): _description_

    Returns:
    -------
        _type_: _description_

    """
    if typename == "GraphSidecar":
        return "Album"
    elif typename == "GraphImage":
        return "Image Post"
    elif typename == "GraphVideo":
        return "Reel"


def aio_create_thumbnail_attachment(tmpdirname: str, recursive: bool = False):
    #######################################################
    # add event to system channel
    #######################################################
    jpg_file_list = glob_file_by_extension(f"{tmpdirname}", extension="*.jpg", recursive=recursive)

    jpg_file = f"{jpg_file_list[0]}"
    LOGGER.debug(f"jpg_file = {jpg_file}")
    print(f"jpg_file = {jpg_file}")

    jpg_attachment = discord.File(jpg_file)
    attachment_url = f"attachment://{jpg_attachment}"
    return attachment_url, jpg_attachment


# async def aio_download_event(
#     ctx: Context,
#     tmpdirname: str,
#     cmd_metadata: cmd_factory.CmdSerializer,
#     is_dropbox_upload: bool = False,
#     recursive: bool = False,
# ):
#     #######################################################
#     # add event to system channel
#     #######################################################
#     json_file_list = glob_file_by_extension(f"{tmpdirname}", extension="*.json", recursive=recursive)

#     json_file = f"{json_file_list[0]}"
#     LOGGER.debug(f"json_file = {json_file}")
#     print(f"json_file = {json_file}")

#     try:
#         json_data = await run_aio_json_loads(json_file)
#     except Exception as ex:
#         await ctx.send(embed=discord.Embed(description="Could not open json metadata file"))
#         print(ex)
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         LOGGER.error(f"Error Class: {str(ex.__class__)}")
#         output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#         await ctx.send(embed=discord.Embed(description=f"{output}"))
#         LOGGER.warning(output)
#         LOGGER.error(f"exc_type: {exc_type}")
#         LOGGER.error(f"exc_value: {exc_value}")
#         traceback.print_tb(exc_traceback)

#     # current_message: Message
#     current_message: Message = ctx.message  # pyright: ignore[reportAttributeAccessIssue]
#     current_channel: discord.TextChannel
#     current_channel = ctx.channel
#     current_guild: discord.Guild
#     current_guild = current_channel.guild

#     if "youtu.be" in f"{cmd_metadata.uri}" or "youtube" in f"{cmd_metadata.uri}":
#         try:
#             # 1. Get guild
#             # ctx.guild.id
#             # = await guild_factory.Guild(id=guild.id)

#             ##########################################
#             full_description = json_data["description"]
#             description = f"{full_description[:75]}.." if len(full_description) > 75 else full_description
#             embed_event = discord.Embed(
#                 title=f"Downloaded: '{json_data['fulltitle']}' in channel #{current_channel.name}",
#                 url=f"{current_message.jump_url}",  # pyright: ignore[reportAttributeAccessIssue]
#                 description=css_syntax_highlight(description),
#                 color=discord.Color.blue(),
#             )
#             # set author
#             embed_event.set_author(
#                 name=json_data["channel"],
#                 url=json_data["uploader_url"],
#                 icon_url=json_data["thumbnail"],
#             )

#             # set thumbnail
#             embed_event.set_thumbnail(url=json_data["thumbnail"])
#             embed_event.set_image(url=json_data["thumbnail"])

#             embed_event.add_field(name="Url", value=f"{cmd_metadata.uri}", inline=False)
#             embed_event.add_field(
#                 name="View Count",
#                 value=css_syntax_highlight(json_data["view_count"]),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Duration in seconds",
#                 value=css_syntax_highlight(json_data["duration"]),
#                 inline=True,
#             )
#             embed_event.set_footer(text=f'Is dropbox upload? "{is_dropbox_upload}"')
#             ##########################################
#             # rich.inspect(current_guild, methods=True)
#             if current_guild.system_channel is not None:
#                 # to_send = 'Welcome {0.mention} to {1.name}!'.format(member, current_guild)
#                 await current_guild.system_channel.send(embed=embed_event)
#         except Exception as ex:
#             await ctx.send(embed=discord.Embed(description="Could not send download event to general"))
#             print(ex)
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             await ctx.send(embed=discord.Embed(description=f"{output}"))
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)

#     elif "instagram" in f"{cmd_metadata.uri}":
#         # 1. first grab the first media file we can find since IG can techincally download multiple media types
#         file_to_upload = get_all_media_files_to_upload(f"{tmpdirname}")

#         media_file_api = pathlib.Path(f"{file_to_upload[0]}")
#         # eg 280546359_2873025409665148_6148590927180637067_n.mp4.json
#         json_metadata_fname = f"{media_file_api.name}.json"

#         # For instagram we want to use the metadata json file instead of the info.json
#         json_file_list = glob_file_by_extension(f"{tmpdirname}", extension=json_metadata_fname, recursive=recursive)

#         json_file = f"{json_file_list[0]}"
#         LOGGER.debug(f"json_file = {json_file}")
#         print(f"json_file = {json_file}")

#         json_data = await run_aio_json_loads(json_file)

#         if "display_url" in json_data:
#             json_data_description = json_data["description"]
#             json_data_title = json_data["description"]
#             json_data_uploader_id = json_data["username"]
#             json_data_uploader_url = f"https://instagram.com/{json_data_uploader_id}"

#             dest_override = f"{tmpdirname}/{json_data['filename']}.jpg"
#             thumbnail, _ = await downloader.download_and_save(json_data["display_url"], dest_override)
#             rich.print(f"WE HAVE THE DISPLAY URL thumbnail => {thumbnail}")
#             rich.print(f"WE HAVE THE DISPLAY URL dest_override => {dest_override}")

#             # since we are not using recursive, it should just find the first jpg in directory which will be at tmpdirname/image.jpg
#             attachment_url, jpg_attachment = aio_create_thumbnail_attachment(f"{tmpdirname}")

#             rich.print(f"WE HAVE THE DISPLAY URL attachment_url => {attachment_url}")
#             rich.print(f"WE HAVE THE DISPLAY URL jpg_attachment => {jpg_attachment}")

#             json_data_icon_url = attachment_url
#             json_data_thumbnail = attachment_url
#             json_data_like_count = json_data["likes"] if "likes" in json_data else "n/a"
#         else:
#             await ctx.send(embed=discord.Embed(description="Key 'display_url' is not in dictonary 'json_data'"))

#         try:
#             ##########################################
#             full_description = json_data_description
#             description = f"{full_description[:75]}.." if len(full_description) > 75 else full_description
#             embed_event = discord.Embed(
#                 title=f"Downloaded: '{description}' in channel #{current_channel.name}",
#                 url=f"{current_message.jump_url}",  # pyright: ignore[reportAttributeAccessIssue]
#                 description=css_syntax_highlight(description),
#                 color=discord.Color.blue(),
#             )
#             # set author
#             embed_event.set_author(
#                 name=json_data_uploader_id,
#                 url=json_data_uploader_url,
#                 icon_url=attachment_url,
#             )

#             # set thumbnail
#             embed_event.set_thumbnail(url=attachment_url)
#             embed_event.set_image(url=attachment_url)

#             embed_event.add_field(name="Url", value=f"{cmd_metadata.uri}", inline=False)
#             embed_event.add_field(
#                 name="Likes",
#                 value=css_syntax_highlight(json_data_like_count),
#                 inline=True,
#             )
#             # embed_event.add_field(
#             #     name="Type",
#             #     value=css_syntax_highlight(ig_type(json_data["typename"])),
#             #     inline=True,
#             # )
#             embed_event.set_footer(text=f'Is dropbox upload? "{is_dropbox_upload}"')
#             ##########################################
#             if current_guild.system_channel is not None:
#                 await current_guild.system_channel.send(file=jpg_attachment, embed=embed_event)
#         except Exception as ex:
#             await ctx.send(embed=discord.Embed(description="Could not send download event to general"))
#             print(ex)
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             await ctx.send(embed=discord.Embed(description=f"{output}"))
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)

#     elif "twitter" in f"{cmd_metadata.uri}":
#         attachment_url = None
#         jpg_attachment = None

#         rich.print(json_data)

#         # This means we used yt-dlp
#         if "description" in json_data:
#             json_data_description = json_data["description"]
#             json_data_title = json_data["title"]
#             json_data_uploader_id = json_data["uploader_id"]
#             json_data_uploader_url = json_data["uploader_url"]
#             json_data_icon_url = json_data["thumbnail"]
#             json_data_thumbnail = json_data["thumbnail"]
#             json_data_like_count = json_data["like_count"]
#             json_data_repost_count = json_data["repost_count"]
#             json_data_comment_count = json_data["comment_count"]
#         # this means we are using gallery-dl to download a tweet
#         elif "content" in json_data:
#             json_data_description = json_data["content"]
#             json_data_title = json_data["content"]
#             json_data_uploader_id = json_data["author"]["name"]
#             json_data_uploader_url = f"https://twitter.com/{json_data_uploader_id}"
#             json_data_icon_url = json_data["author"]["profile_image"]
#             attachment_url, jpg_attachment = aio_create_thumbnail_attachment(f"{tmpdirname}", recursive=True)
#             json_data_thumbnail = attachment_url
#             json_data_like_count = json_data["favorite_count"]
#             json_data_repost_count = json_data["retweet_count"]
#             json_data_comment_count = json_data["reply_count"]

#         try:
#             # 1. Get guild
#             # ctx.guild.id
#             # = await guild_factory.Guild(id=guild.id)

#             ##########################################
#             full_description = json_data_description
#             description = f"{full_description[:75]}.." if len(full_description) > 75 else full_description
#             embed_event = discord.Embed(
#                 title=f"Downloaded: '{json_data_title}' in channel #{current_channel.name}",
#                 url=f"{current_message.jump_url}",  # pyright: ignore[reportAttributeAccessIssue]
#                 description=css_syntax_highlight(description),
#                 color=discord.Color.blue(),
#             )
#             # set author
#             embed_event.set_author(
#                 name=json_data_uploader_id,
#                 url=json_data_uploader_url,
#                 icon_url=json_data_icon_url,
#             )

#             # set thumbnail
#             embed_event.set_thumbnail(url=json_data_thumbnail)
#             embed_event.set_image(url=json_data_thumbnail)

#             embed_event.add_field(name="Url", value=f"{cmd_metadata.uri}", inline=False)
#             embed_event.add_field(
#                 name="Like Count",
#                 value=css_syntax_highlight(json_data_like_count),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Retweets",
#                 value=css_syntax_highlight(json_data_repost_count),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Comments",
#                 value=css_syntax_highlight(json_data_comment_count),
#                 inline=True,
#             )
#             embed_event.set_footer(text=f'Is dropbox upload? "{is_dropbox_upload}"')
#             if current_guild.system_channel is not None:
#                 if json_data_thumbnail is attachment_url:
#                     await current_guild.system_channel.send(file=jpg_attachment, embed=embed_event)
#                 else:
#                     await current_guild.system_channel.send(embed=embed_event)
#         except Exception as ex:
#             await ctx.send(embed=discord.Embed(description="Could not send download event to general"))
#             print(ex)
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             await ctx.send(embed=discord.Embed(description=f"{output}"))
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)
#     elif "reddit" in f"{cmd_metadata.uri}" or "redd.it" in f"{cmd_metadata.uri}":
#         # rich.print(json_data)

#         try:
#             # 1. Get guild
#             # ctx.guild.id
#             # = await guild_factory.Guild(id=guild.id)

#             ##########################################
#             full_description = json_data["title"]
#             if "nsfw" in json_data["thumbnail"]:
#                 thumbnail = json_data["preview"]["images"][0]["source"]["url"]
#             else:
#                 thumbnail = json_data["thumbnail"]

#             description = f"{full_description[:75]}.." if len(full_description) > 75 else full_description
#             embed_event = discord.Embed(
#                 title=f"Downloaded: '{json_data['title']}' in channel #{current_channel.name}",
#                 url=f"{current_message.jump_url}",  # pyright: ignore[reportAttributeAccessIssue]
#                 description=css_syntax_highlight(description),
#                 color=discord.Color.blue(),
#             )
#             # set author
#             embed_event.set_author(
#                 name=json_data["author"],
#                 url=f"https://www.reddit.com/user/{json_data['author']}/",
#                 icon_url=thumbnail,
#             )

#             # set thumbnail
#             embed_event.set_thumbnail(url=thumbnail)
#             embed_event.set_image(url=thumbnail)

#             embed_event.add_field(name="Url", value=f"{cmd_metadata.uri}", inline=False)
#             embed_event.add_field(
#                 name="Upvotes",
#                 value=css_syntax_highlight(json_data["score"]),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Comments",
#                 value=css_syntax_highlight(json_data["num_comments"]),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Subreddit",
#                 value=css_syntax_highlight(json_data["subreddit"]),
#                 inline=True,
#             )
#             embed_event.set_footer(text=f'Is dropbox upload? "{is_dropbox_upload}"')
#             ##########################################
#             # rich.inspect(current_guild, methods=True)
#             if current_guild.system_channel is not None:
#                 await current_guild.system_channel.send(embed=embed_event)
#         except Exception as ex:
#             await ctx.send(embed=discord.Embed(description="Could not send download event to general"))
#             print(ex)
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             await ctx.send(embed=discord.Embed(description=f"{output}"))
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)

#     elif "tiktok" in f"{cmd_metadata.uri}":
#         # NOTE: Tiktok doesn't have a public thumbnail url, so we need to create an attachment and upload it
#         attachment_url, jpg_attachment = aio_create_thumbnail_attachment(f"{tmpdirname}")

#         try:
#             # 1. Get guild
#             # ctx.guild.id
#             # = await guild_factory.Guild(id=guild.id)

#             ##########################################
#             full_description = json_data["description"]
#             description = f"{full_description[:75]}.." if len(full_description) > 75 else full_description
#             embed_event = discord.Embed(
#                 title=f"Downloaded: '{json_data['title']}' in channel #{current_channel.name}",
#                 url=f"{current_message.jump_url}",  # pyright: ignore[reportAttributeAccessIssue]
#                 description=css_syntax_highlight(description),
#                 color=discord.Color.blue(),
#             )
#             # set author
#             embed_event.set_author(
#                 name=json_data["uploader"],
#                 url=f"https://tiktok.com/@{json_data['uploader']}",
#                 icon_url=attachment_url,
#             )

#             # set thumbnail
#             embed_event.set_thumbnail(url=attachment_url)
#             embed_event.set_image(url=attachment_url)

#             embed_event.add_field(name="Url", value=f"{cmd_metadata.uri}", inline=False)
#             embed_event.add_field(
#                 name="View Count",
#                 value=css_syntax_highlight(json_data["view_count"]),
#                 inline=True,
#             )
#             embed_event.add_field(
#                 name="Like Count",
#                 value=css_syntax_highlight(json_data["like_count"]),
#                 inline=True,
#             )
#             embed_event.set_footer(text=f'Is dropbox upload? "{is_dropbox_upload}"')
#             ##########################################
#             # rich.inspect(current_guild, methods=True)
#             if current_guild.system_channel is not None:
#                 # to_send = 'Welcome {0.mention} to {1.name}!'.format(member, current_guild)
#                 await current_guild.system_channel.send(file=jpg_attachment, embed=embed_event)
#         except Exception as ex:
#             await ctx.send(embed=discord.Embed(description="Could not send download event to general"))
#             print(ex)
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             await ctx.send(embed=discord.Embed(description=f"{output}"))
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)

#     else:
#         if current_guild.system_channel is not None:
#             await current_guild.system_channel.send(
#                 embed=discord.Embed(
#                     description=f"```css\nSorry, **aio_download_event** isn't configured to deal with these urls yet: {cmd_metadata.uri}\n```"
#                 )
#             )

#         rich.print(json_data)
