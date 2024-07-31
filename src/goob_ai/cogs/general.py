# pylint: disable=no-name-in-module
# mypy: disable-error-code="arg-type"
# SOURCE: https://github.com/ausboss/DiscordLangAgent/blob/58a7289a49864fdadb3a70d8a106b2fc92c2ee7b/cogs/general.py#L13
from __future__ import annotations

import platform
import random

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Protocol, TypeVar, Union

import aiohttp
import discord

from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Command, Context
from goob_ai.aio_settings import aiosettings
from goob_ai.factories import cmd_factory, guild_factory
from goob_ai.goob_bot import AsyncGoobBot
from goob_ai.helpers import checks
from loguru import logger as LOGGER


if TYPE_CHECKING:
    from ..goob_bot import AsyncGoobBot


class General(commands.Cog, name="general"):
    def __init__(self, bot: AsyncGoobBot):
        self.bot: AsyncGoobBot = bot

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        print(f"{type(self).__name__} Cog ready.")

    @commands.Cog.listener()
    async def on_guild_join(self, guild: guild_factory.Guild) -> None:
        """Add new guilds to the database"""
        _ = await guild_factory.Guild(id=guild.id)

    @commands.hybrid_command(name="help", description="List all commands the bot has loaded.")
    # @checks.not_blacklisted()
    async def help(self, context: Context) -> None:
        prefix = aiosettings.prefix
        embed = discord.Embed(title="Help", description="List of available commands:", color=0x9C84EF)
        for i in self.bot.cogs:
            cog = self.bot.get_cog(i.lower())
            commands: list[Command] = cog.get_commands()
            data = []
            for command in commands:
                description = command.description.partition("\n")[0]  # pyright: ignore[reportAttributeAccessIssue]
                data.append(f"{prefix}{command.name} - {description}")  # pyright: ignore[reportAttributeAccessIssue]
            help_text = "\n".join(data)
            embed.add_field(name=i.capitalize(), value=f"```{help_text}```", inline=False)
        await context.send(embed=embed)

    @commands.hybrid_command(
        name="botinfo",
        description="Get some useful (or not) information about the bot.",
    )
    # @checks.not_blacklisted()
    async def botinfo(self, context: Context) -> None:
        """
        Get some useful (or not) information about the bot.

        :param context: The hybrid command context.
        """
        embed = discord.Embed(
            description="Used markdown template",
            color=0x9C84EF,
        )
        embed.set_author(name="Bot Information")
        embed.add_field(name="Owner:", value="Krypton#7331", inline=True)
        embed.add_field(name="Python Version:", value=f"{platform.python_version()}", inline=True)
        embed.add_field(
            name="Prefix:",
            value=f"/ (Slash Commands) or {aiosettings.prefix} for normal commands",
            inline=False,
        )
        embed.set_footer(text=f"Requested by {context.author}")
        await context.send(embed=embed)

    @commands.hybrid_command(
        name="serverinfo",
        description="Get some useful (or not) information about the server.",
    )
    # @checks.not_blacklisted()
    async def serverinfo(self, context: Context) -> None:
        """
        Get some useful (or not) information about the server.

        :param context: The hybrid command context.
        """
        roles = [role.name for role in context.guild.roles]
        if len(roles) > 50:
            roles = roles[:50]
            roles.append(f">>>> Displaying[50/{len(roles)}] Roles")
        roles = ", ".join(roles)

        embed = discord.Embed(title="**Server Name:**", description=f"{context.guild}", color=0x9C84EF)
        if context.guild.icon is not None:
            embed.set_thumbnail(url=context.guild.icon.url)
        embed.add_field(name="Server ID", value=context.guild.id)
        embed.add_field(name="Member Count", value=context.guild.member_count)
        embed.add_field(name="Text/Voice Channels", value=f"{len(context.guild.channels)}")
        embed.add_field(name=f"Roles ({len(context.guild.roles)})", value=roles)
        embed.set_footer(text=f"Created at: {context.guild.created_at}")
        await context.send(embed=embed)

    @commands.hybrid_command(
        name="ping",
        description="Check if the bot is alive.",
    )
    # @checks.not_blacklisted()
    async def ping(self, context: Context) -> None:
        """
        Check if the bot is alive.

        :param context: The hybrid command context.
        """
        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"The bot latency is {round(self.bot.latency * 1000)}ms.",
            color=0x9C84EF,
        )
        await context.send(embed=embed)

    # @commands.hybrid_command(
    #     name="invite",
    #     description="Get the invite link of the bot to be able to invite it.",
    # )
    # # @checks.not_blacklisted()
    # async def invite(self, context: Context) -> None:
    #     """
    #     Get the invite link of the bot to be able to invite it.

    #     :param context: The hybrid command context.
    #     """

    #     embed = discord.Embed(
    #         description=f"Invite me by clicking [here](https://discordapp.com/oauth2/authorize?&client_id={self.bot.config['application_id']}&scope=bot+applications.commands&permissions={self.bot.config['permissions']}).",
    #         color=0xD75BF4,
    #     )
    #     try:
    #         # To know what permissions to give to your bot, please see here: https://discordapi.com/permissions.html and remember to not give Administrator permissions.
    #         await context.author.send(embed=embed)
    #         await context.send("I sent you a private message!")
    #     except discord.Forbidden:
    #         await context.send(embed=embed)

    @commands.hybrid_command(
        name="server",
        description="Get the invite link of the discord server of the bot for some support.",
    )
    async def server(self, context: Context) -> None:
        """
        Get the invite link of the discord server of the bot for some support.

        :param context: The hybrid command context.
        """
        embed = discord.Embed(
            description="Join the support server for the bot by clicking [here](https://discord.gg/mTBrXyWxAF).",
            color=0xD75BF4,
        )
        try:
            await context.author.send(embed=embed)
            await context.send("I sent you a private message!")
        except discord.Forbidden:
            await context.send(embed=embed)

    @commands.hybrid_command(
        name="8ball",
        description="Ask any question to the bot.",
    )
    # @checks.not_blacklisted()
    @app_commands.describe(question="The question you want to ask.")
    async def eight_ball(self, context: Context, *, question: str) -> None:
        """
        Ask any question to the bot.

        :param context: The hybrid command context.
        :param question: The question that should be asked by the user.
        """
        answers = [
            "It is certain.",
            "It is decidedly so.",
            "You may rely on it.",
            "Without a doubt.",
            "Yes - definitely.",
            "As I see, yes.",
            "Most likely.",
            "Outlook good.",
            "Yes.",
            "Signs point to yes.",
            "Reply hazy, try again.",
            "Ask again later.",
            "Better not tell you now.",
            "Cannot predict now.",
            "Concentrate and ask again later.",
            "Don't count on it.",
            "My reply is no.",
            "My sources say no.",
            "Outlook not so good.",
            "Very doubtful.",
        ]
        embed = discord.Embed(
            title="**My Answer:**",
            description=f"{random.choice(answers)}",
            color=0x9C84EF,
        )
        embed.set_footer(text=f"The question was: {question}")
        await context.send(embed=embed)

    @commands.hybrid_command(
        name="bitcoin",
        description="Get the current price of bitcoin.",
    )
    # @checks.not_blacklisted()
    async def bitcoin(self, context: Context) -> None:
        """
        Get the current price of bitcoin.

        :param context: The hybrid command context.
        """
        # This will prevent your bot from stopping everything when doing a web request - see: https://discordpy.readthedocs.io/en/stable/faq.html#how-do-i-make-a-web-request
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.coindesk.com/v1/bpi/currentprice/BTC.json") as request:
                if request.status == 200:
                    data = await request.json(
                        content_type="application/javascript"
                    )  # For some reason the returned content is of type JavaScript
                    embed = discord.Embed(
                        title="Bitcoin price",
                        description=f"The current price is {data['bpi']['USD']['rate']} :dollar:",
                        color=0x9C84EF,
                    )
                else:
                    embed = discord.Embed(
                        title="Error!",
                        description="There is something wrong with the API, please try again later",
                        color=0xE02B2B,
                    )
                await context.send(embed=embed)


async def setup(bot: AsyncGoobBot):
    await bot.add_cog(General(bot))
