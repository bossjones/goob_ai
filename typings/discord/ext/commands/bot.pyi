"""
This type stub file was generated by pyright.
"""

import types
import discord
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING, Type, TypeVar, Union, overload
from discord import app_commands
from .core import Command, GroupMixin
from .context import Context
from . import errors
from .help import HelpCommand
from .cog import Cog
from .hybrid import CommandCallback, ContextT, HybridCommand, HybridGroup, P
from typing_extensions import Self
from discord.message import Message
from discord.interactions import Interaction
from discord.abc import Snowflake, User
from ._types import BotT, ContextT, CoroFunc, MaybeAwaitableFunc, UserCheck, _Bot

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
if TYPE_CHECKING:
    _Prefix = Union[Iterable[str], str]
    _PrefixCallable = MaybeAwaitableFunc[[BotT, Message], _Prefix]
    PrefixType = Union[_Prefix, _PrefixCallable[BotT]]
__all__ = ('when_mentioned', 'when_mentioned_or', 'Bot', 'AutoShardedBot')
T = TypeVar('T')
CFT = TypeVar('CFT', bound='CoroFunc')
_log = ...
def when_mentioned(bot: _Bot, msg: Message, /) -> List[str]:
    """A callable that implements a command prefix equivalent to being mentioned.

    These are meant to be passed into the :attr:`.Bot.command_prefix` attribute.

        .. versionchanged:: 2.0

            ``bot`` and ``msg`` parameters are now positional-only.
    """
    ...

def when_mentioned_or(*prefixes: str) -> Callable[[_Bot, Message], List[str]]:
    """A callable that implements when mentioned or other prefixes provided.

    These are meant to be passed into the :attr:`.Bot.command_prefix` attribute.

    Example
    --------

    .. code-block:: python3

        bot = commands.Bot(command_prefix=commands.when_mentioned_or('!'))


    .. note::

        This callable returns another callable, so if this is done inside a custom
        callable, you must call the returned callable, for example:

        .. code-block:: python3

            async def get_prefix(bot, message):
                extras = await prefixes_for(message.guild) # returns a list
                return commands.when_mentioned_or(*extras)(bot, message)


    See Also
    ----------
    :func:`.when_mentioned`
    """
    ...

class _DefaultRepr:
    def __repr__(self): # -> Literal['<default-help-command>']:
        ...
    


_default: Any = ...
class BotBase(GroupMixin[None]):
    def __init__(self, command_prefix: PrefixType[BotT], *, help_command: Optional[HelpCommand] = ..., tree_cls: Type[app_commands.CommandTree[Any]] = ..., description: Optional[str] = ..., intents: discord.Intents, **options: Any) -> None:
        ...
    
    def dispatch(self, event_name: str, /, *args: Any, **kwargs: Any) -> None:
        ...
    
    @discord.utils.copy_doc(discord.Client.close)
    async def close(self) -> None:
        ...
    
    @discord.utils.copy_doc(GroupMixin.add_command)
    def add_command(self, command: Command[Any, ..., Any], /) -> None:
        ...
    
    @discord.utils.copy_doc(GroupMixin.remove_command)
    def remove_command(self, name: str, /) -> Optional[Command[Any, ..., Any]]:
        ...
    
    def hybrid_command(self, name: Union[str, app_commands.locale_str] = ..., with_app_command: bool = ..., *args: Any, **kwargs: Any) -> Callable[[CommandCallback[Any, ContextT, P, T]], HybridCommand[Any, P, T]]:
        """A shortcut decorator that invokes :func:`~discord.ext.commands.hybrid_command` and adds it to
        the internal command list via :meth:`add_command`.

        Returns
        --------
        Callable[..., :class:`HybridCommand`]
            A decorator that converts the provided method into a Command, adds it to the bot, then returns it.
        """
        ...
    
    def hybrid_group(self, name: Union[str, app_commands.locale_str] = ..., with_app_command: bool = ..., *args: Any, **kwargs: Any) -> Callable[[CommandCallback[Any, ContextT, P, T]], HybridGroup[Any, P, T]]:
        """A shortcut decorator that invokes :func:`~discord.ext.commands.hybrid_group` and adds it to
        the internal command list via :meth:`add_command`.

        Returns
        --------
        Callable[..., :class:`HybridGroup`]
            A decorator that converts the provided method into a Group, adds it to the bot, then returns it.
        """
        ...
    
    async def on_command_error(self, context: Context[BotT], exception: errors.CommandError, /) -> None:
        """|coro|

        The default command error handler provided by the bot.

        By default this logs to the library logger, however it could be
        overridden to have a different implementation.

        This only fires if you do not specify any listeners for command error.

        .. versionchanged:: 2.0

            ``context`` and ``exception`` parameters are now positional-only.
            Instead of writing to ``sys.stderr`` this now uses the library logger.
        """
        ...
    
    def check(self, func: T, /) -> T:
        r"""A decorator that adds a global check to the bot.

        A global check is similar to a :func:`.check` that is applied
        on a per command basis except it is run before any command checks
        have been verified and applies to every command the bot has.

        .. note::

            This function can either be a regular function or a coroutine.

        Similar to a command :func:`.check`\, this takes a single parameter
        of type :class:`.Context` and can only raise exceptions inherited from
        :exc:`.CommandError`.

        Example
        ---------

        .. code-block:: python3

            @bot.check
            def check_commands(ctx):
                return ctx.command.qualified_name in allowed_commands

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.
        """
        ...
    
    def add_check(self, func: UserCheck[ContextT], /, *, call_once: bool = ...) -> None:
        """Adds a global check to the bot.

        This is the non-decorator interface to :meth:`.check`
        and :meth:`.check_once`.

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        .. seealso:: The :func:`~discord.ext.commands.check` decorator

        Parameters
        -----------
        func
            The function that was used as a global check.
        call_once: :class:`bool`
            If the function should only be called once per
            :meth:`.invoke` call.
        """
        ...
    
    def remove_check(self, func: UserCheck[ContextT], /, *, call_once: bool = ...) -> None:
        """Removes a global check from the bot.

        This function is idempotent and will not raise an exception
        if the function is not in the global checks.

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        Parameters
        -----------
        func
            The function to remove from the global checks.
        call_once: :class:`bool`
            If the function was added with ``call_once=True`` in
            the :meth:`.Bot.add_check` call or using :meth:`.check_once`.
        """
        ...
    
    def check_once(self, func: CFT, /) -> CFT:
        r"""A decorator that adds a "call once" global check to the bot.

        Unlike regular global checks, this one is called only once
        per :meth:`.invoke` call.

        Regular global checks are called whenever a command is called
        or :meth:`.Command.can_run` is called. This type of check
        bypasses that and ensures that it's called only once, even inside
        the default help command.

        .. note::

            When using this function the :class:`.Context` sent to a group subcommand
            may only parse the parent command and not the subcommands due to it
            being invoked once per :meth:`.Bot.invoke` call.

        .. note::

            This function can either be a regular function or a coroutine.

        Similar to a command :func:`.check`\, this takes a single parameter
        of type :class:`.Context` and can only raise exceptions inherited from
        :exc:`.CommandError`.

        Example
        ---------

        .. code-block:: python3

            @bot.check_once
            def whitelist(ctx):
                return ctx.message.author.id in my_whitelist

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        """
        ...
    
    async def can_run(self, ctx: Context[BotT], /, *, call_once: bool = ...) -> bool:
        ...
    
    async def is_owner(self, user: User, /) -> bool:
        """|coro|

        Checks if a :class:`~discord.User` or :class:`~discord.Member` is the owner of
        this bot.

        If an :attr:`owner_id` is not set, it is fetched automatically
        through the use of :meth:`~.Bot.application_info`.

        .. versionchanged:: 1.3
            The function also checks if the application is team-owned if
            :attr:`owner_ids` is not set.

        .. versionchanged:: 2.0

            ``user`` parameter is now positional-only.

        Parameters
        -----------
        user: :class:`.abc.User`
            The user to check for.

        Returns
        --------
        :class:`bool`
            Whether the user is the owner.
        """
        ...
    
    def before_invoke(self, coro: CFT, /) -> CFT:
        """A decorator that registers a coroutine as a pre-invoke hook.

        A pre-invoke hook is called directly before the command is
        called. This makes it a useful function to set up database
        connections or any type of set up required.

        This pre-invoke hook takes a sole parameter, a :class:`.Context`.

        .. note::

            The :meth:`~.Bot.before_invoke` and :meth:`~.Bot.after_invoke` hooks are
            only called if all checks and argument parsing procedures pass
            without error. If any check or argument parsing procedures fail
            then the hooks are not called.

        .. versionchanged:: 2.0

            ``coro`` parameter is now positional-only.

        Parameters
        -----------
        coro: :ref:`coroutine <coroutine>`
            The coroutine to register as the pre-invoke hook.

        Raises
        -------
        TypeError
            The coroutine passed is not actually a coroutine.
        """
        ...
    
    def after_invoke(self, coro: CFT, /) -> CFT:
        r"""A decorator that registers a coroutine as a post-invoke hook.

        A post-invoke hook is called directly after the command is
        called. This makes it a useful function to clean-up database
        connections or any type of clean up required.

        This post-invoke hook takes a sole parameter, a :class:`.Context`.

        .. note::

            Similar to :meth:`~.Bot.before_invoke`\, this is not called unless
            checks and argument parsing procedures succeed. This hook is,
            however, **always** called regardless of the internal command
            callback raising an error (i.e. :exc:`.CommandInvokeError`\).
            This makes it ideal for clean-up scenarios.

        .. versionchanged:: 2.0

            ``coro`` parameter is now positional-only.

        Parameters
        -----------
        coro: :ref:`coroutine <coroutine>`
            The coroutine to register as the post-invoke hook.

        Raises
        -------
        TypeError
            The coroutine passed is not actually a coroutine.
        """
        ...
    
    def add_listener(self, func: CoroFunc, /, name: str = ...) -> None:
        """The non decorator alternative to :meth:`.listen`.

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        Parameters
        -----------
        func: :ref:`coroutine <coroutine>`
            The function to call.
        name: :class:`str`
            The name of the event to listen for. Defaults to ``func.__name__``.

        Example
        --------

        .. code-block:: python3

            async def on_ready(): pass
            async def my_message(message): pass

            bot.add_listener(on_ready)
            bot.add_listener(my_message, 'on_message')

        """
        ...
    
    def remove_listener(self, func: CoroFunc, /, name: str = ...) -> None:
        """Removes a listener from the pool of listeners.

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        Parameters
        -----------
        func
            The function that was used as a listener to remove.
        name: :class:`str`
            The name of the event we want to remove. Defaults to
            ``func.__name__``.
        """
        ...
    
    def listen(self, name: str = ...) -> Callable[[CFT], CFT]:
        """A decorator that registers another function as an external
        event listener. Basically this allows you to listen to multiple
        events from different places e.g. such as :func:`.on_ready`

        The functions being listened to must be a :ref:`coroutine <coroutine>`.

        Example
        --------

        .. code-block:: python3

            @bot.listen()
            async def on_message(message):
                print('one')

            # in some other file...

            @bot.listen('on_message')
            async def my_message(message):
                print('two')

        Would print one and two in an unspecified order.

        Raises
        -------
        TypeError
            The function being listened to is not a coroutine.
        """
        ...
    
    async def add_cog(self, cog: Cog, /, *, override: bool = ..., guild: Optional[Snowflake] = ..., guilds: Sequence[Snowflake] = ...) -> None:
        """|coro|

        Adds a "cog" to the bot.

        A cog is a class that has its own event listeners and commands.

        If the cog is a :class:`.app_commands.Group` then it is added to
        the bot's :class:`~discord.app_commands.CommandTree` as well.

        .. note::

            Exceptions raised inside a :class:`.Cog`'s :meth:`~.Cog.cog_load` method will be
            propagated to the caller.

        .. versionchanged:: 2.0

            :exc:`.ClientException` is raised when a cog with the same name
            is already loaded.

        .. versionchanged:: 2.0

            ``cog`` parameter is now positional-only.

        .. versionchanged:: 2.0

            This method is now a :term:`coroutine`.

        Parameters
        -----------
        cog: :class:`.Cog`
            The cog to register to the bot.
        override: :class:`bool`
            If a previously loaded cog with the same name should be ejected
            instead of raising an error.

            .. versionadded:: 2.0
        guild: Optional[:class:`~discord.abc.Snowflake`]
            If the cog is an application command group, then this would be the
            guild where the cog group would be added to. If not given then
            it becomes a global command instead.

            .. versionadded:: 2.0
        guilds: List[:class:`~discord.abc.Snowflake`]
            If the cog is an application command group, then this would be the
            guilds where the cog group would be added to. If not given then
            it becomes a global command instead. Cannot be mixed with
            ``guild``.

            .. versionadded:: 2.0

        Raises
        -------
        TypeError
            The cog does not inherit from :class:`.Cog`.
        CommandError
            An error happened during loading.
        ClientException
            A cog with the same name is already loaded.
        """
        ...
    
    def get_cog(self, name: str, /) -> Optional[Cog]:
        """Gets the cog instance requested.

        If the cog is not found, ``None`` is returned instead.

        .. versionchanged:: 2.0

            ``name`` parameter is now positional-only.

        Parameters
        -----------
        name: :class:`str`
            The name of the cog you are requesting.
            This is equivalent to the name passed via keyword
            argument in class creation or the class name if unspecified.

        Returns
        --------
        Optional[:class:`Cog`]
            The cog that was requested. If not found, returns ``None``.
        """
        ...
    
    async def remove_cog(self, name: str, /, *, guild: Optional[Snowflake] = ..., guilds: Sequence[Snowflake] = ...) -> Optional[Cog]:
        """|coro|

        Removes a cog from the bot and returns it.

        All registered commands and event listeners that the
        cog has registered will be removed as well.

        If no cog is found then this method has no effect.

        .. versionchanged:: 2.0

            ``name`` parameter is now positional-only.

        .. versionchanged:: 2.0

            This method is now a :term:`coroutine`.

        Parameters
        -----------
        name: :class:`str`
            The name of the cog to remove.
        guild: Optional[:class:`~discord.abc.Snowflake`]
            If the cog is an application command group, then this would be the
            guild where the cog group would be removed from. If not given then
            a global command is removed instead instead.

            .. versionadded:: 2.0
        guilds: List[:class:`~discord.abc.Snowflake`]
            If the cog is an application command group, then this would be the
            guilds where the cog group would be removed from. If not given then
            a global command is removed instead instead. Cannot be mixed with
            ``guild``.

            .. versionadded:: 2.0

        Returns
        -------
        Optional[:class:`.Cog`]
             The cog that was removed. ``None`` if not found.
        """
        ...
    
    @property
    def cogs(self) -> Mapping[str, Cog]:
        """Mapping[:class:`str`, :class:`Cog`]: A read-only mapping of cog name to cog."""
        ...
    
    async def load_extension(self, name: str, *, package: Optional[str] = ...) -> None:
        """|coro|

        Loads an extension.

        An extension is a python module that contains commands, cogs, or
        listeners.

        An extension must have a global function, ``setup`` defined as
        the entry point on what to do when the extension is loaded. This entry
        point must have a single argument, the ``bot``.

        .. versionchanged:: 2.0

            This method is now a :term:`coroutine`.

        Parameters
        ------------
        name: :class:`str`
            The extension name to load. It must be dot separated like
            regular Python imports if accessing a sub-module. e.g.
            ``foo.test`` if you want to import ``foo/test.py``.
        package: Optional[:class:`str`]
            The package name to resolve relative imports with.
            This is required when loading an extension using a relative path, e.g ``.foo.test``.
            Defaults to ``None``.

            .. versionadded:: 1.7

        Raises
        --------
        ExtensionNotFound
            The extension could not be imported.
            This is also raised if the name of the extension could not
            be resolved using the provided ``package`` parameter.
        ExtensionAlreadyLoaded
            The extension is already loaded.
        NoEntryPointError
            The extension does not have a setup function.
        ExtensionFailed
            The extension or its setup function had an execution error.
        """
        ...
    
    async def unload_extension(self, name: str, *, package: Optional[str] = ...) -> None:
        """|coro|

        Unloads an extension.

        When the extension is unloaded, all commands, listeners, and cogs are
        removed from the bot and the module is un-imported.

        The extension can provide an optional global function, ``teardown``,
        to do miscellaneous clean-up if necessary. This function takes a single
        parameter, the ``bot``, similar to ``setup`` from
        :meth:`~.Bot.load_extension`.

        .. versionchanged:: 2.0

            This method is now a :term:`coroutine`.

        Parameters
        ------------
        name: :class:`str`
            The extension name to unload. It must be dot separated like
            regular Python imports if accessing a sub-module. e.g.
            ``foo.test`` if you want to import ``foo/test.py``.
        package: Optional[:class:`str`]
            The package name to resolve relative imports with.
            This is required when unloading an extension using a relative path, e.g ``.foo.test``.
            Defaults to ``None``.

            .. versionadded:: 1.7

        Raises
        -------
        ExtensionNotFound
            The name of the extension could not
            be resolved using the provided ``package`` parameter.
        ExtensionNotLoaded
            The extension was not loaded.
        """
        ...
    
    async def reload_extension(self, name: str, *, package: Optional[str] = ...) -> None:
        """|coro|

        Atomically reloads an extension.

        This replaces the extension with the same extension, only refreshed. This is
        equivalent to a :meth:`unload_extension` followed by a :meth:`load_extension`
        except done in an atomic way. That is, if an operation fails mid-reload then
        the bot will roll-back to the prior working state.

        Parameters
        ------------
        name: :class:`str`
            The extension name to reload. It must be dot separated like
            regular Python imports if accessing a sub-module. e.g.
            ``foo.test`` if you want to import ``foo/test.py``.
        package: Optional[:class:`str`]
            The package name to resolve relative imports with.
            This is required when reloading an extension using a relative path, e.g ``.foo.test``.
            Defaults to ``None``.

            .. versionadded:: 1.7

        Raises
        -------
        ExtensionNotLoaded
            The extension was not loaded.
        ExtensionNotFound
            The extension could not be imported.
            This is also raised if the name of the extension could not
            be resolved using the provided ``package`` parameter.
        NoEntryPointError
            The extension does not have a setup function.
        ExtensionFailed
            The extension setup function had an execution error.
        """
        ...
    
    @property
    def extensions(self) -> Mapping[str, types.ModuleType]:
        """Mapping[:class:`str`, :class:`py:types.ModuleType`]: A read-only mapping of extension name to extension."""
        ...
    
    @property
    def help_command(self) -> Optional[HelpCommand]:
        ...
    
    @help_command.setter
    def help_command(self, value: Optional[HelpCommand]) -> None:
        ...
    
    @property
    def tree(self) -> app_commands.CommandTree[Self]:
        """:class:`~discord.app_commands.CommandTree`: The command tree responsible for handling the application commands
        in this bot.

        .. versionadded:: 2.0
        """
        ...
    
    async def get_prefix(self, message: Message, /) -> Union[List[str], str]:
        """|coro|

        Retrieves the prefix the bot is listening to
        with the message as a context.

        .. versionchanged:: 2.0

            ``message`` parameter is now positional-only.

        Parameters
        -----------
        message: :class:`discord.Message`
            The message context to get the prefix of.

        Returns
        --------
        Union[List[:class:`str`], :class:`str`]
            A list of prefixes or a single prefix that the bot is
            listening for.
        """
        ...
    
    @overload
    async def get_context(self, origin: Union[Message, Interaction], /) -> Context[Self]:
        ...
    
    @overload
    async def get_context(self, origin: Union[Message, Interaction], /, *, cls: Type[ContextT]) -> ContextT:
        ...
    
    async def get_context(self, origin: Union[Message, Interaction], /, *, cls: Type[ContextT] = ...) -> Any:
        r"""|coro|

        Returns the invocation context from the message or interaction.

        This is a more low-level counter-part for :meth:`.process_commands`
        to allow users more fine grained control over the processing.

        The returned context is not guaranteed to be a valid invocation
        context, :attr:`.Context.valid` must be checked to make sure it is.
        If the context is not valid then it is not a valid candidate to be
        invoked under :meth:`~.Bot.invoke`.

        .. note::

            In order for the custom context to be used inside an interaction-based
            context (such as :class:`HybridCommand`) then this method must be
            overridden to return that class.

        .. versionchanged:: 2.0

            ``message`` parameter is now positional-only and renamed to ``origin``.

        Parameters
        -----------
        origin: Union[:class:`discord.Message`, :class:`discord.Interaction`]
            The message or interaction to get the invocation context from.
        cls
            The factory class that will be used to create the context.
            By default, this is :class:`.Context`. Should a custom
            class be provided, it must be similar enough to :class:`.Context`\'s
            interface.

        Returns
        --------
        :class:`.Context`
            The invocation context. The type of this can change via the
            ``cls`` parameter.
        """
        ...
    
    async def invoke(self, ctx: Context[BotT], /) -> None:
        """|coro|

        Invokes the command given under the invocation context and
        handles all the internal event dispatch mechanisms.

        .. versionchanged:: 2.0

            ``ctx`` parameter is now positional-only.

        Parameters
        -----------
        ctx: :class:`.Context`
            The invocation context to invoke.
        """
        ...
    
    async def process_commands(self, message: Message, /) -> None:
        """|coro|

        This function processes the commands that have been registered
        to the bot and other groups. Without this coroutine, none of the
        commands will be triggered.

        By default, this coroutine is called inside the :func:`.on_message`
        event. If you choose to override the :func:`.on_message` event, then
        you should invoke this coroutine as well.

        This is built using other low level tools, and is equivalent to a
        call to :meth:`~.Bot.get_context` followed by a call to :meth:`~.Bot.invoke`.

        This also checks if the message's author is a bot and doesn't
        call :meth:`~.Bot.get_context` or :meth:`~.Bot.invoke` if so.

        .. versionchanged:: 2.0

            ``message`` parameter is now positional-only.

        Parameters
        -----------
        message: :class:`discord.Message`
            The message to process commands for.
        """
        ...
    
    async def on_message(self, message: Message, /) -> None:
        ...
    


class Bot(BotBase, discord.Client):
    """Represents a Discord bot.

    This class is a subclass of :class:`discord.Client` and as a result
    anything that you can do with a :class:`discord.Client` you can do with
    this bot.

    This class also subclasses :class:`.GroupMixin` to provide the functionality
    to manage commands.

    Unlike :class:`discord.Client`, this class does not require manually setting
    a :class:`~discord.app_commands.CommandTree` and is automatically set upon
    instantiating the class.

    .. container:: operations

        .. describe:: async with x

            Asynchronously initialises the bot and automatically cleans up.

            .. versionadded:: 2.0

    Attributes
    -----------
    command_prefix
        The command prefix is what the message content must contain initially
        to have a command invoked. This prefix could either be a string to
        indicate what the prefix should be, or a callable that takes in the bot
        as its first parameter and :class:`discord.Message` as its second
        parameter and returns the prefix. This is to facilitate "dynamic"
        command prefixes. This callable can be either a regular function or
        a coroutine.

        An empty string as the prefix always matches, enabling prefix-less
        command invocation. While this may be useful in DMs it should be avoided
        in servers, as it's likely to cause performance issues and unintended
        command invocations.

        The command prefix could also be an iterable of strings indicating that
        multiple checks for the prefix should be used and the first one to
        match will be the invocation prefix. You can get this prefix via
        :attr:`.Context.prefix`.

        .. note::

            When passing multiple prefixes be careful to not pass a prefix
            that matches a longer prefix occurring later in the sequence.  For
            example, if the command prefix is ``('!', '!?')``  the ``'!?'``
            prefix will never be matched to any message as the previous one
            matches messages starting with ``!?``. This is especially important
            when passing an empty string, it should always be last as no prefix
            after it will be matched.
    case_insensitive: :class:`bool`
        Whether the commands should be case insensitive. Defaults to ``False``. This
        attribute does not carry over to groups. You must set it to every group if
        you require group commands to be case insensitive as well.
    description: :class:`str`
        The content prefixed into the default help message.
    help_command: Optional[:class:`.HelpCommand`]
        The help command implementation to use. This can be dynamically
        set at runtime. To remove the help command pass ``None``. For more
        information on implementing a help command, see :ref:`ext_commands_help_command`.
    owner_id: Optional[:class:`int`]
        The user ID that owns the bot. If this is not set and is then queried via
        :meth:`.is_owner` then it is fetched automatically using
        :meth:`~.Bot.application_info`.
    owner_ids: Optional[Collection[:class:`int`]]
        The user IDs that owns the bot. This is similar to :attr:`owner_id`.
        If this is not set and the application is team based, then it is
        fetched automatically using :meth:`~.Bot.application_info`.
        For performance reasons it is recommended to use a :class:`set`
        for the collection. You cannot set both ``owner_id`` and ``owner_ids``.

        .. versionadded:: 1.3
    strip_after_prefix: :class:`bool`
        Whether to strip whitespace characters after encountering the command
        prefix. This allows for ``!   hello`` and ``!hello`` to both work if
        the ``command_prefix`` is set to ``!``. Defaults to ``False``.

        .. versionadded:: 1.7
    tree_cls: Type[:class:`~discord.app_commands.CommandTree`]
        The type of application command tree to use. Defaults to :class:`~discord.app_commands.CommandTree`.

        .. versionadded:: 2.0
    """
    ...


class AutoShardedBot(BotBase, discord.AutoShardedClient):
    """This is similar to :class:`.Bot` except that it is inherited from
    :class:`discord.AutoShardedClient` instead.

    .. container:: operations

        .. describe:: async with x

            Asynchronously initialises the bot and automatically cleans.

            .. versionadded:: 2.0
    """
    ...


