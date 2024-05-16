"""
This type stub file was generated by pyright.
"""

import typing
import pathlib
import discord

"""
    Main module for supporting predicate-style assertions.
    Handles checking various state matches the desired outcome.

    All verify types should be re-exported at ``discord.ext.test``, this is the primary
    entry point for assertions in the library

    See also:
        :mod:`discord.ext.test.runner`
"""
class _Undef:
    _singleton = ...
    def __new__(cls): # -> Self:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    


_undefined = ...
class VerifyMessage:
    """
        Builder for message verifications. When done building, should be asserted.

        **Example**:
        ``assert dpytest.verify().message().content("Hello World!")``
    """
    _invert: bool
    _contains: bool
    _peek: bool
    _nothing: bool
    _content: typing.Union[None, _Undef, str]
    _embed: typing.Union[None, _Undef, discord.Embed]
    _attachment: typing.Union[None, _Undef, str, pathlib.Path]
    def __init__(self) -> None:
        ...
    
    def __del__(self) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def contains(self) -> VerifyMessage:
        """
            Only check whether content/embed list/etc contain the desired input, not that they necessarily match
            exactly

        :return: Self for chaining
        """
        ...
    
    def peek(self) -> VerifyMessage:
        """
            Don't remove the verified message from the queue

        :return: Self for chaining
        """
        ...
    
    def nothing(self) -> VerifyMessage:
        """
            Check that no message was sent

        :return: Self for chaining
        """
        ...
    
    def content(self, content: typing.Optional[str]) -> VerifyMessage:
        """
            Check that the message content matches the input

        :param content: Content to match against, or None to ensure no content
        :return: Self for chaining
        """
        ...
    
    def embed(self, embed: typing.Optional[discord.Embed]) -> VerifyMessage:
        """
            Check that the message embed matches the input

        :param embed: Embed to match against, or None to ensure no embed
        :return: Self for chaining
        """
        ...
    
    def attachment(self, attach: typing.Optional[typing.Union[str, pathlib.Path]]) -> VerifyMessage:
        """
            Check that the message attachment matches the input

        :param attach: Attachment path to match against, or None to ensure no attachment
        :return: Self for chaining
        """
        ...
    


class VerifyActivity:
    """
        Builder for activity verifications. When done building, should be asserted

        **Example**:
        ``assert not dpytest.verify().activity().name("Foobar")``
    """
    def __init__(self) -> None:
        ...
    
    def __del__(self) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def matches(self, activity) -> VerifyActivity:
        """
            Ensure that the bot activity exactly matches the passed activity. Most restrictive possible check.

        :param activity: Activity to compare against
        :return: Self for chaining
        """
        ...
    
    def name(self, name: str) -> VerifyActivity:
        """
            Check that the activity name matches the input

        :param name: Name to match against
        :return: Self for chaining
        """
        ...
    
    def url(self, url: str) -> VerifyActivity:
        """
            Check the the activity url matches the input

        :param url: Url to match against
        :return: Self for chaining
        """
        ...
    
    def type(self, type: discord.ActivityType) -> VerifyActivity:
        """
            Check the activity type matches the input

        :param type: Type to match against
        :return: Self for chaining
        """
        ...
    


class Verify:
    """
        Base for all kinds of verification builders. Used as an
        intermediate step for the return of verify().
    """
    def __init__(self) -> None:
        ...
    
    def message(self) -> VerifyMessage:
        """
            Verify a message

        :return: Message verification builder
        """
        ...
    
    def activity(self) -> VerifyActivity:
        """
            Verify the bot's activity

        :return: Activity verification builder
        """
        ...
    


def verify() -> Verify:
    """
        Verification entry point. Call to begin building a verification.

        **Warning**: All verification builders do nothing until asserted, used in an if statement,
        or otherwise converted into a bool. They will raise RuntimeWarning if this isn't done to help
        catch possible errors.

    :return: Verification builder
    """
    ...

