"""
This type stub file was generated by pyright.
"""

import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, overload
from .enums import ActivityType
from .colour import Colour
from .partial_emoji import PartialEmoji
from .types.activity import Activity as ActivityPayload
from .state import ConnectionState

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
__all__ = ('BaseActivity', 'Activity', 'Streaming', 'Game', 'Spotify', 'CustomActivity')
if TYPE_CHECKING:
    ...
class BaseActivity:
    """The base activity that all user-settable activities inherit from.
    A user-settable activity is one that can be used in :meth:`Client.change_presence`.

    The following types currently count as user-settable:

    - :class:`Activity`
    - :class:`Game`
    - :class:`Streaming`
    - :class:`CustomActivity`

    Note that although these types are considered user-settable by the library,
    Discord typically ignores certain combinations of activity depending on
    what is currently set. This behaviour may change in the future so there are
    no guarantees on whether Discord will actually let you set these types.

    .. versionadded:: 1.3
    """
    __slots__ = ...
    def __init__(self, **kwargs: Any) -> None:
        ...
    
    @property
    def created_at(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user started doing this activity in UTC.

        .. versionadded:: 1.3
        """
        ...
    
    def to_dict(self) -> ActivityPayload:
        ...
    


class Activity(BaseActivity):
    """Represents an activity in Discord.

    This could be an activity such as streaming, playing, listening
    or watching.

    For memory optimisation purposes, some activities are offered in slimmed
    down versions:

    - :class:`Game`
    - :class:`Streaming`

    Attributes
    ------------
    application_id: Optional[:class:`int`]
        The application ID of the game.
    name: Optional[:class:`str`]
        The name of the activity.
    url: Optional[:class:`str`]
        A stream URL that the activity could be doing.
    type: :class:`ActivityType`
        The type of activity currently being done.
    state: Optional[:class:`str`]
        The user's current state. For example, "In Game".
    details: Optional[:class:`str`]
        The detail of the user's current activity.
    timestamps: :class:`dict`
        A dictionary of timestamps. It contains the following optional keys:

        - ``start``: Corresponds to when the user started doing the
          activity in milliseconds since Unix epoch.
        - ``end``: Corresponds to when the user will finish doing the
          activity in milliseconds since Unix epoch.

    assets: :class:`dict`
        A dictionary representing the images and their hover text of an activity.
        It contains the following optional keys:

        - ``large_image``: A string representing the ID for the large image asset.
        - ``large_text``: A string representing the text when hovering over the large image asset.
        - ``small_image``: A string representing the ID for the small image asset.
        - ``small_text``: A string representing the text when hovering over the small image asset.

    party: :class:`dict`
        A dictionary representing the activity party. It contains the following optional keys:

        - ``id``: A string representing the party ID.
        - ``size``: A list of up to two integer elements denoting (current_size, maximum_size).
    buttons: List[:class:`str`]
        A list of strings representing the labels of custom buttons shown in a rich presence.

        .. versionadded:: 2.0

    emoji: Optional[:class:`PartialEmoji`]
        The emoji that belongs to this activity.
    """
    __slots__ = ...
    def __init__(self, **kwargs: Any) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    @property
    def start(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user started doing this activity in UTC, if applicable."""
        ...
    
    @property
    def end(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user will stop doing this activity in UTC, if applicable."""
        ...
    
    @property
    def large_image_url(self) -> Optional[str]:
        """Optional[:class:`str`]: Returns a URL pointing to the large image asset of this activity, if applicable."""
        ...
    
    @property
    def small_image_url(self) -> Optional[str]:
        """Optional[:class:`str`]: Returns a URL pointing to the small image asset of this activity, if applicable."""
        ...
    
    @property
    def large_image_text(self) -> Optional[str]:
        """Optional[:class:`str`]: Returns the large image asset hover text of this activity, if applicable."""
        ...
    
    @property
    def small_image_text(self) -> Optional[str]:
        """Optional[:class:`str`]: Returns the small image asset hover text of this activity, if applicable."""
        ...
    


class Game(BaseActivity):
    """A slimmed down version of :class:`Activity` that represents a Discord game.

    This is typically displayed via **Playing** on the official Discord client.

    .. container:: operations

        .. describe:: x == y

            Checks if two games are equal.

        .. describe:: x != y

            Checks if two games are not equal.

        .. describe:: hash(x)

            Returns the game's hash.

        .. describe:: str(x)

            Returns the game's name.

    Parameters
    -----------
    name: :class:`str`
        The game's name.

    Attributes
    -----------
    name: :class:`str`
        The game's name.
    """
    __slots__ = ...
    def __init__(self, name: str, **extra: Any) -> None:
        ...
    
    @property
    def type(self) -> ActivityType:
        """:class:`ActivityType`: Returns the game's type. This is for compatibility with :class:`Activity`.

        It always returns :attr:`ActivityType.playing`.
        """
        ...
    
    @property
    def start(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user started playing this game in UTC, if applicable."""
        ...
    
    @property
    def end(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user will stop playing this game in UTC, if applicable."""
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    


class Streaming(BaseActivity):
    """A slimmed down version of :class:`Activity` that represents a Discord streaming status.

    This is typically displayed via **Streaming** on the official Discord client.

    .. container:: operations

        .. describe:: x == y

            Checks if two streams are equal.

        .. describe:: x != y

            Checks if two streams are not equal.

        .. describe:: hash(x)

            Returns the stream's hash.

        .. describe:: str(x)

            Returns the stream's name.

    Attributes
    -----------
    platform: Optional[:class:`str`]
        Where the user is streaming from (ie. YouTube, Twitch).

        .. versionadded:: 1.3

    name: Optional[:class:`str`]
        The stream's name.
    details: Optional[:class:`str`]
        An alias for :attr:`name`
    game: Optional[:class:`str`]
        The game being streamed.

        .. versionadded:: 1.3

    url: :class:`str`
        The stream's URL.
    assets: :class:`dict`
        A dictionary comprising of similar keys than those in :attr:`Activity.assets`.
    """
    __slots__ = ...
    def __init__(self, *, name: Optional[str], url: str, **extra: Any) -> None:
        ...
    
    @property
    def type(self) -> ActivityType:
        """:class:`ActivityType`: Returns the game's type. This is for compatibility with :class:`Activity`.

        It always returns :attr:`ActivityType.streaming`.
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def twitch_name(self) -> Optional[str]:
        """Optional[:class:`str`]: If provided, the twitch name of the user streaming.

        This corresponds to the ``large_image`` key of the :attr:`Streaming.assets`
        dictionary if it starts with ``twitch:``. Typically set by the Discord client.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    


class Spotify:
    """Represents a Spotify listening activity from Discord. This is a special case of
    :class:`Activity` that makes it easier to work with the Spotify integration.

    .. container:: operations

        .. describe:: x == y

            Checks if two activities are equal.

        .. describe:: x != y

            Checks if two activities are not equal.

        .. describe:: hash(x)

            Returns the activity's hash.

        .. describe:: str(x)

            Returns the string 'Spotify'.
    """
    __slots__ = ...
    def __init__(self, **data: Any) -> None:
        ...
    
    @property
    def type(self) -> ActivityType:
        """:class:`ActivityType`: Returns the activity's type. This is for compatibility with :class:`Activity`.

        It always returns :attr:`ActivityType.listening`.
        """
        ...
    
    @property
    def created_at(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user started listening in UTC.

        .. versionadded:: 1.3
        """
        ...
    
    @property
    def colour(self) -> Colour:
        """:class:`Colour`: Returns the Spotify integration colour, as a :class:`Colour`.

        There is an alias for this named :attr:`color`"""
        ...
    
    @property
    def color(self) -> Colour:
        """:class:`Colour`: Returns the Spotify integration colour, as a :class:`Colour`.

        There is an alias for this named :attr:`colour`"""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    @property
    def name(self) -> str:
        """:class:`str`: The activity's name. This will always return "Spotify"."""
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def title(self) -> str:
        """:class:`str`: The title of the song being played."""
        ...
    
    @property
    def artists(self) -> List[str]:
        """List[:class:`str`]: The artists of the song being played."""
        ...
    
    @property
    def artist(self) -> str:
        """:class:`str`: The artist of the song being played.

        This does not attempt to split the artist information into
        multiple artists. Useful if there's only a single artist.
        """
        ...
    
    @property
    def album(self) -> str:
        """:class:`str`: The album that the song being played belongs to."""
        ...
    
    @property
    def album_cover_url(self) -> str:
        """:class:`str`: The album cover image URL from Spotify's CDN."""
        ...
    
    @property
    def track_id(self) -> str:
        """:class:`str`: The track ID used by Spotify to identify this song."""
        ...
    
    @property
    def track_url(self) -> str:
        """:class:`str`: The track URL to listen on Spotify.

        .. versionadded:: 2.0
        """
        ...
    
    @property
    def start(self) -> datetime.datetime:
        """:class:`datetime.datetime`: When the user started playing this song in UTC."""
        ...
    
    @property
    def end(self) -> datetime.datetime:
        """:class:`datetime.datetime`: When the user will stop playing this song in UTC."""
        ...
    
    @property
    def duration(self) -> datetime.timedelta:
        """:class:`datetime.timedelta`: The duration of the song being played."""
        ...
    
    @property
    def party_id(self) -> str:
        """:class:`str`: The party ID of the listening party."""
        ...
    


class CustomActivity(BaseActivity):
    """Represents a custom activity from Discord.

    .. container:: operations

        .. describe:: x == y

            Checks if two activities are equal.

        .. describe:: x != y

            Checks if two activities are not equal.

        .. describe:: hash(x)

            Returns the activity's hash.

        .. describe:: str(x)

            Returns the custom status text.

    .. versionadded:: 1.3

    Attributes
    -----------
    name: Optional[:class:`str`]
        The custom activity's name.
    emoji: Optional[:class:`PartialEmoji`]
        The emoji to pass to the activity, if any.
    """
    __slots__ = ...
    def __init__(self, name: Optional[str], *, emoji: Optional[PartialEmoji] = ..., **extra: Any) -> None:
        ...
    
    @property
    def type(self) -> ActivityType:
        """:class:`ActivityType`: Returns the activity's type. This is for compatibility with :class:`Activity`.

        It always returns :attr:`ActivityType.custom`.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    


ActivityTypes = Union[Activity, Game, CustomActivity, Streaming, Spotify]
@overload
def create_activity(data: ActivityPayload, state: ConnectionState) -> ActivityTypes:
    ...

@overload
def create_activity(data: None, state: ConnectionState) -> None:
    ...

def create_activity(data: Optional[ActivityPayload], state: ConnectionState) -> Optional[ActivityTypes]:
    ...

