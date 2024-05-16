"""
This type stub file was generated by pyright.
"""

from . import types

"""
HTML5 color algorithms.

Note that these functions are written in a way that may seem strange to
developers familiar with Python, because they do not use the most
efficient or idiomatic way of accomplishing their tasks. This is
because, for compliance, these functions are written as literal
translations into Python of the algorithms in HTML5:

https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#colours

For ease of understanding, the relevant steps of the algorithm from
the standard are included as comments interspersed in the
implementation.

"""
def html5_parse_simple_color(value: str) -> types.HTML5SimpleColor:
    """
    Apply the HTML5 simple color parsing algorithm.

    Examples:

    .. doctest::

        >>> html5_parse_simple_color("#ffffff")
        HTML5SimpleColor(red=255, green=255, blue=255)
        >>> html5_parse_simple_color("#fff")
        Traceback (most recent call last):
            ...
        ValueError: An HTML5 simple color must be a string seven characters long.

    :param value: The color to parse.
    :type value: :class:`str`, which must consist of exactly
        the character ``"#"`` followed by six hexadecimal digits
    :raises ValueError: when the given value is not a Unicode string of
       length 7, consisting of exactly the character ``#`` followed by
       six hexadecimal digits.


    """
    ...

def html5_serialize_simple_color(simple_color: types.IntTuple) -> str:
    """
    Apply the HTML5 simple color serialization algorithm.

    Examples:

    .. doctest::

        >>> html5_serialize_simple_color((0, 0, 0))
        '#000000'
        >>> html5_serialize_simple_color((255, 255, 255))
        '#ffffff'

    :param simple_color: The color to serialize.

    """
    ...

def html5_parse_legacy_color(value: str) -> types.HTML5SimpleColor:
    """
    Apply the HTML5 legacy color parsing algorithm.

    Note that, since this algorithm is intended to handle many types of
    malformed color values present in real-world Web documents, it is
    *extremely* forgiving of input, but the results of parsing inputs
    with high levels of "junk" (i.e., text other than a color value)
    may be surprising.

    Examples:

    .. doctest::

        >>> html5_parse_legacy_color("black")
        HTML5SimpleColor(red=0, green=0, blue=0)
        >>> html5_parse_legacy_color("chucknorris")
        HTML5SimpleColor(red=192, green=0, blue=0)
        >>> html5_parse_legacy_color("Window")
        HTML5SimpleColor(red=0, green=13, blue=0)

    :param value: The color to parse.

    :raises ValueError: when the given value is not a Unicode string, when it is the
       empty string, or when it is precisely the string ``"transparent"``.

    """
    ...

