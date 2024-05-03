"""goob_ai.utils.aiomodels"""
import typing

import aiohttp
import bs4
from bs4 import BeautifulSoup
from fuzzywuzzy import process
from markdownify import MarkdownConverter
from pytablewriter import TableWriterFactory

VALID_TABLE_HEADERS = ["Model Name", "author", "Scale", "Purpose (short)", "sample"]

MODEL_TABLE_LOOKUP = {
    "Universal Models": 8,
    "Realistic Photos": 10,
    "Art/Pixel Art": 13,
    "Anime": 16,
    "Manga": 18,
    "Cartoons": 20,
    # document.querySelector("#mw-content-text > div > table:nth-child(22)")
    "Digital Animation": 22,
    "Drawings": 24,
    # document.querySelector("#mw-content-text > div > table:nth-child(26)")
    "General Animation": 26,
    # document.querySelector("#mw-content-text > div > table:nth-child(29)")
    "Traditional Animation": 29,
    # document.querySelector("#mw-content-text > div > table:nth-child(34)")
    "JPEG Artifacts": 34,
    "Aliasing": 36,
    "GIF": 38,
    "DDS (BC1/DXT1, BC3/DXT5 Compression)": 40,
    "Dithering": 42,
    "Blurring": 44,
    "Banding": 46,
    "Halo Removal": 48,
    "Noise": 50,
    # document.querySelector("#mw-content-text > div > table:nth-child(53)")
    "Oversharpening": 53,
    "DeToon": 55,
    "Image De/Colorization": 59,
    "Images": 62,
    "Text": 65,
    "Inpainting": 67,
    "Fabric/Cloth": 69,
    "Alphas": 72,
    "CGI": 74,
    "Luminance/Chroma": 76,
    "Cats": 78,
    "Coins": 80,
    "Faces": 82,
    "Foliage/Ground": 84,
    "Game Screenshots": 87,
    "Normal Map/Bump Map Generation": 90,
    "Video Game Textures": 92,
    "Video Compression": 96,
    "VHS Tapes": 98,
    "Model Collections": 100,
    # document.querySelector("#mw-content-text > div > table:nth-child(113)")
    "CAIN Models": 113,
}

SELECTED_URL = "https://upscale.wiki/wiki/Model_Database"


async def get_site_content():
    async with aiohttp.ClientSession() as session:
        async with session.get(SELECTED_URL) as resp:
            text = await resp.read()

    return BeautifulSoup(text.decode("utf-8"), "lxml")


# SOURCE: https://gist.github.com/ergoithz/6cf043e3fdedd1b94fcf


def xpath_soup(element):
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        components.append(child.name if siblings == [child] else "%s[%d]" % (child.name, 1 + siblings.index(child)))
        child = parent
    components.reverse()
    return f'/{"/".join(components)}'


def get_all_tables(sites_soup: BeautifulSoup) -> bs4.element.ResultSet:
    """Find all h3 tags on page, and add them to an array, all of these items are the headers for model sections.

    Args:
        sites_soup (BeautifulSoup): BeautifulSoup parser object

    Returns:
        bs4.element.ResultSet: List containing string representations of the h3 names, eg "Universal Models"
    """
    return sites_soup.select("#mw-content-text > div > table:nth-child(n)")


def get_tables_by_name(sites_soup: BeautifulSoup, table_name: str) -> bs4.element.ResultSet:
    """Get html from table based on section name

    Args:
        sites_soup (BeautifulSoup): BeautifulSoup parser
        table_name (str): Name of section, eg. "CAIN Models"

    Returns:
        bs4.element.ResultSet: result set html
    """
    fuzzy_table_name = fuzzy_match_model_string(table_name)
    n = MODEL_TABLE_LOOKUP[fuzzy_table_name]
    js_path = f"#mw-content-text > div > table:nth-child({n})"
    return sites_soup.select(js_path)


def get_h3s(sites_soup: BeautifulSoup) -> bs4.element.ResultSet:
    """Find all h3 tags on page, and add them to an array, all of these items are the headers for model sections.

    Args:
        sites_soup (BeautifulSoup): BeautifulSoup parser object

    Returns:
        bs4.element.ResultSet: List containing string representations of the h3 names, eg "Universal Models"
    """
    return sites_soup.find_all("h3")


def get_sections(sites_soup: BeautifulSoup) -> typing.List[str]:
    """Find all h3 tags on page, and add them to an array, all of these items are the headers for model sections.

    Args:
        sites_soup (BeautifulSoup): BeautifulSoup parser object

    Returns:
        typing.List[str]: List containing string representations of the h3 names, eg "Universal Models"
    """
    all_model_sections = get_h3s(sites_soup)

    return [sec.text for sec in all_model_sections]


def list_all_model_types():
    return list(MODEL_TABLE_LOOKUP.keys())


def fuzzy_match_model_string(lookup: str) -> str:
    """Get the Levenshtein Distance of a string and return the correct value.

    Args:
        lookup (str): substring to look up, eg "anime"

    Returns:
        str: Returns actual value, eg "Anime"
    """
    model_list = list_all_model_types()
    res = process.extractOne(lookup, model_list)
    return res[0]


# @snoop
def get_result_set_sublist(records: bs4.element.ResultSet) -> typing.List[str]:
    # VALID_TABLE_HEADERS = ["Model Name", "author", "Scale", "Purpose (short)", "sample"]
    rows = []
    for row in records:
        # bpdb.set_trace()
        row_parser = row.find_all("td")
        # print(f" count = {count}, row = {row}")
        rows.append(
            [
                md_from_beautifulsoup(row_parser[0]),
                row_parser[1].get_text(strip=True),
                row_parser[2].get_text(strip=True),
                row_parser[5].get_text(strip=True),
                md_from_beautifulsoup(row_parser[9]),
            ]
        )

    # print(rows)
    return rows


# SOURCE: https://stackoverflow.com/questions/35755153/extract-only-specific-rows-and-columns-from-a-table-td-in-beautifulsoup-pytho


def get_html_table_headers(res_table: bs4.element.ResultSet):
    """Parse a result set and return only the values we care about

    Args:
        res_table (bs4.element.ResultSet): _description_
    """
    headers = [c.get_text(strip=True) for c in res_table[0].find("tr").find_all("th")]

    # only include the ones we care about
    final_headers = [f"{h}" for h in headers if h in VALID_TABLE_HEADERS]

    # get all table records and nuke the headers only
    all_table_records = res_table[0].find_all("tr")
    del all_table_records[0]

    table_rows_list = get_result_set_sublist(all_table_records)

    return final_headers, table_rows_list


# Create shorthand method for conversion


def md_from_beautifulsoup(sites_soup: BeautifulSoup, **options):
    """Converting BeautifulSoup objects"""
    return MarkdownConverter(**options).convert_soup(sites_soup)


def generate_markdown_table(table_name, final_table_headers, table_table_rows_list, margin):
    # SOURCE: https://github.com/thombashi/pytest-md-report/blob/aeff356c0b0831ad594cf5af45fca9e08dd1f92d/pytest_md_report/plugin.py
    writer = TableWriterFactory.create_from_format_name("md")
    writer.table_name = fuzzy_match_model_string(table_name)
    writer.margin = margin
    writer.headers = final_table_headers
    writer.value_matrix = table_table_rows_list

    return writer.dumps()


async def upscale_model_markdown(fuzzy_search_str: str):
    sites_soup = await get_site_content()
    # model_sections_list = get_sections(sites_soup)
    # all_th_list = sites_soup.find_all('th')
    model_table = get_tables_by_name(sites_soup, fuzzy_search_str)
    # anime_markdown = md_from_beautifulsoup(model_table[0])
    final_table_headers, table_table_rows_list = get_html_table_headers(model_table)
    return generate_markdown_table(fuzzy_search_str, final_table_headers, table_table_rows_list, 1)


# # TODO: implement multi download https://stackoverflow.com/questions/64282309/aiohttp-download-large-list-of-pdf-files
# async def async_download_file(data: dict, dl_dir="./"):
#     async with aiohttp.ClientSession() as session:
#         url: str = data["url"]
#         username: str = data["tweet"]["username"]
#         p = pathlib.Path(url)
#         p_dl_dir = pathlib.Path(dl_dir)
#         full_path_dl_dir = f"{p_dl_dir.absolute()}"
#         LOGGER.debug(f"Downloading {url} to {full_path_dl_dir}/{p.name}")
#         async with session.get(url) as resp:
#             content = await resp.read()

#             # Check everything went well
#             if resp.status != 200:
#                 LOGGER.error(f"Download failed: {resp.status}")
#                 return

#             if resp.status == 200:
#                 async with aiofiles.open(f"{full_path_dl_dir}/{p.name}", mode="+wb") as f:
#                     await f.write(content)
#                     # No need to use close(f) when using with statement
#                 # f = await aiofiles.open(f"{full_path_dl_dir}/{p.name}", mode='wb')
#                 # await f.write(await resp.read())
#                 # await f.close()
# loop = asyncio.get_event_loop()
# sites_soup = loop.run_until_complete(get_site_content())
# loop.close()


# # print(sites_soup)

# # all_tables = sites_soup.find_all('table')

# fuzzy_search_str = "anime"


# # print(markdown_str_final)
