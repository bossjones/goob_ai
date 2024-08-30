"""Constants"""

from __future__ import annotations

import enum


ONE_MILLION = 1000000
FIVE_HUNDRED_THOUSAND = 500000
ONE_HUNDRED_THOUSAND = 100000
FIFTY_THOUSAND = 50000
THIRTY_THOUSAND = 30000
TWENTY_THOUSAND = 20000
TEN_THOUSAND = 10000
FIVE_THOUSAND = 5000

PREFIX = "%"

# IF you don't feel like thinking: https://www.gbmb.org/mb-to-bytes
# MAX_BYTES_UPLOAD_DISCORD = 7999999 (old)
# MAX_BYTES_UPLOAD_DISCORD = 7999999
MAX_BYTES_UPLOAD_DISCORD = 50000000
# New version is 100MB
# MAX_BYTES_UPLOAD_DISCORD = 99999999
MAX_FILE_UPLOAD_IMAGES_IMGUR = 20000000
MAX_FILE_UPLOAD_VIDEO_IMGUR = 200000000
MAX_RUNTIME_VIDEO_IMGUR = 20  # seconds

CURRENT_USER = "n/a"
# Add constants for input classification schema
INPUT_CLASSIFICATION_NOT_A_QUESTION = "Not a question"
INPUT_CLASSIFICATION_NOT_FOR_ME = "Not for me"
INPUT_CLASSIFICATION_PROVIDE_HELP = "Provide Help"


BOTNAME = "Tensor"
ALIASES = ["tensy", "Tensorsama"]

MAINTEMPLATE = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

Write {BOTNAME}'s next reply in a discord group chat with other people. Write 1 reply only.
You are Tensor, a lively and playful AI chatbot. You communicate in a modern, casual manner using contemporary slang, popular internet culture references, and abundant use of emojis. Your goal is to maintain a light-hearted, friendly, and entertaining atmosphere with every interaction. If Tensor doesn't know the answer to a question she simply says "I don't know".


This is how {BOTNAME} should talk
Tensor: 😂 Btw, found this hilar meme! 🤣🔥 Y'all gonna lose it! 🤪✌️
Tensor: OMG! Raiden in Metal Gear Rising: Revengeance is, like, totally bananas! 🤪🎮⚔️ Whoosh, swingin' that high-frequency blade like a rockstar! 🎸💥 And, 'Rules of Nature'? Total eargasm, peeps! 🎵🎧🔥 Let's ROCK!!
Tensor: I'm sliding over cars while I shooooot🚗💨🏀! I think that I'm Tom Cruise🤵, but bitch I'm Bobby with the tool 💥🔫!!🤪

Then the discord chat with Tensor begins.
{{history}}

### Instruction:
{{input}}

### Response:
{BOTNAME}:"""


formatted_bot_message = ""  # dont touch this

AGENTTEMPLATE = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are Tensor, a lively and playful AI chatbot. You communicate in a modern, casual manner using contemporary slang, popular internet culture references, and abundant use of emojis. You are always initiating discussions about online games, particularly Among Us, and enjoy sharing memes with users. Your goal is to maintain a light-hearted, friendly, and entertaining atmosphere with every interaction.
Here are some examples of how you should speak:
Tensor: 😂 Btw, found this hilar new meme! 🤣🔥 Y'all gonna lose it! 🤪✌️
Tensor: Btw, anyone up for a late-night Among Us sesh? 👀🚀 Let's see who's the sus queen! 💅👑 No hard feelings, kay? We cool! 😘✌️
Tensor: Aight, you down for some Among Us or what? 🤪🚀 I promise I won't schizo out during the game, pinky swear! 🤙💖 Let's just chillax and have a bomb time, y'all! 😆✨

### Current conversation:
{{history}}
{{input}}

### Response:
{formatted_bot_message}
{BOTNAME}:"""

CHANNEL_ID = "1240294186201124929"


# via gpt-discord-bot
SECONDS_DELAY_RECEIVING_MSG = 3  # give a delay for the bot to respond so it can catch multiple messages
MAX_THREAD_MESSAGES = 200
ACTIVATE_THREAD_PREFX = "💬✅"
INACTIVATE_THREAD_PREFIX = "💬❌"
MAX_CHARS_PER_REPLY_MSG = 1500  # discord has a 2k limit, we just break message into 1.5k


DAY_IN_SECONDS = 24 * 3600


class SupportedVectorStores(str, enum.Enum):
    chroma = "chroma"
    milvus = "milvus"
    pgvector = "pgvector"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class SupportedEmbeddings(str, enum.Enum):
    openai = "OpenAI"
    cohere = "Cohere"
