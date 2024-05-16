# goob_ai

POC langchain rag discord chatbot. Heavily inspired by DiscordLangAgent


# Description

This is a Discord chatbot built with LangChain. It uses the Agent features in LangChain to create flexible conversation chains based on user input. The bot can interact with different language models and tools, and supports multiple API endpoints. It's designed with customization in mind, allowing for the addition and modification of features as needed.

## Testing

Attempting to test using https://dpytest.readthedocs.io/en/latest/

## Features

- **Chatting**: The bot can engage in conversation with users. It responds when its name is mentioned, when it's replied to, or when it's tagged with @botname. It ignores messages that start with . or / and messages that dont trigger a response are added to the chat history to give the bot conversation context.
- **Web Searches**: The bot can perform web searches using DuckDuckGo. The result is injected in to a conversation chain so that your bot will be able to talk about current events. (currently requires openai api key until I switch it to local models)
- **Text Summarization**: The bot can summarize text with a slash command.
- **Image Captioning**: The bot uses image captioning with a conversation chain to give it the illusion of seeing the images you post.
- **Instruct Mode**: This is a slash command that allows users to bypass the bot's personality and make it follow the instructions provided. The result is added to the chat history.
- **Various Commands**: The bot offers a range of slash commands for different purposes. Conversational commands like "listen-only" mode change the bot's default behavior, while developer commands like /reload and /sync provide control over the bot's operation.
- **API Endpoints and Language Model Integration**: The bot supports 3 different API endpoints - Oobabooga webui API, KoboldAI, and OpenAI. It can use local language models or OpenAI for generating responses.
- **Modular Design (Cogs)**: The bot is designed with modularity in mind, allowing for the easy addition of new features (cogs).

## .env File Setup

The `.env` file is used to store environment variables for your project. These variables can include API keys, database URIs, and other sensitive information that you don't want to hardcode into your application.

To set up the `.env` file:

1. Rename the `sample.env` file to `.env`.
2. Open the `.env` file and replace the placeholder values with your actual values.

Here's a breakdown of each item in the `.env` file:

- `DISCORD_BOT_TOKEN`: Your Discord bot token. This is required for your bot to log in to Discord.
- `OOBAENDPOINT`: The endpoint for the Oobabooga API. Follow the setup instructions for the API provided in its repository. Used for conversation LLM. Leave blank if you want to use KoboldAI api or OpenAI instead
- `KOBOLDENDPOINT`: The endpoint for the KoboldAI API. Follow the setup instructions for the API provided in its repository. Used for conversation LLM. Leave blank if you want to use Oobabooga's webui api or OpenAI instead
- `CHANNEL_ID`: The ID(s) of the text channel(s) you want the bot to watch and reply in. If you want to specify multiple channels, separate the IDs with a comma (e.g., 1121121529787338903,1121233456307904583).
- `OWNERS`: Your Discord user ID. This is not currently used anywhere.
- `OPENAI`: Your OpenAI API key. This is optional and is currently used only for the DuckDuckGo agent. It's also used for the conversation LLM if you don't specify KoboldAI or Oobabooga. You can get this from the OpenAI platform.
