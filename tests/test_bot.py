# """test_bot"""
# from __future__ import annotations
# import asyncio
# import os

# import pytest
# import torch

# from goob_ai import bot

# IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


# @pytest.mark.skipif(
#     os.getenv("GOOB_AI_BOT_SANITY"),  # noqa
#     reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
# )
# @pytest.mark.integration
# class TestBot:
#     @pytest.mark.asyncio
#     @pytest.mark.justaio
#     async def test_defaults(self, event_loop: asyncio.AbstractEventLoop) -> None:
#         # pytest-asyncio provides and manages the `event_loop`
#         # and it should be set as the default loop for this session
#         assert event_loop  # but it's not == asyncio.get_event_loop() ?

#         test_bot: bot.GoobBot = bot.GoobBot(loop=event_loop)

#         test_bot.intents.members = True  # pylint: disable=assigning-non-slot
#         # NOTE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
#         test_bot.version = "fake"
#         test_bot.guild_data = {}

#         # import bpdb

#         # bpdb.set_trace()

#         assert test_bot.version == "fake"
#         assert test_bot.description == "Better than the last one"
#         assert not test_bot.tasks
#         assert test_bot.num_workers == 3
#         assert not test_bot.job_queue

#         # This group of variables are used in the upscaling process
#         assert not test_bot.last_model
#         assert not test_bot.last_in_nc
#         assert not test_bot.last_out_nc
#         assert not test_bot.last_nf
#         assert not test_bot.last_nb
#         assert not test_bot.last_scale
#         assert not test_bot.last_kind
#         assert not test_bot.model
#         assert not test_bot.autocrop_model

#         # NOTE: these are temporary
#         assert (
#             test_bot.ml_models_path == "/Users/malcolm/dev/universityofprofessorex/cerebro-bot/goob_ai/ml_models/"
#         )

#         assert test_bot.my_custom_ml_models_path == "/Users/malcolm/Downloads/"

#         # import bpdb

#         # bpdb.set_trace()

#         if not torch.cuda.is_available():
#             assert str(test_bot.device) == "cpu"

#         # await test_bot.metrics_api.stop()
