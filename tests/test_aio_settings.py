"""test_settings"""
# pylint: disable=no-self-use
from __future__ import annotations
import os
from pathlib import Path, PosixPath

import pytest

from goob_ai import aio_settings
from goob_ai.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))

# TODO: Make sure os,environ unsets values while running tests
@pytest.mark.unittest
class TestSettings:
    def test_defaults(
        self,
    ) -> None:
        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()
        assert test_settings.monitor_host == "localhost"
        assert test_settings.monitor_port == 50102
        assert test_settings.prefix == "/"
        assert test_settings.discord_admin_user_id == 3282
        assert test_settings.discord_general_channel == 908894727779258390
        assert test_settings.discord_admin_user_invited == False
        assert test_settings.better_exceptions == 1
        assert test_settings.pythonasynciodebug == 1
        assert test_settings.globals_try_patchmatch == True
        assert test_settings.globals_always_use_cpu == False
        assert test_settings.globals_internet_available == True
        assert test_settings.globals_full_precision == False
        assert test_settings.globals_ckpt_convert == False
        assert test_settings.globals_log_tokenization == False
        assert test_settings.redis_host == "localhost"
        assert test_settings.redis_port == 7600
        assert test_settings.redis_user is None
        assert test_settings.redis_pass is None
        assert test_settings.redis_base is None
        assert str(test_settings.redis_url) == "redis://localhost:7600"
