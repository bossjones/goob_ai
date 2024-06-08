```

----------------------------------- generated xml file: /Users/malcolm/dev/bossjones/goob_ai/junit/test-results.xml -----------------------------------
================================================================ slowest 10 durations =================================================================
6.03s call     tests/clients/http_client_test.py::test_post_bad_http_code
4.05s call     tests/backend/cache/test_goobredis_session.py::test_redis_session_manager_utility
2.05s call     tests/backend/cache/test_goobredis.py::test_redis_ops
1.00s call     tests/test_async_typer.py::test_async_command
1.00s call     tests/utils/test__async.py::TestUtilsAsync::test_to_async
0.14s call     tests/test_bot.py::TestBot::test_defaults
0.13s call     tests/backend/cache/test_goobredis.py::test_redis_pubsub

(3 durations < 0.05s hidden.  Use -vv to show these durations.)
=============================================================== short test summary info ===============================================================
FAILED tests/utils/test_file_functions.py::test_aio_read_jsonfile - AttributeError: __aenter__
FAILED tests/utils/test_file_functions.py::test_aio_json_loads - TypeError: the JSON object must be str, bytes or bytearray, not AsyncMock
FAILED tests/utils/test_file_functions.py::test_run_aio_json_loads - TypeError: the JSON object must be str, bytes or bytearray, not AsyncMock
FAILED tests/utils/test_file_functions.py::test_filter_pth - AssertionError: assert [] == ['file1.pth']
FAILED tests/utils/test_file_functions.py::test_filter_json - AssertionError: assert [] == ['file1.json']
FAILED tests/utils/test_file_functions.py::test_rename_without_cachebuster - AssertionError: assert [] == ['file1', 'file2']
FAILED tests/utils/test_file_functions.py::test_filter_videos - AssertionError: assert [] == ['file1.mp4']
FAILED tests/utils/test_file_functions.py::test_filter_audio - AssertionError: assert [] == ['file1.mp3']
FAILED tests/utils/test_file_functions.py::test_filter_gif - AssertionError: assert [] == ['file1.gif']
FAILED tests/utils/test_file_functions.py::test_filter_mkv - AssertionError: assert [] == ['file1.mkv']
FAILED tests/utils/test_file_functions.py::test_filter_m3u8 - AssertionError: assert [] == ['file1.m3u8']
FAILED tests/utils/test_file_functions.py::test_filter_webm - AssertionError: assert [] == ['file1.webm']
FAILED tests/utils/test_file_functions.py::test_filter_images - AssertionError: assert [] == ['file1.png']
FAILED tests/utils/test_file_functions.py::test_filter_pdfs - AssertionError: assert [] == ['file1.pdf']
FAILED tests/utils/test_file_functions.py::test_sort_dataframe - AssertionError: Expected DataFrame:    col1  col2
FAILED tests/utils/test_file_functions.py::test_tree - AssertionError: assert [PosixPath('/...ob_ai/file1')] == [PosixPath('f...Path('file1')]
FAILED tests/utils/test_file_functions.py::test_format_size - AssertionError: Expected: 1 KB, but got: 1024 B
FAILED tests/utils/test_file_functions.py::test_aiowrite_file - AttributeError: __aenter__
FAILED tests/utils/test_file_functions.py::test_airead_file - AttributeError: __aenter__

Results (17.62s):
      64 passed
      19 failed
         - tests/utils/test_file_functions.py:55 test_aio_read_jsonfile
         - tests/utils/test_file_functions.py:65 test_aio_json_loads
         - tests/utils/test_file_functions.py:75 test_run_aio_json_loads
         - tests/utils/test_file_functions.py:107 test_filter_pth
         - tests/utils/test_file_functions.py:112 test_filter_json
         - tests/utils/test_file_functions.py:117 test_rename_without_cachebuster
         - tests/utils/test_file_functions.py:124 test_filter_videos
         - tests/utils/test_file_functions.py:129 test_filter_audio
         - tests/utils/test_file_functions.py:134 test_filter_gif
         - tests/utils/test_file_functions.py:139 test_filter_mkv
         - tests/utils/test_file_functions.py:144 test_filter_m3u8
         - tests/utils/test_file_functions.py:149 test_filter_webm
         - tests/utils/test_file_functions.py:154 test_filter_images
         - tests/utils/test_file_functions.py:159 test_filter_pdfs
         - tests/utils/test_file_functions.py:189 test_sort_dataframe
         - tests/utils/test_file_functions.py:263 test_tree
         - tests/utils/test_file_functions.py:270 test_format_size
         - tests/utils/test_file_functions.py:276 test_aiowrite_file
```
