# old cropper data for pennywise

```
atar': '239d07cd2a9146198d455d96010e8e94'}, 'attachments': [], 'guild_id': '908894726793601094'}}
2024-06-08 23:51:26.801 | DEBUG    | discord.client:dispatch:462 - Dispatching event socket_event_type
2024-06-08 23:51:26.801 | DEBUG    | discord.client:dispatch:462 - Dispatching event message
2024-06-08 23:51:26.810 | DEBUG    | discord.http:request:625 - POST https://discord.com/api/v10/channels/1072528677620949102/messages with {"embeds":[{"type":"rich","description":"Autocropping ['/tmp/tmpx7afqiwm/orig_1249209208058937444_screenshot_image_larger00013.PNG']...."}],"content":null,"components":[],"tts":false} has returned 200
2024-06-08 23:51:26.811 | DEBUG    | discord.http:request:673 - POST https://discord.com/api/v10/channels/1072528677620949102/messages has received {'type': 0, 'channel_id': '1072528677620949102', 'content': '', 'attachments': [], 'embeds': [{'type': 'rich', 'description': "Autocropping ['/tmp/tmpx7afqiwm/orig_1249209208058937444_screenshot_image_larger00013.PNG']....", 'content_scan_version': 0}], 'timestamp': '2024-06-09T03:51:26.766000+00:00', 'edited_timestamp': None, 'flags': 0, 'components': [], 'id': '1249209218041643060', 'author': {'id': '1088910543315804170', 'username': 'cerebrotest', 'avatar': '239d07cd2a9146198d455d96010e8e94', 'discriminator': '0265', 'public_flags': 0, 'flags': 0, 'bot': True, 'banner': None, 'accent_color': None, 'global_name': None, 'avatar_decoration_data': None, 'banner_color': None, 'clan': None}, 'mentions': [], 'mention_roles': [], 'pinned': False, 'mention_everyone': False, 'tts': False}
2024-06-08 23:51:26.815 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IHDR' 16 13
2024-06-08 23:51:26.817 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'iCCP' 41 373
2024-06-08 23:51:26.818 | DEBUG    | PIL.PngImagePlugin:chunk_iCCP:405 - iCCP profile name b'kCGColorSpaceDisplayP3'
2024-06-08 23:51:26.818 | DEBUG    | PIL.PngImagePlugin:chunk_iCCP:406 - Compression method 0
2024-06-08 23:51:26.819 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'eXIf' 426 26
2024-06-08 23:51:26.820 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'iTXt' 464 495
2024-06-08 23:51:26.821 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IDAT' 971 16384
2024-06-08 23:51:26.877 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IHDR' 16 13
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'iCCP' 41 373
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:chunk_iCCP:405 - iCCP profile name b'kCGColorSpaceDisplayP3'
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:chunk_iCCP:406 - Compression method 0
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'eXIf' 426 26
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'iTXt' 464 495
2024-06-08 23:51:26.878 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IDAT' 971 16384
  0%|                                                                                                                                                                                         | 0/1 [00:00<?, ?it/s](224, 224, 3)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.35it/s]
tensor([[   5.9265,  252.9969, 1182.9475, 1941.2478]], dtype=torch.float32)
count: 0 - predict threadpool <PIL.Image.Image image mode=RGB size=1179x2556 at 0x7FF1B25369D0> tensor([[   5.9265,  252.9969, 1182.9475, 1941.2478]], dtype=torch.float32)
252 1941 5 1182
count: 0 - autocrop threadpool /tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG
2024-06-08 23:51:31.116 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IHDR' 16 13
2024-06-08 23:51:31.116 | DEBUG    | PIL.PngImagePlugin:call:202 - STREAM b'IDAT' 41 8192
GOT COLOR darkmode -- 23,31,42
count: 0 - Resized threadpool /tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG
deleting ... /tmp/tmpx7afqiwm/orig_1249209208058937444_screenshot_image_larger00013.PNG
count: 0 - Unlink /tmp/tmpx7afqiwm/orig_1249209208058937444_screenshot_image_larger00013.PNG
+ /tmp/tmpx7afqiwm
    + cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG
tree_list ->
[PosixPath('/tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG')]
['/tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG']
2024-06-08 23:51:35.092 | DEBUG    | cerebro_bot.cogs.autocrop:autocrop:868 - AutoCrop -> file_to_upload_list = ['/tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG']
2024-06-08 23:51:35.209 | DEBUG    | discord.gateway:received_message:500 - For Shard ID 0: WebSocket Event: {'t': 'MESSAGE_CREATE', 's': 72, 'op': 0, 'd': {'type': 0, 'tts': False, 'timestamp': '2024-06-09T03:51:35.168000+00:00', 'pinned': False, 'mentions': [], 'mention_roles': [], 'mention_everyone': False, 'member': {'roles': ['1088913950990676022'], 'premium_since': None, 'pending': False, 'nick': None, 'mute': False, 'joined_at': '2023-03-24T19:55:17.054000+00:00', 'flags': 0, 'deaf': False, 'communication_disabled_until': None, 'avatar': None}, 'id': '1249209253282185217', 'flags': 0, 'embeds': [{'type': 'rich', 'description': 'Uploading batch 0....', 'content_scan_version': 0}], 'edited_timestamp': None, 'content': '', 'components': [], 'channel_id': '1072528677620949102', 'author': {'username': 'cerebrotest', 'public_flags': 0, 'id': '1088910543315804170', 'global_name': None, 'discriminator': '0265', 'clan': None, 'bot': True, 'avatar_decoration_data': None, 'avatar': '239d07cd2a9146198d455d96010e8e94'}, 'attachments': [], 'guild_id': '908894726793601094'}}
2024-06-08 23:51:35.210 | DEBUG    | discord.client:dispatch:462 - Dispatching event socket_event_type
2024-06-08 23:51:35.211 | DEBUG    | discord.client:dispatch:462 - Dispatching event message
2024-06-08 23:51:35.238 | DEBUG    | discord.http:request:625 - POST https://discord.com/api/v10/channels/1072528677620949102/messages with {"embeds":[{"type":"rich","description":"Uploading batch 0...."}],"content":null,"components":[],"tts":false} has returned 200
2024-06-08 23:51:35.239 | DEBUG    | discord.http:request:673 - POST https://discord.com/api/v10/channels/1072528677620949102/messages has received {'type': 0, 'channel_id': '1072528677620949102', 'content': '', 'attachments': [], 'embeds': [{'type': 'rich', 'description': 'Uploading batch 0....', 'content_scan_version': 0}], 'timestamp': '2024-06-09T03:51:35.168000+00:00', 'edited_timestamp': None, 'flags': 0, 'components': [], 'id': '1249209253282185217', 'author': {'id': '1088910543315804170', 'username': 'cerebrotest', 'avatar': '239d07cd2a9146198d455d96010e8e94', 'discriminator': '0265', 'public_flags': 0, 'flags': 0, 'bot': True, 'banner': None, 'accent_color': None, 'global_name': None, 'avatar_decoration_data': None, 'banner_color': None, 'clan': None}, 'mentions': [], 'mention_roles': [], 'pinned': False, 'mention_everyone': False, 'tts': False}
/tmp/tmpx7afqiwm/cropped-ObjLocModelV1-orig_1249209208058937444_screenshot_image_larger00013.PNG
[<discord.file.File object at 0x7ff29d03c5e0>]
2024-06-08 23:51:35.242 | DEBUG    | cerebro_bot.cogs.autocrop:autocrop:892 - AutoCrop -> my_files = [<discord.file.File object at 0x7ff29d03c5e0>]

```
