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

# Compare cerebro model to goob-ai

What is model ensembling?
Model ensembling combines the predictions from multiple models together. Traditionally this is done by running each model on some inputs separately and then combining the predictions. However, if you’re running models with the same architecture, then it may be possible to combine them together using torch.vmap. vmap is a function transform that maps functions across dimensions of the input tensors. One of its use cases is eliminating for-loops and speeding them up through vectorization.

Let’s demonstrate how to do this using an ensemble of simple MLPs.

extending a model is called finetuning.

It's actually called transfer learning.

#### cerebro

what is backbone?

```
Generally a backbone refers to some “base model” of the overall architecture, which can be changed for another one.
E.g. some detection models use a resnet as the default backbone, which you could swap for e.g. a VGG model or your custom one.
```

```
>>> rich.inspect(model)
╭─ ObjLocModel(   (backbone): EfficientNet(     (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(─╮
│ def (*input, **kwargs):                                                                               │
│                                                                                                       │
│  dump_patches = False                                                                                 │
│          name = 'ObjLocModelV1'                                                                       │
│ T_destination = ~T_destination                                                                        │
│      training = False                                                                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>> from cerebro_bot import debugger
>>> debugger.rich_inspect(model)
╭─ ObjLocModel(   (backbone): EfficientNet(     (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(─╮
│ def (*input, **kwargs):                                                                               │
│                                                                                                       │
│                       dump_patches = False                                                            │
│                               name = 'ObjLocModelV1'                                                  │
│                      T_destination = ~T_destination                                                   │
│                           training = False                                                            │
│                         add_module = def add_module(name: str, module:                                │
│                                      Optional[ForwardRef('Module')]) -> None: Adds a child module to  │
│                                      the current module.                                              │
│                              apply = def apply(fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T: │
│                                      Applies ``fn`` recursively to every submodule (as returned by    │
│                                      ``.children()``)                                                 │
│                                      as well as self. Typical use includes initializing the           │
│                                      parameters of a model                                            │
│                                      (see also :ref:`nn-init-doc`).                                   │
│                           backbone = def backbone(*input, **kwargs): EfficientNet                     │
│                           bfloat16 = def bfloat16() -> ~T: Casts all floating point parameters and    │
│                                      buffers to ``bfloat16`` datatype.                                │
│                            buffers = def buffers(recurse: bool = True) -> Iterator[torch.Tensor]:     │
│                                      Returns an iterator over module buffers.                         │
│                           children = def children() -> Iterator[ForwardRef('Module')]: Returns an     │
│                                      iterator over immediate children modules.                        │
│                                cpu = def cpu() -> ~T: Moves all model parameters and buffers to the   │
│                                      CPU.                                                             │
│                               cuda = def cuda(device: Union[int, torch.device, NoneType] = None) ->   │
│                                      ~T: Moves all model parameters and buffers to the GPU.           │
│                             double = def double() -> ~T: Casts all floating point parameters and      │
│                                      buffers to ``double`` datatype.                                  │
│                               eval = def eval() -> ~T: Sets the module in evaluation mode.            │
│                         extra_repr = def extra_repr() -> str: Set the extra representation of the     │
│                                      module                                                           │
│                              float = def float() -> ~T: Casts all floating point parameters and       │
│                                      buffers to ``float`` datatype.                                   │
│                            forward = def forward(images, gt_bboxes=None): Defines the computation     │
│                                      performed at every call.                                         │
│                         get_buffer = def get_buffer(target: str) -> 'Tensor':                         │
│                                      Returns the buffer given by ``target`` if it exists,             │
│                                      otherwise throws an error.                                       │
│                    get_extra_state = def get_extra_state() -> Any:                                    │
│                                      Returns any extra state to include in the module's state_dict.   │
│                                      Implement this and a corresponding :func:`set_extra_state` for   │
│                                      your module                                                      │
│                                      if you need to store extra state. This function is called when   │
│                                      building the                                                     │
│                                      module's `state_dict()`.                                         │
│                      get_parameter = def get_parameter(target: str) -> 'Parameter':                   │
│                                      Returns the parameter given by ``target`` if it exists,          │
│                                      otherwise throws an error.                                       │
│                      get_submodule = def get_submodule(target: str) -> 'Module':                      │
│                                      Returns the submodule given by ``target`` if it exists,          │
│                                      otherwise throws an error.                                       │
│                               half = def half() -> ~T: Casts all floating point parameters and        │
│                                      buffers to ``half`` datatype.                                    │
│                                ipu = def ipu(device: Union[int, torch.device, NoneType] = None) ->    │
│                                      ~T: Moves all model parameters and buffers to the IPU.           │
│                    load_state_dict = def load_state_dict(state_dict: Mapping[str, Any], strict: bool  │
│                                      = True):                                                         │
│                                      Copies parameters and buffers from :attr:`state_dict` into       │
│                                      this module and its descendants. If :attr:`strict` is ``True``,  │
│                                      then                                                             │
│                                      the keys of :attr:`state_dict` must exactly match the keys       │
│                                      returned                                                         │
│                                      by this module's :meth:`~torch.nn.Module.state_dict` function.   │
│                            modules = def modules() -> Iterator[ForwardRef('Module')]: Returns an      │
│                                      iterator over all modules in the network.                        │
│                      named_buffers = def named_buffers(prefix: str = '', recurse: bool = True) ->     │
│                                      Iterator[Tuple[str, torch.Tensor]]:                              │
│                                      Returns an iterator over module buffers, yielding both the       │
│                                      name of the buffer as well as the buffer itself.                 │
│                     named_children = def named_children() -> Iterator[Tuple[str,                      │
│                                      ForwardRef('Module')]]:                                          │
│                                      Returns an iterator over immediate children modules, yielding    │
│                                      both                                                             │
│                                      the name of the module as well as the module itself.             │
│                      named_modules = def named_modules(memo: Optional[Set[ForwardRef('Module')]] =    │
│                                      None, prefix: str = '', remove_duplicate: bool = True):          │
│                                      Returns an iterator over all modules in the network, yielding    │
│                                      both the name of the module as well as the module itself.        │
│                   named_parameters = def named_parameters(prefix: str = '', recurse: bool = True) ->  │
│                                      Iterator[Tuple[str, torch.nn.parameter.Parameter]]:              │
│                                      Returns an iterator over module parameters, yielding both the    │
│                                      name of the parameter as well as the parameter itself.           │
│                         parameters = def parameters(recurse: bool = True) ->                          │
│                                      Iterator[torch.nn.parameter.Parameter]: Returns an iterator over │
│                                      module parameters.                                               │
│             register_backward_hook = def register_backward_hook(hook: Callable[[ForwardRef('Module'), │
│                                      Union[Tuple[torch.Tensor, ...], torch.Tensor],                   │
│                                      Union[Tuple[torch.Tensor, ...], torch.Tensor]],                  │
│                                      Optional[torch.Tensor]]) -> torch.utils.hooks.RemovableHandle:   │
│                                      Registers a backward hook on the module.                         │
│                    register_buffer = def register_buffer(name: str, tensor: Optional[torch.Tensor],   │
│                                      persistent: bool = True) -> None: Adds a buffer to the module.   │
│              register_forward_hook = def register_forward_hook(hook: Callable[..., NoneType]) ->      │
│                                      torch.utils.hooks.RemovableHandle: Registers a forward hook on   │
│                                      the module.                                                      │
│          register_forward_pre_hook = def register_forward_pre_hook(hook: Callable[..., NoneType]) ->  │
│                                      torch.utils.hooks.RemovableHandle: Registers a forward pre-hook  │
│                                      on the module.                                                   │
│        register_full_backward_hook = def register_full_backward_hook(hook:                            │
│                                      Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...],  │
│                                      torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]],   │
│                                      Optional[torch.Tensor]]) -> torch.utils.hooks.RemovableHandle:   │
│                                      Registers a backward hook on the module.                         │
│ register_load_state_dict_post_hook = def register_load_state_dict_post_hook(hook):                    │
│                                      Registers a post hook to be run after module's                   │
│                                      ``load_state_dict``                                              │
│                                      is called.                                                       │
│                    register_module = def register_module(name: str, module:                           │
│                                      Optional[ForwardRef('Module')]) -> None: Alias for               │
│                                      :func:`add_module`.                                              │
│                 register_parameter = def register_parameter(name: str, param:                         │
│                                      Optional[torch.nn.parameter.Parameter]) -> None: Adds a          │
│                                      parameter to the module.                                         │
│                     requires_grad_ = def requires_grad_(requires_grad: bool = True) -> ~T:            │
│                                      Change if autograd should record operations on parameters in     │
│                                      this                                                             │
│                                      module.                                                          │
│                    set_extra_state = def set_extra_state(state: Any):                                 │
│                                      This function is called from :func:`load_state_dict` to handle   │
│                                      any extra state                                                  │
│                                      found within the `state_dict`. Implement this function and a     │
│                                      corresponding                                                    │
│                                      :func:`get_extra_state` for your module if you need to store     │
│                                      extra state within its                                           │
│                                      `state_dict`.                                                    │
│                       share_memory = def share_memory() -> ~T: See :meth:`torch.Tensor.share_memory_` │
│                         state_dict = def state_dict(*args, destination=None, prefix='',               │
│                                      keep_vars=False): Returns a dictionary containing references to  │
│                                      the whole state of the module.                                   │
│                                 to = def to(*args, **kwargs): Moves and/or casts the parameters and   │
│                                      buffers.                                                         │
│                           to_empty = def to_empty(*, device: Union[str, torch.device]) -> ~T: Moves   │
│                                      the parameters and buffers to the specified device without       │
│                                      copying storage.                                                 │
│                              train = def train(mode: bool = True) -> ~T: Sets the module in training  │
│                                      mode.                                                            │
│                               type = def type(dst_type: Union[torch.dtype, str]) -> ~T: Casts all     │
│                                      parameters and buffers to :attr:`dst_type`.                      │
│                                xpu = def xpu(device: Union[int, torch.device, NoneType] = None) ->    │
│                                      ~T: Moves all model parameters and buffers to the XPU.           │
│                          zero_grad = def zero_grad(set_to_none: bool = False) -> None:                │
│                                      Sets gradients of all model parameters to zero. See similar      │
│                                      function                                                         │
│                                      under :class:`torch.optim.Optimizer` for more context.           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>>
```

dump_dict

```
{   '_backward_hooks': OrderedDict(),
    '_buffers': OrderedDict(),
    '_forward_hooks': OrderedDict(),
    '_forward_pre_hooks': OrderedDict(),
    '_is_full_backward_hook': None,
    '_load_state_dict_post_hooks': OrderedDict(),
    '_load_state_dict_pre_hooks': OrderedDict(),
    '_modules': OrderedDict([   (   'backbone',
                                    EfficientNet(
  (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn1): BatchNormAct2d(
    32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    (drop): Identity()
    (act): SiLU(inplace=True)
  )
  (blocks): Sequential(
    (0): Sequential(
      (0): DepthwiseSeparableConv(
        (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=Fals
e)
        (bn1): BatchNormAct2d(
          32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn2): BatchNormAct2d(
          16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (1): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=Fals
e)
        (bn2): BatchNormAct2d(
          96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=F
alse)
        (bn2): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (2): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=F
alse)
        (bn2): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=F
alse)
        (bn2): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (3): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=F
alse)
        (bn2): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (4): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (5): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (3): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (6): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
  )
  (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn2): BatchNormAct2d(
    1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    (drop): Identity()
    (act): SiLU(inplace=True)
  )
  (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
  (classifier): Linear(in_features=1280, out_features=4, bias=True)
))]),
    '_non_persistent_buffers_set': set(),
    '_parameters': OrderedDict(),
    '_state_dict_hooks': OrderedDict(),
    'name': 'ObjLocModelV1',
    'training': False}
{'training': False, '_parameters': OrderedDict(), '_buffers': OrderedDict(), '_non_persistent_buffers_set
': set(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict
(), '_forward_pre_hooks': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_load_state_dict_pre_hooks'
: OrderedDict(), '_load_state_dict_post_hooks': OrderedDict(), '_modules': OrderedDict([('backbone', Effi
cientNet(
  (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn1): BatchNormAct2d(
    32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    (drop): Identity()
    (act): SiLU(inplace=True)
  )
  (blocks): Sequential(
    (0): Sequential(
      (0): DepthwiseSeparableConv(
        (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=Fals
e)
        (bn1): BatchNormAct2d(
          32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn2): BatchNormAct2d(
          16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (1): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=Fals
e)
        (bn2): BatchNormAct2d(
          96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=F
alse)
        (bn2): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (2): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=F
alse)
        (bn2): BatchNormAct2d(
          144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=F
alse)
        (bn2): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (3): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=F
alse)
        (bn2): BatchNormAct2d(
          240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (4): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=F
alse)
        (bn2): BatchNormAct2d(
          480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (5): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=F
alse)
        (bn2): BatchNormAct2d(
          672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
      (3): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
    (6): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bia
s=False)
        (bn2): BatchNormAct2d(
          1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): SiLU(inplace=True)
        )
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNormAct2d(
          320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          (drop): Identity()
          (act): Identity()
        )
        (drop_path): Identity()
      )
    )
  )
  (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn2): BatchNormAct2d(
    1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    (drop): Identity()
    (act): SiLU(inplace=True)
  )
  (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
  (classifier): Linear(in_features=1280, out_features=4, bias=True)
))]), 'name': 'ObjLocModelV1'}
>
```

backbone

```
>>> model.backbone.num_classes
4
>>> model.backbone.num_features
1280
>>>



>>> debugger.rich_inspect(model.backbone)
╭─ EfficientNet(   (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   (bn─╮
│ def (*input, **kwargs):                                                                                          │
│                                                                                                                  │
│ EfficientNet                                                                                                     │
│                                                                                                                  │
│                        default_cfg = {                                                                           │
│                                          'url':                                                                  │
│                                      'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-… │
│                                          'num_classes': 1000,                                                    │
│                                          'input_size': (3, 224, 224),                                            │
│                                          'pool_size': (7, 7),                                                    │
│                                          'crop_pct': 0.875,                                                      │
│                                          'interpolation': 'bicubic',                                             │
│                                          'mean': (0.485, 0.456, 0.406),                                          │
│                                          'std': (0.229, 0.224, 0.225),                                           │
│                                          'first_conv': 'conv_stem',                                              │
│                                          'classifier': 'classifier',                                             │
│                                          'architecture': 'efficientnet_b0'                                       │
│                                      }                                                                           │
│                          drop_rate = 0.0                                                                         │
│                       dump_patches = False                                                                       │
│                       feature_info = [                                                                           │
│                                          {                                                                       │
│                                              'stage': 1,                                                         │
│                                              'reduction': 2,                                                     │
│                                              'module': 'blocks.0.0',                                             │
│                                              'hook_type': '',                                                    │
│                                              'num_chs': 16                                                       │
│                                          },                                                                      │
│                                          {                                                                       │
│                                              'stage': 2,                                                         │
│                                              'reduction': 4,                                                     │
│                                              'module': 'blocks.1.1',                                             │
│                                              'hook_type': '',                                                    │
│                                              'num_chs': 24                                                       │
│                                          },                                                                      │
│                                          {                                                                       │
│                                              'stage': 3,                                                         │
│                                              'reduction': 8,                                                     │
│                                              'module': 'blocks.2.1',                                             │
│                                              'hook_type': '',                                                    │
│                                              'num_chs': 40                                                       │
│                                          },                                                                      │
│                                          {                                                                       │
│                                              'stage': 5,                                                         │
│                                              'reduction': 16,                                                    │
│                                              'module': 'blocks.4.2',                                             │
│                                              'hook_type': '',                                                    │
│                                              'num_chs': 112                                                      │
│                                          },                                                                      │
│                                          {                                                                       │
│                                              'stage': 7,                                                         │
│                                              'reduction': 32,                                                    │
│                                              'module': 'blocks.6.0',                                             │
│                                              'hook_type': '',                                                    │
│                                              'num_chs': 320                                                      │
│                                          }                                                                       │
│                                      ]                                                                           │
│                 grad_checkpointing = False                                                                       │
│                        num_classes = 4                                                                           │
│                       num_features = 1280                                                                        │
│                     pretrained_cfg = {                                                                           │
│                                          'url':                                                                  │
│                                      'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-… │
│                                          'num_classes': 1000,                                                    │
│                                          'input_size': (3, 224, 224),                                            │
│                                          'pool_size': (7, 7),                                                    │
│                                          'crop_pct': 0.875,                                                      │
│                                          'interpolation': 'bicubic',                                             │
│                                          'mean': (0.485, 0.456, 0.406),                                          │
│                                          'std': (0.229, 0.224, 0.225),                                           │
│                                          'first_conv': 'conv_stem',                                              │
│                                          'classifier': 'classifier',                                             │
│                                          'architecture': 'efficientnet_b0'                                       │
│                                      }                                                                           │
│                      T_destination = ~T_destination                                                              │
│                           training = False                                                                       │
│                         add_module = def add_module(name: str, module: Optional[ForwardRef('Module')]) -> None:  │
│                                      Adds a child module to the current module.                                  │
│                              apply = def apply(fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T:            │
│                                      Applies ``fn`` recursively to every submodule (as returned by               │
│                                      ``.children()``)                                                            │
│                                      as well as self. Typical use includes initializing the parameters of a      │
│                                      model                                                                       │
│                                      (see also :ref:`nn-init-doc`).                                              │
│                      as_sequential = def as_sequential():                                                        │
│                           bfloat16 = def bfloat16() -> ~T: Casts all floating point parameters and buffers to    │
│                                      ``bfloat16`` datatype.                                                      │
│                             blocks = def blocks(*input, **kwargs):                                               │
│                                      A sequential container.                                                     │
│                                      Modules will be added to it in the order they are passed in the             │
│                                      constructor. Alternatively, an ``OrderedDict`` of modules can be            │
│                                      passed in. The ``forward()`` method of ``Sequential`` accepts any           │
│                                      input and forwards it to the first module it contains. It then              │
│                                      "chains" outputs to inputs sequentially for each subsequent module,         │
│                                      finally returning the output of the last module.                            │
│                                bn1 = def bn1(*input, **kwargs): BatchNorm + Activation                           │
│                                bn2 = def bn2(*input, **kwargs): BatchNorm + Activation                           │
│                            buffers = def buffers(recurse: bool = True) -> Iterator[torch.Tensor]: Returns an     │
│                                      iterator over module buffers.                                               │
│                           children = def children() -> Iterator[ForwardRef('Module')]: Returns an iterator over  │
│                                      immediate children modules.                                                 │
│                         classifier = def classifier(*input, **kwargs): Applies a linear transformation to the    │
│                                      incoming data: :math:`y = xA^T + b`                                         │
│                          conv_head = def conv_head(*input, **kwargs):                                            │
│                                      Applies a 2D convolution over an input signal composed of several input     │
│                                      planes.                                                                     │
│                          conv_stem = def conv_stem(*input, **kwargs):                                            │
│                                      Applies a 2D convolution over an input signal composed of several input     │
│                                      planes.                                                                     │
│                                cpu = def cpu() -> ~T: Moves all model parameters and buffers to the CPU.         │
│                               cuda = def cuda(device: Union[int, torch.device, NoneType] = None) -> ~T: Moves    │
│                                      all model parameters and buffers to the GPU.                                │
│                             double = def double() -> ~T: Casts all floating point parameters and buffers to      │
│                                      ``double`` datatype.                                                        │
│                               eval = def eval() -> ~T: Sets the module in evaluation mode.                       │
│                         extra_repr = def extra_repr() -> str: Set the extra representation of the module         │
│                              float = def float() -> ~T: Casts all floating point parameters and buffers to       │
│                                      ``float`` datatype.                                                         │
│                            forward = def forward(x): Defines the computation performed at every call.            │
│                   forward_features = def forward_features(x):                                                    │
│                       forward_head = def forward_head(x, pre_logits: bool = False):                              │
│                         get_buffer = def get_buffer(target: str) -> 'Tensor':                                    │
│                                      Returns the buffer given by ``target`` if it exists,                        │
│                                      otherwise throws an error.                                                  │
│                     get_classifier = def get_classifier():                                                       │
│                    get_extra_state = def get_extra_state() -> Any:                                               │
│                                      Returns any extra state to include in the module's state_dict.              │
│                                      Implement this and a corresponding :func:`set_extra_state` for your module  │
│                                      if you need to store extra state. This function is called when building the │
│                                      module's `state_dict()`.                                                    │
│                      get_parameter = def get_parameter(target: str) -> 'Parameter':                              │
│                                      Returns the parameter given by ``target`` if it exists,                     │
│                                      otherwise throws an error.                                                  │
│                      get_submodule = def get_submodule(target: str) -> 'Module':                                 │
│                                      Returns the submodule given by ``target`` if it exists,                     │
│                                      otherwise throws an error.                                                  │
│                        global_pool = def global_pool(*input, **kwargs): Selectable global pooling layer with     │
│                                      dynamic input kernel size                                                   │
│                      group_matcher = def group_matcher(coarse=False):                                            │
│                               half = def half() -> ~T: Casts all floating point parameters and buffers to        │
│                                      ``half`` datatype.                                                          │
│                                ipu = def ipu(device: Union[int, torch.device, NoneType] = None) -> ~T: Moves all │
│                                      model parameters and buffers to the IPU.                                    │
│                    load_state_dict = def load_state_dict(state_dict: Mapping[str, Any], strict: bool = True):    │
│                                      Copies parameters and buffers from :attr:`state_dict` into                  │
│                                      this module and its descendants. If :attr:`strict` is ``True``, then        │
│                                      the keys of :attr:`state_dict` must exactly match the keys returned         │
│                                      by this module's :meth:`~torch.nn.Module.state_dict` function.              │
│                            modules = def modules() -> Iterator[ForwardRef('Module')]: Returns an iterator over   │
│                                      all modules in the network.                                                 │
│                      named_buffers = def named_buffers(prefix: str = '', recurse: bool = True) ->                │
│                                      Iterator[Tuple[str, torch.Tensor]]:                                         │
│                                      Returns an iterator over module buffers, yielding both the                  │
│                                      name of the buffer as well as the buffer itself.                            │
│                     named_children = def named_children() -> Iterator[Tuple[str, ForwardRef('Module')]]:         │
│                                      Returns an iterator over immediate children modules, yielding both          │
│                                      the name of the module as well as the module itself.                        │
│                      named_modules = def named_modules(memo: Optional[Set[ForwardRef('Module')]] = None, prefix: │
│                                      str = '', remove_duplicate: bool = True):                                   │
│                                      Returns an iterator over all modules in the network, yielding               │
│                                      both the name of the module as well as the module itself.                   │
│                   named_parameters = def named_parameters(prefix: str = '', recurse: bool = True) ->             │
│                                      Iterator[Tuple[str, torch.nn.parameter.Parameter]]:                         │
│                                      Returns an iterator over module parameters, yielding both the               │
│                                      name of the parameter as well as the parameter itself.                      │
│                         parameters = def parameters(recurse: bool = True) ->                                     │
│                                      Iterator[torch.nn.parameter.Parameter]: Returns an iterator over module     │
│                                      parameters.                                                                 │
│             register_backward_hook = def register_backward_hook(hook: Callable[[ForwardRef('Module'),            │
│                                      Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor,    │
│                                      ...], torch.Tensor]], Optional[torch.Tensor]]) ->                           │
│                                      torch.utils.hooks.RemovableHandle: Registers a backward hook on the module. │
│                    register_buffer = def register_buffer(name: str, tensor: Optional[torch.Tensor], persistent:  │
│                                      bool = True) -> None: Adds a buffer to the module.                          │
│              register_forward_hook = def register_forward_hook(hook: Callable[..., NoneType]) ->                 │
│                                      torch.utils.hooks.RemovableHandle: Registers a forward hook on the module.  │
│          register_forward_pre_hook = def register_forward_pre_hook(hook: Callable[..., NoneType]) ->             │
│                                      torch.utils.hooks.RemovableHandle: Registers a forward pre-hook on the      │
│                                      module.                                                                     │
│        register_full_backward_hook = def register_full_backward_hook(hook: Callable[[ForwardRef('Module'),       │
│                                      Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor,    │
│                                      ...], torch.Tensor]], Optional[torch.Tensor]]) ->                           │
│                                      torch.utils.hooks.RemovableHandle: Registers a backward hook on the module. │
│ register_load_state_dict_post_hook = def register_load_state_dict_post_hook(hook):                               │
│                                      Registers a post hook to be run after module's ``load_state_dict``          │
│                                      is called.                                                                  │
│                    register_module = def register_module(name: str, module: Optional[ForwardRef('Module')]) ->   │
│                                      None: Alias for :func:`add_module`.                                         │
│                 register_parameter = def register_parameter(name: str, param:                                    │
│                                      Optional[torch.nn.parameter.Parameter]) -> None: Adds a parameter to the    │
│                                      module.                                                                     │
│                     requires_grad_ = def requires_grad_(requires_grad: bool = True) -> ~T:                       │
│                                      Change if autograd should record operations on parameters in this           │
│                                      module.                                                                     │
│                   reset_classifier = def reset_classifier(num_classes, global_pool='avg'):                       │
│                    set_extra_state = def set_extra_state(state: Any):                                            │
│                                      This function is called from :func:`load_state_dict` to handle any extra    │
│                                      state                                                                       │
│                                      found within the `state_dict`. Implement this function and a corresponding  │
│                                      :func:`get_extra_state` for your module if you need to store extra state    │
│                                      within its                                                                  │
│                                      `state_dict`.                                                               │
│             set_grad_checkpointing = def set_grad_checkpointing(enable=True):                                    │
│                       share_memory = def share_memory() -> ~T: See :meth:`torch.Tensor.share_memory_`            │
│                         state_dict = def state_dict(*args, destination=None, prefix='', keep_vars=False):        │
│                                      Returns a dictionary containing references to the whole state of the        │
│                                      module.                                                                     │
│                                 to = def to(*args, **kwargs): Moves and/or casts the parameters and buffers.     │
│                           to_empty = def to_empty(*, device: Union[str, torch.device]) -> ~T: Moves the          │
│                                      parameters and buffers to the specified device without copying storage.     │
│                              train = def train(mode: bool = True) -> ~T: Sets the module in training mode.       │
│                               type = def type(dst_type: Union[torch.dtype, str]) -> ~T: Casts all parameters and │
│                                      buffers to :attr:`dst_type`.                                                │
│                                xpu = def xpu(device: Union[int, torch.device, NoneType] = None) -> ~T: Moves all │
│                                      model parameters and buffers to the XPU.                                    │
│                          zero_grad = def zero_grad(set_to_none: bool = False) -> None:                           │
│                                      Sets gradients of all model parameters to zero. See similar function        │
│                                      under :class:`torch.optim.Optimizer` for more context.                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>>


>>> model.backbone.default_cfg
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth', 'num_classes': 1000, 'input_siz
e': (3, 224, 224), 'pool_size': (7, 7), 'crop_pct': 0.875, 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'first_
conv': 'conv_stem', 'classifier': 'classifier', 'architecture': 'efficientnet_b0'}
>>>

>>> rich.print(model.backbone.default_cfg)
{
    'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
    'num_classes': 1000,
    'input_size': (3, 224, 224),
    'pool_size': (7, 7),
    'crop_pct': 0.875,
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'first_conv': 'conv_stem',
    'classifier': 'classifier',
    'architecture': 'efficientnet_b0'
}
>>>


>>> model.backbone.get_classifier()
Linear(in_features=1280, out_features=4, bias=True)
>>>


>>> model.backbone.default_cfg
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth', 'num_classes': 1000, 'input_siz
e': (3, 224, 224), 'pool_size': (7, 7), 'crop_pct': 0.875, 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'first_
conv': 'conv_stem', 'classifier': 'classifier', 'architecture': 'efficientnet_b0'}


>>> rich.print(model.backbone.default_cfg)
{
    'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
    'num_classes': 1000,
    'input_size': (3, 224, 224),
    'pool_size': (7, 7),
    'crop_pct': 0.875,
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'first_conv': 'conv_stem',
    'classifier': 'classifier',
    'architecture': 'efficientnet_b0'
}
>>>

```

#### dump dir

```
[   'T_destination',
    '__annotations__',
    '__call__',
    '__class__',
    '__delattr__',
    '__dict__',
    '__dir__',
    '__doc__',
    '__eq__',
    '__format__',
    '__ge__',
    '__getattr__',
    '__getattribute__',
    '__gt__',
    '__hash__',
    '__init__',
    '__init_subclass__',
    '__le__',
    '__lt__',
    '__module__',
    '__ne__',
    '__new__',
    '__reduce__',
    '__reduce_ex__',
    '__repr__',
    '__setattr__',
    '__setstate__',
    '__sizeof__',
    '__str__',
    '__subclasshook__',
    '__weakref__',
    '_apply',
    '_backward_hooks',
    '_buffers',
    '_call_impl',
    '_forward_hooks',
    '_forward_pre_hooks',
    '_get_backward_hooks',
    '_get_name',
    '_is_full_backward_hook',
    '_load_from_state_dict',
    '_load_state_dict_post_hooks',
    '_load_state_dict_pre_hooks',
    '_maybe_warn_non_full_backward_hook',
    '_modules',
    '_named_members',
    '_non_persistent_buffers_set',
    '_parameters',
    '_register_load_state_dict_pre_hook',
    '_register_state_dict_hook',
    '_replicate_for_data_parallel',
    '_save_to_state_dict',
    '_slow_forward',
    '_state_dict_hooks',
    '_version',
    'add_module',
    'apply',
    'backbone',
    'bfloat16',
    'buffers',
    'children',
    'cpu',
    'cuda',
    'double',
    'dump_patches',
    'eval',
    'extra_repr',
    'float',
    'forward',
    'get_buffer',
    'get_extra_state',
    'get_parameter',
    'get_submodule',
    'half',
    'ipu',
    'load_state_dict',
    'modules',
    'name',
    'named_buffers',
    'named_children',
    'named_modules',
    'named_parameters',
    'parameters',
    'register_backward_hook',
    'register_buffer',
    'register_forward_hook',
    'register_forward_pre_hook',
    'register_full_backward_hook',
    'register_load_state_dict_post_hook',
    'register_module',
    'register_parameter',
    'requires_grad_',
    'set_extra_state',
    'share_memory',
    'state_dict',
    'to',
    'to_empty',
    'train',
    'training',
    'type',
    'xpu',
    'zero_grad']
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr
__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__red
uce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_backward_hooks', '_buffers',
'_call_impl', '_forward_hooks', '_forward_pre_hooks', '_get_backward_hooks', '_get_name', '_is_full_backward_hook', '_load_from_state_dict', '_load_state_d
ict_post_hooks', '_load_state_dict_pre_hooks', '_maybe_warn_non_full_backward_hook', '_modules', '_named_members', '_non_persistent_buffers_set', '_paramet
ers', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_state_di
ct_hooks', '_version', 'add_module', 'apply', 'backbone', 'bfloat16', 'buffers', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr',
 'float', 'forward', 'get_buffer', 'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu', 'load_state_dict', 'modules', 'name', 'named_buffers
', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_for
ward_pre_hook', 'register_full_backward_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'requires_grad_', 'set_extra_
state', 'share_memory', 'state_dict', 'to', 'to_empty', 'train', 'training', 'type', 'xpu', 'zero_grad']
>>>
```

# Chroma + docker-compose

run the following:

```
git clone --depth 1 --branch feature-boss-chroma https://github.com/bossjones/chroma.git vendored_chroma
cd vendored_chroma
docker-compose up -d --build
docker ps -a
```

enter: `http://host.docker.internal:8010` for your connection string in browser.

goto <http://localhost:3000/collections/test_chroma_service_e2e_add_to_chroma_url>

# Langsmith

- custom tool(chroma_question_answering): <https://smith.langchain.com/o/631f824f-4072-5bc6-b1f6-924eb5dfd83f/projects/p/9c97a8d8-3b8b-4b42-85be-6a17c4aab36d?timeModel=%7B%22duration%22%3A%227d%22%7D&runtab=0&peek=7c25a24d-9240-4736-a109-02013fa8aa90>
