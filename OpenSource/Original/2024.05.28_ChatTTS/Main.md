# ChatTTS

代码结构树
```
└── ChatTTS
    ├── asset
    │   ├── Decoder.pt: 98 M
    │   ├── DVAE.pt: 26 M
    │   ├── GPT.pt: 859 M
    │   ├── tokenizer.pt: 328 K
    │   └── Vocos.pt: 51 M
    ├── config
    │   ├── decoder.yaml
    │   ├── dvae.yaml
    │   ├── gpt.yaml
    │   ├── path.yaml
    │   └── vocos.yaml
    ├── experimental
    │   └── llm.py
    ├── infer
    │   └── api.py
    ├── model
    │   ├── dvae.py
    │   └── gpt.py
    ├── utils
    │   ├── gpu_utils.py
    │   └── infer_utils.py
    ├── __init__.py
    ├── core.py
    └── README.md
```

下面从 `infer.ipynb` 文件开始分析:

```
import ChatTTS

chat = ChatTTS.Chat()
```

从 ChatTTS 包中导入 `Chat` 类 (`__init__.py` 中从 `.core` 文件中导入 `Chat` 类)

`core.py` 文件中定义了:
- `Chat` 类:
  - `self.load_models(source)`: 从 huggingface 下载模型, 并调用 `self._load()`
    - 读取 `config/path.yaml` 文件, 配置文件路径和权重文件路径输入到 `self._load()` 中.
  - `self._load(...)` 加载预训练模型.
    - 设置运行设备 `device`
    - 若 `vocos_config_path` 配置文件存在, 则使用 **Vocos 包** 创建 vocos 模型
      - 从 `vocos_ckpt_path` 加载权重
      - 将模型 `vocos` 存入预训练模型字典中 `self.pretrained_models['vocos']`
    - 若 `dvae_config_path` 配置文件存在, 则使用 OmegaConf 加载配置, 用 `DVAE` 类创建 dvae 模型
      - 从 `dvae_ckpt_path` 加载权重
      - 将模型 `dvae` 存入预训练模型字典中 `self.pretrained_models['dvae']`
    - 若 `gpt_config_path` 配置文件存在, 则使用 OmegaConf 加载配置, 用 [`GPT_warpper` 类](#GPT_wrapper)创建 gpt 模型
      - 从 `gpt_ckpt_path` 加载权重
      - 将模型 `gpt` 存入预训练模型字典中 `self.pretrained_models['gpt']`
    - 若 `decoder_config_path` 配置文件存在, 则使用 OmegaConf 加载配置, 用 `DVAE` 类创建 decoder 模型
      - 从 `decoder_ckpt_path` 加载权重
      - 将模型 `decoder` 存入预训练模型字典中 `self.pretrained_models['decoder']`
    - 若 `tokenizer_path` 存在, 则用 `torch.load()` 加载 tokenizer
      - 设置 padding_side = left
      - 将 `tokenizer` 存入预训练模型字典中 `self.pretrained_models['tokenizer']`
    - 调用 `self.check_model()` 检查预训练模型是否初始化
  - `self.check_model(use_decoder=False)`: 检查预训练模型是否顺利初始化.
    - 注意默认检查列表为 `vocos, gpt, tokenizer`, 当 `use_decoder=True` 时, 增加一个 `decoder` 否则添加 `dvae`.
  - `self.infer(text, skip_refine_text=False, params_refine_text={}, params_infer_code={}, use_decoder=False)`
    - 若 `skip_refine_text=False`: 
      - 调用 `refine_text()` 取 ids 得到 text_tokens, 然后过滤掉小于 `[break_0]` 对应 ids 的部分 #TODO ?
      - 调用 `tokenizer` 的 `batch_decode` 转换回文本.
    - 调用 `infer_code()` 得到 result
    - 若 `use_decoder=True`: 使用 `decoder` 解码得到梅尔频谱, 否则用 `dvae`.
    - 调用 `vocos` 解码梅尔频谱得到音频.

上述代码需要另外调用的有:
- `vocos` -> Vocos
- `model/dvae.py` -> DVAE
- `model/gpt.py` -> GPT_wrapper
- `utils/gpu_utils.py` -> select_device
- `infer/api.py` -> refine_text, infer_code

---

下面从 `infer/api.py` 文件开始分析:
- `refine_text(models, text)`:
  - 获取 gpt 模型的运行设备.
  - 将 text 包装为 list 类型
  - 遍历 text, 重写为固定格式 `[Sbreak]{text[i]}[Pbreak]{prompt}`
  - 调用 `tokenizer` 生成 text_token
  - text_mask 默认全为 1.
  - 综上构造输入为 {`input_ids` `text_mask` `attention_mask`}
  - 若 repetition_penalty 不为 None, 且不为 1, 则添加自定义的重复惩罚处理
  - 调用 `gpt` 模型的 `generate()` 函数
- `infer_code(models, text,)`
  - 获取 gpt 模型的运行设备.
  - 将 text 包装为 list 类型
  - 遍历 text, 重写为固定格式 `[Stts][spk_emb]text{i}[uv_break][Ptts]`
  - 调用 `tokenizer` 生成 text_token
  - text_mask 默认全为 1.
  - 综上构造输入为 {`input_ids` `text_mask` `attention_mask`}
  - emb = gpt 模型 get_emb(**input)
  - 如果 spk_emb 存在, 则修改 emb 的部分
  - num_code = gpt 模型的 emb_code[0].num_embeddings-1
  - 调用 gpt 的 generate() 函数, 得到结果

核心步骤在 `GPT_wrapper` 的 `generate()` 函数.

---

下面分析 `model/dvae.py`:

- `DVAE` 类:
  - `self.__init__(decoder_config, vq_config, dim=512)`
    - 调用 `register_buffer` 创建 `self.coef` 为随机的 (1,100,1).
    - 调用 `DVAEDecoder(**decoder_config)` 创建 `self.decoder`;
    - 调用 `nn.Conv1d()` 创建 `self.out_conv`
      - 输入通道: dim
      - 输出通道: 100
      - 卷积核大小: 3
      - 步长: 1
      - 填充: 1
      - 偏置: False
    - 若 `vq_config` 不为空, 则调用 `GFSQ(**vq_config)` 创建 `self.vq_layer, 否则为 None
  - `self.forward(inp)`
    - 若 `self.vq_layer` 存在, 则调用 `self.vq_layer._embed(inp)` 得到 vq 特征, 否则直接取 `inp`
    - 调用 `torch.chunk()` 对 vq 特征进行分块 #TODO ? 然后进行堆叠, 再修改形状为 ? 再转置.
    - 调用 `self.decoder()` 对 vq 特征进行解码得到 dec_out
    - 调用 `self.out_conv()` 对 dec_out 进行卷积得到 dec_out
    - mel = dec_out * `self.coef`
    - 返回 mel
- `DVAEDecoder` 类:
  - `DVAE` 类调用了此解码器
  - `self.__init__()`:
    - i_dim
    - o_dim
    - n_layer: 默认 12 层
    - bn_dim: 默认 64
    - hidden: 默认 256
    - kernel: 默认 7
    - dilation: 默认 2
    - up: False
    - 创建 `self.conv_in`: 
      - `Conv1d(输入通道=i_dim, 输出通道=bn_dim, 卷积核大小=3, 步长=1, 填充=1)`
      - `ELU()`
      - `Conv1d(输入通道=bn_dim, 输出通道=hidden, 卷积核大小=3, 步长=1, 填充=1)`
    - 创建 `self.decoder_block`:
      - n_layer 个 [`ConvNeXtBlock(hidden, hidden*4, kernel, dilation)`](#ConvNeXtBlock)
    - 创建 `self.conv_out`: 
      - `Conv1d(hidden, o_dim, 1, bias=False)`
  - `self.forward(x, conditioning)`:
    - 输入的 x 先转置 (1,2)
    - 调用 `self.conv_in(x)` -> 遍历 `self.decoder_block(x)`, 输入 conditioning -> 调用 `self.conv_out(x)` -> 转置 (1,2) -> 结果
- `GFSQ` 类:
  - `DVAE` 类调用了此类
  - `self.__init__()`
    - dim
    - levels
    - groups=G
    - num_quantizers=R
    - eps=1e-5
    - transpose=True
    - 从 `vector_quantize_pytorch` 调用 `GroupedResidualFSQ(dim, levels, num_quantizers, groups)` 创建 self.quantizer
  - `self._embed(x)`:
    - 若转置, 则对 (1,2) 进行转置
    - x = rearrange(x, "b t (g r) -> g b t r", g=self.G, r=self.R)
    - 调用 `self.quantizer.get_output_from_indices(x)` 得到 feat, 若转置再转置
  - `self.forward(x)`:
    - 若转置, 则对 (1,2) 进行转置
    - feat, ind = self.quantizer(x)
    - 使用 one_hot 编码得到 embed_onehot
    - 对前两个维度取均值得到 e_mean, 然后归一化 eps 防止除零.
    - 困惑度 perplexity = exp(-sum(e_mean * log(e_mean + eps)))
    - 返回: 全零困惑度? feat转置, 困惑度, None, ind转置 #TODO ? 和训练相关
- `ConvNeXtBlock` 类: <a id="ConvNeXtBlock"></a> 
  - 来自于 Vocos
  - `self.__init__()`:
    - 创建 `self.dwconv`: `nn.Conv1d(dim, dim, kernel, dilation*(kernel//2), dilation, groups=dim)`
    - 创建 `self.norm`: `nn.Layernorm(dim, eps=1e-6)`
    - 创建 `self.pwconv1`: `nn.Linear(dim, intermediate_dim)`
    - 创建 `self.act`: `nn.GELU()`
    - 创建 `self.pwconv2`: `nn.Linear(intermediate_dim, dim)`
    - 创建 `self.gamma`: `layer_scale_init_value=1e-6` * `dim` 长度的全一张量
  - `self.forward(x, cond?)`:
    - residual = x
    - x -> `self.dwconv` [B,C,T] -> `transpose(1,2)` [B,T,C]-> `self.norm` -> `self.pwconv1` -> `self.act` -> `self.pwconv2` -> * `self.gamma` -> `transpose(1,2)` [B, C, T]
    - x = x + residual
    - return x

---

下面分析 `model/gpt.py`:


- `GPT_wrapper` 类: <a id="GPT_wrapper"></a> 
  - `self.build_model(config)`:
    - 调用 LLaMAConfig 读取 config, 然后从 transformers 调用并创建 model = LlamaModel(cfg)
    - 注: 删除了 embed_tokens 属性.
    - 赋给 `self.gpt`
    - `self.model_dim` = self.gpt.config.hidden_size
  - `self.emb_code`: num_vq=4 个 `nn.Embedding(num_audio_tokens, self.model_dim)`
  - `self.emb_text`: `nn.Embedding(num_text_tokens, self.model_dim)`
  - `self.head_text`: `weight_norm(nn.Linear(self.model_dim, num_text_tokens, bias=False))`
  - `self.head_code`: num_vq=4 个 `weight_norm(nn.Linear(self.model_dim, num_audio_tokens, bias=False))`
  - `self.get_emb(input_ids, text_mask)`:
    - 取得 `self.emb_text(input_ids[text_mask][:,0])` 
    - emb_code = 循环 num_vq 取得 `self.emb_code[i](input_ids[~text_mask][:,i])`
    - 然后按轴 2 堆叠.
    - 创建 `emb_text` + `emb_code` 大小的 `emb`
    - 然后赋值.
  - **`self.generate()`**:
    - 循环 max_new_token:
      - model_input = `self.prepare_inputs_for_generation()`
      - outputs = `self.gpt.forward(**model_input, output_attentions=return_attn)`
      - 用 self.head_text 得到 logits, 然后处理 logits (top k, top p), 用 softmax 得到 scores
      - idx_next = torch.multinomial(scores, 1)
      - 判断是否终止

假设文本为 `你可以的` 四个字, 得到的 `input_ids` 尺寸为: `(1, 6, 4)`
```
tensor([[[21134, 21134, 21134, 21134],
         [  872,   872,   872,   872],
         [ 1377,  1377,  1377,  1377],
         [  809,   809,   809,   809],
         [ 4638,  4638,  4638,  4638],
         [21135, 21135, 21135, 21135]]], device='cuda:0')
```

emb 尺寸为 (1,6,768)
start_idx = 6
end_idx = tensor([0])
finish = tensor([False])
temperature 尺寸为 (1,1), rearrange ?
attention_mask 尺寸为 (1,6)
max_new_token = 384
attention_mask_cache 尺寸为 (1,390) = 1, 6+max_new_token, 将 attention_mask 赋值过去对应起始位置

若 i=0, 则创建 model_input
- position_ids: `[[0,1,2,3,4,5]]`
- cache_position: `[0,1,2,3,4,5]`
- past_key_values: None
- use_cache: True
- attention_mask: `[[True, True, True, True, True, True]]`
- inputs_embeds: emb

经过 LLaMA 模型的 forward 得到 outputs (BaseModelOutputWithPast)
- last_hidden_state: (1, 6, 768)
- past_key_values: 20 个 Tuple, 每个 Tuple 有两个 Tensor, 每个 Tensor 尺寸为 (1,12,6,64)
- attentions: None

然后将 attentions 加入列表中, 并将 last_hidden_state 赋值给 `hidden_states`

如果需要返回 hidden, 则将 `hidden_states[:,-1]` 添加到 hiddens 列表中.

若 `infer_text` 为真, 则调用 `self.head_text(hidden_states)` 得到 `logits` (1,6,21178);
logits 取第二个轴的最后一个元素, (1,21178)

logits_token = input_ids[:, start_idx:, 0] ?


## 第三方包

- transformers
- vocos
- vector-quantize-pytorch