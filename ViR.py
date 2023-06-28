import math
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation, MLP
from collections import OrderedDict
from typing import Callable
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional

T_MAX = 1024
RWKV_FLOAT_MODE = "bf16"
ACCELERATOR = "gpu"

from torch.utils.cpp_extension import load

if RWKV_FLOAT_MODE == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
else:
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        if RWKV_FLOAT_MODE == "bf16":
            w = -torch.exp(w.float().contiguous())
            u = u.bfloat16().contiguous()
            k = k.bfloat16().contiguous()
            v = v.bfloat16().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
        elif "32" in RWKV_FLOAT_MODE:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        ctx.save_for_backward(w, u, k, v, y)
        if "32" in RWKV_FLOAT_MODE:
            return y
        elif RWKV_FLOAT_MODE == "fp16":
            return y.half()
        elif RWKV_FLOAT_MODE == "bf16":
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors
        if RWKV_FLOAT_MODE == "bf16":
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
        else:
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            if "32" in RWKV_FLOAT_MODE:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if "32" in RWKV_FLOAT_MODE:
            return (None, None, None, gw, gu, gk, gv)
        elif RWKV_FLOAT_MODE == "fp16":
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif RWKV_FLOAT_MODE == "bf16":
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

class RWKV_TimeMix(nn.Module):
    def __init__(
            self, 
            config: Dict[str, float]
        ):
        super().__init__()
        self.n_embd = config["n_embd"]
        self.dim_att = config["dim_att"]

        self.time_decay = nn.Parameter(torch.empty(config["dim_att"]))
        self.time_first = nn.Parameter(torch.empty(config["dim_att"]))
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, config["n_embd"]))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, config["n_embd"]))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, config["n_embd"]))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(config["n_embd"], config["dim_att"], bias=False)
        self.value = nn.Linear(config["n_embd"], config["dim_att"], bias=False) # Randomness needs to be provided here only.
        self.receptance = nn.Linear(config["n_embd"], config["dim_att"], bias=False)
        self.output = nn.Linear(config["dim_att"], config["n_embd"], bias=False)

        # Inits
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        for l in [self.time_decay, self.time_first, self.time_mix_k, self.time_mix_v, self.time_mix_r]:
            nn.init.normal_(l, std=0.02)

    def jit_func(self, x):
        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    @torch.jit.ignore
    def run_cuda(self, B, T, C, w, u, k, v):
        return WKV.apply(B, T, C, w, u, k, v)

    def forward(self, input: torch.Tensor):
        B, T, C = input.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(input)
        B = torch.tensor(B)
        T = torch.tensor(T)
        C = torch.tensor(self.dim_att)
        rwkv = sr * self.run_cuda(B, T, C, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)
        
class RWKV_ChannelMix(nn.Module):
    def __init__(
            self, 
            config: Dict[str, float]
        ):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, config["n_embd"]))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, config["n_embd"]))
        self.key = nn.Linear(config["n_embd"], config["dim_ffn"], bias=False)
        self.receptance = nn.Linear(config["n_embd"], config["n_embd"], bias=False)
        self.value = nn.Linear(config["dim_ffn"], config["n_embd"], bias=False)

        # Inits
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        for l in [self.time_mix_k, self.time_mix_r]:
            nn.init.normal_(l, std=0.02)

    def forward(self, input: torch.Tensor):
        input_ts = self.time_shift(input)
        input_k = input * self.time_mix_k + input_ts * (1 - self.time_mix_k)
        input_r = input * self.time_mix_r + input_ts * (1 - self.time_mix_r)
        k = self.key(input_k)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(input_r)) * kv

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        # Inits
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class Block(nn.Module):
    def __init__(
            self, 
            config: Dict[str, float], 
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
        ):
        super().__init__()

        self.ln1 = norm_layer(config["n_embd"])

        self.layers = nn.Sequential()
        for i in range(config["num_heads"]):
            self.layers.add_module(f"RWKV_TimeMix_{i}", RWKV_TimeMix(config))
            self.layers.add_module(f"RWKV_ChannelMix_{i}", RWKV_ChannelMix(config))

        self.dropout = nn.Dropout(config["dropout"])

        # MLP block
        self.ln2 = norm_layer(config["n_embd"])
        self.mlp = MLPBlock(config["n_embd"], config["dim_ffn"], config["dropout"])

    def forward(self, input: torch.Tensor):
        x = input + self.dropout(self.layers(self.ln1(input)))

        y = self.ln2(x)
        y = self.mlp(y)

        return x + y

class Encoder(nn.Module):
    """Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        config: Dict[str, float],
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_emb = nn.Parameter(torch.empty(((config["image_size"]//config["patch_size"])**2, config["n_embd"])).normal_(std=0.02))
        self.ln0 = nn.LayerNorm(config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(config["n_layer"]):
            layers[f"encoder_layer_{i}"] = Block(config)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(config["n_embd"])

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input[:, 1:] + self.pos_emb
        return self.ln(self.layers(self.ln0(self.dropout(input))))

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class VisionRWKV(nn.Module):
    def __init__(
        self,
        config: Dict[str, float],
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_classes: int,
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = Encoder(config)

        self.heads = nn.Linear(hidden_dim, num_classes)

        # Init
        if isinstance(self.conv_proj, nn.Conv2d):
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        nn.init.zeros_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, include_head: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        if include_head:
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
            x = self.heads(x)
        return x
    
