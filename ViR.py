import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import deepspeed
from collections import OrderedDict
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from typing import Callable
from functools import partial
from typing import Dict, Union

T_MAX = 1024
RWKV_FLOAT_MODE = "fp32"
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
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
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
            config: Dict[str, float], 
            layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = config["n_embd"]
        self.dim_att = config["dim_att"]

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (config["n_layer"] - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config["n_layer"])  # 1 to ~0
            ddd = torch.ones(1, 1, config["n_embd"])
            for i in range(config["n_embd"]):
                ddd[0, 0, i] = i / config["n_embd"]
            
            # fancy time_decay
            decay_speed = torch.ones(config["dim_att"])
            for h in range(config["dim_att"]):
                decay_speed[h] = -5 + 8 * (h / (config["dim_att"] - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(config["dim_att"])]) * 0.5
            self.time_first = nn.Parameter(torch.ones(config["dim_att"]) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(config["n_embd"], config["dim_att"], bias=False)
        self.value = nn.Linear(config["n_embd"], config["dim_att"], bias=False) # Randomness needs to be provided here only.
        self.receptance = nn.Linear(config["n_embd"], config["dim_att"], bias=False)
        self.output = nn.Linear(config["dim_att"], config["n_embd"], bias=False)

        # Init
        for l in [self.key, self.receptance, self.output]:
            nn.init.zeros_(l.weight)

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
            config: Dict[str, float], 
            layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / config["n_layer"])  # 1 to ~0
            ddd = torch.ones(1, 1, config["n_embd"])
            for i in range(config["n_embd"]):
                ddd[0, 0, i] = i / config["n_embd"]
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        self.key = nn.Linear(config["n_embd"], config["dim_ffn"], bias=False)
        self.receptance = nn.Linear(config["n_embd"], config["n_embd"], bias=False)
        self.value = nn.Linear(config["dim_ffn"], config["n_embd"], bias=False)

        # Init
        for l in [self.value, self.receptance]:
            nn.init.zeros_(l.weight)

    def forward(self, input: torch.Tensor):
        input_ts = self.time_shift(input)
        input_k = input * self.time_mix_k + input_ts * (1 - self.time_mix_k)
        input_r = input * self.time_mix_r + input_ts * (1 - self.time_mix_r)
        k = self.key(input_k)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(input_r)) * kv

class Block(nn.Module):
    def __init__(
            self, 
            config: Dict[str, float], 
            layer_id: int):
        super().__init__()

        ln1 = nn.LayerNorm(config["n_embd"])
        ln2 = nn.LayerNorm(config["n_embd"])

        att = RWKV_TimeMix(config, layer_id)
        ffn = RWKV_ChannelMix(config, layer_id)

        if layer_id == 0:
            ln0 = nn.LayerNorm(config["n_embd"])
            self.layers_1 = nn.Sequential(ln0, ln1, att)
        else:
            self.layers_1 = nn.Sequential(ln1, att)
        self.layers_2 = nn.Sequential(ln2, ffn)

    def forward(self, input: torch.Tensor):
        input = input + self.layers_1(input)
        input = input + self.layers_2(input)

        return input

class RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not hasattr(config, 'dim_att'):
            config["dim_att"] = config["n_embd"]
        if not hasattr(config, 'dim_ffn'):
            config["dim_ffn"] = config["n_embd"] * 4
        if not hasattr(config, 'tiny_att_layer'):
            config["tiny_att_layer"] = -1
        if not hasattr(config, 'tiny_att_dim'):
            config["tiny_att_dim"] = -1

        self.emb = nn.Embedding(config["vocab_size"], config["n_embd"])

        self.blocks = nn.ModuleList([Block(config, i) for i in range(config["n_layer"])])

        self.ln_out = nn.LayerNorm(config["n_embd"])
        self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

    def configure_optimizers(self):
        config = self.config
        if config["layerwise_lr"] > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0},
            ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
            ]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.config["lr_init"], betas=(self.config["beta_1"], self.config["beta_2"]), eps=self.config["adam_eps"], bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.config["lr_init"], betas=(self.config["beta_1"], self.config["beta_2"]), eps=self.config["adam_eps"], bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        config = self.config
        B, T = idx.size()

        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def generate_init_weight(self):
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.config["lr_init"]
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if ACCELERATOR.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if RWKV_FLOAT_MODE == "fp16":
                m[n] = m[n].half()
            elif RWKV_FLOAT_MODE == "bf16":
                m[n] = m[n].bfloat16()

        gc.collect()
        torch.cuda.empty_cache()
        return m

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        config: Dict[str, float],
        seq_length: int,
        hidden_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(config["n_layer"]):
            layers[f"encoder_layer_{i}"] = Block(config, i)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class VisionRWKV(nn.Module):
    def __init__(
        self,
        config: Dict[str, float],
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_classes: int
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.seq_length = seq_length
        self.encoder = Encoder(config, self.seq_length, config["n_embd"], 0.0)

        self.heads = nn.Linear(hidden_dim, num_classes)

        # Init
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

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
    