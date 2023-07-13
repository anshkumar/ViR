import math
import torch
import numpy as np
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation, MLP
from collections import OrderedDict
from typing import Callable
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional

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

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        device: Optional[torch.device] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = torch.empty(1, seq_length, hidden_dim).to(device)  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

        # Init 
        n = 10000
        for k in range(seq_length):
            for i in np.arange(int(hidden_dim/2)):
                denominator = np.power(n, 2*i/hidden_dim)
                self.pos_embedding[0, k, 2*i] = np.sin(k/denominator)
                self.pos_embedding[0, k, 2*i+1] = np.cos(k/denominator)
        # self.pos_embedding = nn.Parameter(self.pos_embedding)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # input = input + self.pos_embedding
        input = input[:, 1:] + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Dict[str, float],
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_classes: int,
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.zo_random_seed = np.random.randint(1000000000)
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.device = device

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

        self.encoder = Encoder(
            (config["image_size"]//config["patch_size"])**2,
            config["n_layer"],
            config["n_layer"],
            config["n_embd"],
            config["dim_ffn"],
            config["dropout"],
            0.0,
            device
            )

        self.heads = nn.Linear(hidden_dim, num_classes)

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

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

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * float(self.config["zo_eps"])

    def loss(self, inputs, labels, include_head=True):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        outputs = self(inputs, include_head=include_head)
        loss = self.criterion(outputs, labels)
        return loss
    
    def zo_step_layer(self, inputs, labels):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(self.zo_random_seed)

        losses = []
        for name, param in self.named_parameters_to_optim:
            # First function evaluation
            scaling_factor=1
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * float(self.config["zo_eps"])            
            loss1 = self.loss(inputs, labels)

            # Second function evaluation
            scaling_factor=-2
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * float(self.config["zo_eps"])
            loss2 = self.loss(inputs, labels)

            self.projected_grad = ((loss1 - loss2) / (2 * float(self.config["zo_eps"]))).item()

            # Reset model back to its parameters at start of step
            scaling_factor=1
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * float(self.config["zo_eps"])

            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.config["learning_rate"] * (self.projected_grad * z + self.config["weight_decay"] * param.data)
            else:
                param.data = param.data - self.config["learning_rate"] * (self.projected_grad * z)
            if self.device == torch.device("cuda"):
                losses.append(loss1.cpu())
            else:
                losses.append(loss1)
        return np.mean(losses)
    
    def zo_step(self, inputs, labels):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.loss(inputs, labels)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.loss(inputs, labels)

        self.projected_grad = ((loss1 - loss2) / (2 * float(self.config["zo_eps"]))).item()
        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)
        
        self.zo_update()
        return loss1
    
    def zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.config["learning_rate"] * (self.projected_grad * z + self.config["weight_decay"] * param.data)
            else:
                param.data = param.data - self.config["learning_rate"] * (self.projected_grad * z)

        # self.lr_scheduler.step()

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
    
