import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)

        x = F.silu(x)

        return self.linear_2(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttention):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, paddding=0)
    
    def forward(self, feature, time):

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqeeze(-1).unsqeeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNetAttention(nn.Module):
    def __init__(self, n_head:int, n_embed:int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_embed
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=True)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=True)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4 * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        
        long_residue = x

        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view((n, c, h * w))

        x = x.transpose(-1, -2)

        short_residue = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += short_residue

        x + self.layernorm_2(x)
        x = self.attention_2(x)
        x += short_residue

        short_residue = x

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += short_residue

        x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_out(x) + long_residue


class UNetOutput(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groupnorm(x)
        x = F.silu(x)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttention(8, 40)),
            
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttention(8, 40)),
            
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(320, 640), UNetAttention(8, 80)),
            
            SwitchSequential(UNetResidualBlock(640, 640), UNetAttention(8, 80)),
            
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(640, 1280), UNetAttention(8, 160)),
            
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttention(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNetResidualBlock(1280, 1280)),

            SwitchSequential(UNetResidualBlock(1280, 1280)),
        ])

        self.bottle_neck = SwitchSequential(
            UNetResidualBlock(1280, 1280),

            UNetAttention(8, 160),

            UNetResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttention(8, 160)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttention(8, 160)),
            
            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttention(8, 160), Upsample(1280)),
            
            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttention(8, 80)),
            
            SwitchSequential(UNetResidualBlock(1280, 640), UNetAttention(8, 80)),
            
            SwitchSequential(UNetResidualBlock(960, 640), UNetAttention(8, 80), Upsample(640)),
            
            SwitchSequential(UNetResidualBlock(960, 320), UNetAttention(8, 40)),
            
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttention(8, 80)),
            
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttention(8, 40)),
        ])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutput(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        time = self.time_embedding(time)

        output = self.unet(latent, context, time)

        return self.final(output)
    