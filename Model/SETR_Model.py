import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
# 在 Transformer 模型中，每一层通常包括一个自注意力机制和一个前馈网络。
# PreNorm 类的作用是在这两层操作之前添加一个层归一化（Layer Normalization）步骤。
class PreNorm(nn.Module):
    def __init__(self, dim, fn):#dim是输入数据维度，fn归一化
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    # 正向传播
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 用于实现 Transformer 模型中的前馈网络（Feed Forward Network）。
# 前馈网络是 Transformer 模型中的一个重要组成部分，它通常位于自注意力机制之后，用于非线性变换和特征提取。
class FeedForward(nn.Module):
    # dim 是输入数据的维度，hidden_dim 是前馈网络中的隐藏层维度，
    # dropout 是前馈网络中使用的 Dropout 比率。
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), #全连接层，将输入数据的维度 dim 映射到隐藏层的维度 hidden_dim。
            nn.GELU(), #GELU（Gaussian Error Linear Unit）激活函数，非线性激活函数，其输出是输入的线性部分与一个高斯噪声的乘积。
            nn.Dropout(dropout), #创建一个 Dropout 层，它在训练过程中随机丢弃输入的一部分，以增加模型的鲁棒性。
            nn.Linear(hidden_dim, dim), #全连接层，将隐藏层的输出维度 hidden_dim 映射回输入数据的维度 dim。
            nn.Dropout(dropout) #创建另一个 Dropout 层，它具有与第一个 Dropout 层相同的 Dropout 比率。
        )

    def forward(self, x):
        return self.net(x)

# 实现 Transformer 模型中的自注意力机制。自注意力机制是 Transformer 模型的核心组成部分，
# 它允许模型在学习特征之间的依赖关系时，能够关注输入序列中不同部分的重要性。
class Attention(nn.Module):
    # dim 是输入数据的维度，heads 是注意力头的数量，dim_head 是每个注意力头的维度，
    # 而 dropout 是注意力机制中使用的 Dropout 比率。
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads #计算内部维度
        # 判断是否需要将注意力机制的输出投影回原始维度。如果 heads 等于 1 且 dim_head 等于 dim，则不需要投影。
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # 创建一个 nn.Softmax 层，并将其存储在 self.attend 属性中。
        # nn.Softmax 层用于计算注意力分数，并将它们缩放到 (0, 1) 范围内。
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 创建一个全连接层，用于将输入数据的维度 dim 映射到内部维度 inner_dim * 3。
        # 这个全连接层将用于生成查询（Query）、键（Key）和值（Value）向量。
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 将输入 x 通过 self.to_qkv 层，然后将其分割成三个部分，分别对应查询、键和值。
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 将分割后的查询、键和值向量重新排列，使其维度符合注意力机制的要求。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数，即查询和键的点积，并乘以缩放因子。
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        #
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., out_indices=(9, 14, 19, 23)):
        super().__init__()
        self.out_indices = out_indices
        assert self.out_indices[-1] == depth - 1

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        out = []
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index in self.out_indices:
                out.append(x)

        return out


# ViT 是一种将 Transformer 架构应用于图像处理的模型，
# 它通过将图像分割成固定大小的补丁（patches），然后将每个补丁转换为序列，从而将图像数据转换为适合 Transformer 处理的序列数据。
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., out_indices=(9, 14, 19, 23)):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices)

        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        out = self.transformer(x)

        for index, transformer_out in enumerate(out):
            # delete cls_tokens and transform output to [b, c, h, w]
            out[index] = self.out(transformer_out[:, 1:, :])

        return out


# 用于实现一个用于语义分割的头部模块。
# 这个头部模块包含了一系列的卷积层、批量归一化层、ReLU 激活函数和上采样层，最后是一个用于生成分割掩膜的卷积层。
class PUPHead(nn.Module):
    def __init__(self, num_classes):
        super(PUPHead, self).__init__()

        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.cls_seg = nn.Conv2d(256, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        x = self.UP_stage_4(x)
        x = self.cls_seg(x)
        return x


# 用于实现一个基于 Transformer 的语义分割模型。
# SETR 模型是一种结合了 ViT（Vision Transformer）和 PUP（Pyramid Upsampling Pyramid）的模型，
# 用于处理图像分割任务。
class SETR(nn.Module):
    '''
    num_classes（分割任务中类别的数量）、image_size（图像大小）、patch_size（补丁大小）、
    dim（Transformer 层的维度）、depth（Transformer 层的数量）、heads（注意力头的数量）、
    mlp_dim（前馈网络的隐藏层维度）、channels（图像的通道数，默认为 3）、dim_head（每个注意力头的维度）、
    dropout（嵌入层的 Dropout 比率）、emb_dropout（Transformer 层的 Dropout 比率）和
     out_indices（Transformer 层的输出索引）。
    '''
    def __init__(self, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0., out_indices=(9, 14, 19, 23)):
        super(SETR, self).__init__()
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.VIT = ViT(image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                       channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
                       out_indices=out_indices)

        self.Head = nn.ModuleDict()

        for index, indices in enumerate(self.out_indices):
            self.Head["Head" + str(indices)] = PUPHead(num_classes)

    def forward(self, x):
        VIT_OUT = self.VIT(x)

        out = []
        for index, indices in enumerate(self.out_indices):
            # 最后一个是最后层的输出
            out.append(self.Head["Head" + str(indices)](VIT_OUT[index]))
        return out


if __name__ == "__main__":
    # VIT-Large  设置了16个patch
    SETRNet = SETR(num_classes=3, image_size=256, patch_size=256 // 16, dim=1024, depth=24, heads=16, mlp_dim=2048,
                   out_indices=(9, 14, 19, 23)).cpu()
    img = torch.randn(1, 3, 256, 256).cpu()
    preds = SETRNet(img)
    for output in preds:
        print("output: ", output.size())
