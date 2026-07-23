import torch
import torch.nn as nn
from mmcv.ops.carafe import Tensor
import torch.nn.functional as F
from .WTConv import WTConv3d
import logging
from einops import rearrange

logger = logging.getLogger('base')

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.Sigmoid(),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

class Unet(nn.Module):
    def __init__(self, in_dim=34, out_dim=34, nc=40):
        super(Unet, self).__init__()
        self.conv1 = ConvBlock(in_dim, nc)
        self.conv2 = ConvBlock(nc, nc*2)
        self.conv3 = ConvBlock(nc*2, nc*4)
        self.conv4 = ConvBlock(nc*4+nc*2, nc*2)
        self.conv5 = ConvBlock(nc*2+nc, nc)
        self.out = ConvBlock(nc, out_dim)
        self.down1 = nn.MaxPool3d(kernel_size=2)
        self.down2 = nn.MaxPool3d(kernel_size=2)
        self.up1 = nn.ConvTranspose2d(nc*4, nc*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(nc*2, nc*2, 2, stride=2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):

        x_1 = self.relu(self.conv1(x))
        x_1_down = self.down1(x_1)
        
        x_2 = self.relu(self.conv2(x_1_down))
        x_2_down = self.down2(x_2)

        x_3 = self.relu(self.conv3(x_2_down))

        x_4_up = self.up1(x_3)
        x_4 = self.relu(self.conv4(torch.cat([x_4_up, x_2], dim=1)))

        x_5_up = self.up2(x_4)
        x_5 = self.relu(self.conv5(torch.cat([x_5_up, x_1], dim=1)))

        out = self.out(x_5) + x
        return out

class ConvBlock_3D(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock_3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.Sigmoid(),
        )
        self.conv11 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out
    
class Unet_3D(nn.Module):
    def __init__(self, in_dim=1, nc=16):
        super(Unet_3D, self).__init__()
        self.conv1 = ConvBlock_3D(in_dim, nc)
        self.conv2 = ConvBlock_3D(nc, nc*2)
        self.conv3 = ConvBlock_3D(nc*2, nc*4)
        self.conv4 = ConvBlock_3D(nc*4+nc*2, nc*2)
        self.conv5 = ConvBlock_3D(nc*2+nc, nc)
        self.out = ConvBlock_3D(nc, 1)
        self.down1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.down2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.up1 = nn.ConvTranspose3d(nc*4, nc*4, (1,2,2), stride=(1,2,2))
        self.up2 = nn.ConvTranspose3d(nc*2, nc*2, (1,2,2), stride=(1,2,2))
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):

        x_1 = self.relu(self.conv1(x))
        x_1_down = self.down1(x_1)
        
        x_2 = self.relu(self.conv2(x_1_down))
        x_2_down = self.down2(x_2)

        x_3 = self.relu(self.conv3(x_2_down))

        x_4_up = self.up1(x_3)
        x_4 = self.relu(self.conv4(torch.cat([x_4_up, x_2], dim=1)))

        x_5_up = self.up2(x_4)
        x_5 = self.relu(self.conv5(torch.cat([x_5_up, x_1], dim=1)))

        out = self.out(x_5) + x
        return out
    

class Unet_3D_noise_frecom_DWTIWT_WTConv3d_decoder(nn.Module):
    def __init__(self, in_dim=1, nc=16):
        super(Unet_3D_noise_frecom_DWTIWT_WTConv3d_decoder, self).__init__()
        self.conv1 = ConvBlock_3D(in_dim, nc)
        self.conv2 = ConvBlock_3D(nc, nc*2)
        self.conv3 = ConvBlock_3D(nc*2+in_dim, nc*4)
        self.conv4 = ConvBlock_3D(nc*4+nc*2+in_dim, nc*2)
        self.conv5 = ConvBlock_3D(nc*2+nc+in_dim, nc)
        self.out = ConvBlock_3D(nc, 1)
        self.down1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.down2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.up1 = nn.ConvTranspose3d(nc*4, nc*4, (1,2,2), stride=(1,2,2))
        self.up2 = nn.ConvTranspose3d(nc*2, nc*2, (1,2,2), stride=(1,2,2))
        self.relu = nn.LeakyReLU(inplace=True)

        self.watconv1 = WTConv3d(in_dim, in_dim, kernel_size=3, wt_levels=2)
        self.wa_down1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.watconv2 = WTConv3d(in_dim, in_dim, kernel_size=3, wt_levels=2)
        self.wa_down2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.watconv3 = WTConv3d(in_dim, in_dim, kernel_size=3, wt_levels=2)
    
    def forward(self, x):

        x_wa_1 = self.watconv1(self.wavelet_noise_extraction_torch(x))
        x_wa_1_down = self.wa_down1(x_wa_1)
        x_wa_2 = self.watconv2(x_wa_1_down)
        x_wa_2_down = self.wa_down2(x_wa_2)
        x_wa_3 = self.watconv3(x_wa_2_down)

        x_1 = self.relu(self.conv1(x))
        x_1_down = self.down1(x_1)
        
        x_2 = self.relu(self.conv2(x_1_down))
        x_2_down = self.down2(x_2)
        
        x_3 = self.relu(self.conv3(torch.cat([x_2_down, x_wa_3], dim=1)))

        x_4_up = self.up1(x_3)
        x_4 = self.relu(self.conv4(torch.cat([x_4_up, x_2, x_wa_2], dim=1)))

        x_5_up = self.up2(x_4)
        x_5 = self.relu(self.conv5(torch.cat([x_5_up, x_1, x_wa_1], dim=1)))

        out = self.out(x_5) + x
        return out
    
    def haar_wavelet_2d(self, x):
        """
        对输入张量的 H, W 维度进行 2D Haar 小波变换，返回低频和高频分量。
        输入: x [B, C, D, H, W]
        输出: LL, LH, HL, HH (低频和高频细节部分)
        """
        B, C, D, H, W = x.shape
        
        # 进行 Haar 变换，相邻像素进行求和与求差
        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2

        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]

        # 低频分量 (LL) 和高频细节 (LH, HL, HH)
        LL = x1 + x2 + x3 + x4  # 低频
        LH = -x1 - x2 + x3 + x4  # 高频（行方向）
        HL = -x1 + x2 - x3 + x4  # 高频（列方向）
        HH = x1 - x2 - x3 + x4  # 高频（对角线）

        return LL, LH, HL, HH
    
    def haar_wavelet_2d_inverse(self, LH, HL, HH):
        """
        对输入的 Haar 小波变换分量 LL, LH, HL, HH 进行 2D Haar 小波逆变换。
        此函数是特定前向变换 haar_wavelet_2d 的逆操作。

        输入:
            LL (torch.Tensor): 低频分量 [B, C, D, H/2, W/2]
            LH (torch.Tensor): 高频（行）分量 [B, C, D, H/2, W/2]
            HL (torch.Tensor): 高频（列）分量 [B, C, D, H/2, W/2]
            HH (torch.Tensor): 高频（对角线）分量 [B, C, D, H/2, W/2]
                            (注意: 为了满足重构要求，输入的 LL 应该已经被置零)

        输出:
            x_rec (torch.Tensor): 重构后的张量 [B, C, D, H, W]
        """
        # 确保所有输入张量在同一设备上且数据类型一致
        device = LH.device
        dtype = LH.dtype

        LL = torch.zeros_like(LH, device=device, dtype=dtype)  # 将 LL 置零

        # --- 步骤 1: 从 LL, LH, HL, HH 计算 x1, x2, x3, x4 ---
        # 注意：这里的因子是 1/4
        x1 = (LL - LH - HL + HH) / 4
        x2 = (LL - LH + HL - HH) / 4
        x3 = (LL + LH - HL - HH) / 4
        x4 = (LL + LH + HL + HH) / 4

        # --- 步骤 2: 重构原始张量 ---
        # 获取原始张量的尺寸
        B, C, D, H_half, W_half = LL.shape
        H = H_half * 2
        W = W_half * 2

        # 创建用于存储重构结果的张量
        # 使用 torch.empty 更高效，因为所有位置都会被填充
        x_rec = torch.empty(B, C, D, H, W, dtype=dtype, device=device)

        # 将计算出的分量放回原始张量的相应位置
        # 注意：需要乘以 2 来抵消前向变换中的预先除以 2
        # x1 -> 偶数行, 偶数列
        x_rec[:, :, :, 0::2, 0::2] = x1 * 2
        # x2 -> 奇数行, 偶数列
        x_rec[:, :, :, 1::2, 0::2] = x2 * 2
        # x3 -> 偶数行, 奇数列
        x_rec[:, :, :, 0::2, 1::2] = x3 * 2
        # x4 -> 奇数行, 奇数列
        x_rec[:, :, :, 1::2, 1::2] = x4 * 2

        return x_rec


    def wavelet_noise_extraction_torch(self, image):
        """
        提取噪声模式 (H, W 方向的小波高频部分)
        输入: image [B, C, D, H, W]
        输出: noise_pattern [B, C, D, H//2, W//2]
        """
        B, C, D, H, W = image.shape
        LL, LH, HL, HH = self.haar_wavelet_2d(image)
        noise_pattern = self.haar_wavelet_2d_inverse(LH, HL, HH)
        noise_pattern = torch.abs(noise_pattern)
        return noise_pattern