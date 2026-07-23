import pywt
import pywt.data
import torch
import torch.nn.functional as F
import torch.nn as nn


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def create_wavelet_filter_3d(wave, in_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)

    # Decomposition filters
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(1),
        dec_lo.unsqueeze(0).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(1),
        dec_lo.unsqueeze(0).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(1),
        dec_lo.unsqueeze(0).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(1),
        dec_hi.unsqueeze(0).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(1),
        dec_hi.unsqueeze(0).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(1),
        dec_hi.unsqueeze(0).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(0) * dec_lo.unsqueeze(1).unsqueeze(1),
        dec_hi.unsqueeze(0).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(0) * dec_hi.unsqueeze(1).unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1, 1)

    # Reconstruction filters
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(1),
        rec_lo.unsqueeze(0).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(1),
        rec_lo.unsqueeze(0).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(1),
        rec_lo.unsqueeze(0).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(1),
        rec_hi.unsqueeze(0).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(1),
        rec_hi.unsqueeze(0).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(1),
        rec_hi.unsqueeze(0).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(0) * rec_lo.unsqueeze(1).unsqueeze(1),
        rec_hi.unsqueeze(0).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(0) * rec_hi.unsqueeze(1).unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(in_size, 1, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform_3d(x, filters):
    b, c, d, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1, filters.shape[4] // 2 - 1)
    x = F.conv3d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 8, d // 2, h // 2, w // 2)
    return x

def inverse_wavelet_transform_3d(x, filters):
    b, c, _, d_half, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1, filters.shape[4] // 2 - 1)
    x = x.reshape(b, c * 8, d_half, h_half, w_half)
    x = F.conv_transpose3d(x, filters, stride=2, groups=c, padding=pad)
    return x



def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin


def dwt_init(x):
    if x.dim() == 5:
        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    if x.dim() == 6:
        r = 2
        in_batch, in_channel, _, in_spectra, in_height, in_width = x.size()
        out_batch, out_channel, out_spectra, out_height, out_width = in_batch,in_channel, in_spectra, r * in_height, r * in_width
        # x1 = x[0:out_batch, :, :, :] / 2
        # x2 = x[out_batch:out_batch * 2, :, :, :, :] / 2
        # x3 = x[out_batch * 2:out_batch * 3, :, :, :, :] / 2
        # x4 = x[out_batch * 3:out_batch * 4, :, :, :, :] / 2
        x1 = x[:,:,0,:,:,:] / 2
        x2 = x[:,:,1,:,:,:] / 2
        x3 = x[:,:,2,:,:,:] / 2
        x4 = x[:,:,3,:,:,:] / 2

        h = torch.zeros([out_batch, out_channel, out_spectra, out_height,
                        out_width]).float().to(x.device)

        h[:, :, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def dwt_init_old(x):
    if x.dim() == 5:
        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init_old(x):
    if x.dim() == 5:
        r = 2
        in_batch, in_channel, in_spectra, in_height, in_width = x.size()
        out_batch, out_channel, out_spectra, out_height, out_width = int(in_batch/(r**2)),in_channel, in_spectra, r * in_height, r * in_width
        x1 = x[0:out_batch, :, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_spectra, out_height,
                        out_width]).float().to(x.device)

        h[:, :, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


def wavelet_transform_3d_dwt(x, func):
    x_LL, x_HL, x_LH, x_HH = func(x)
    x = torch.cat([x_LL.unsqueeze(2), x_HL.unsqueeze(2), x_LH.unsqueeze(2), x_HH.unsqueeze(2)], dim=2)
    return x

def inverse_wavelet_transform_3d_iwt(x, func):
    x = func(x)
    return x


# 测试代码
if __name__ == "__main__":
    B, C, D, H, W = 2, 3, 34, 128, 128  # 高光谱数据：34 个光谱通道
    x = torch.rand(B, C, D, H, W)  # 假设是光谱数据

    tmp = dwt_init_old(x)
    out = iwt_init_old(tmp)

    print("输入:", x.shape)
    print("中间:", tmp.shape)
    print("输出:", out.shape)

    mae_loss = F.l1_loss(x, out, reduction='mean')  # 计算均值误差
    print("Mean Absolute Error (MAE):", mae_loss.item())


    tmp = wavelet_transform_3d_dwt(x, DWT())
    out = inverse_wavelet_transform_3d_iwt(tmp, IWT())

    print("输入:", x.shape)
    print("中间:", tmp.shape)
    print("输出:", out.shape)

    mae_loss = F.l1_loss(x, out, reduction='mean')  # 计算均值误差
    print("Mean Absolute Error (MAE):", mae_loss.item())