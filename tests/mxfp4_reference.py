"""CPU-only MXFP4 reference: nearest representable value, ties to even code."""
import torch


def quantize(x):
    x = x.cpu().double().reshape(x.shape[0], -1, 32)
    bad = ~torch.isfinite(x).all(-1)
    finite = torch.where(torch.isfinite(x), x, 0.)
    mx = finite.abs().amax(-1)
    e = torch.ceil(torch.log2((mx / 6.).clamp_min(2.**-126))).clamp(-126, 127)
    e[mx == 0] = 0
    v = finite / torch.exp2(e)[..., None]
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    distance = (v.abs()[..., None] - levels).abs()
    nearest = distance == distance.amin(-1, keepdim=True)
    codes = torch.arange(8)
    priority = torch.where(codes % 2 == 0, codes, codes + 8)
    choice = torch.where(nearest, priority, 100).argmin(-1).to(torch.uint8)
    choice |= torch.signbit(v).to(torch.uint8) * 8
    packed = choice[..., ::2] | (choice[..., 1::2] << 4)
    packed[bad] = 0
    scale = (e + 127).to(torch.uint8)
    scale[bad] = 255
    return packed.flatten(1), scale


def dequantize(q, s):
    q, s = q.cpu(), s.cpu()
    c = torch.stack((q & 15, q >> 4), -1).flatten(1)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    v = levels[(c & 7).long()] * torch.where((c & 8) != 0, -1., 1.)
    scale = torch.exp2(s.double() - 127)
    scale[s == 255] = float('nan')
    return (v.reshape(q.shape[0], -1, 32) * scale[..., None]).flatten(1)
