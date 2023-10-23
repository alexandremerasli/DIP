import torch
import torch.nn as nn



class MeanOnlyBatchNorm(nn.Module):
    
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.Tensor(num_features))
        self.bias.data.zero_()

    def forward(self, inp):
        size = list(inp.size())
        beta = self.bias.view(1, self.num_features, 1, 1) 
        avg = torch.mean(inp.view(size[0], self.num_features, -1), dim=(0,2,3)) 

        output = inp - avg.view(size[0], size[1], 1, 1)
        output = output + beta

        return output


def bn(num_features,mean_only=False):
    if mean_only:
        return MeanOnlyBatchNorm(num_features)
    else:
        return nn.BatchNorm2d(num_features)
    #return nn.BatchNorm2d(num_features)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight'):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        _,w_svd,_ = torch.svd(w.view(height,-1).data, some=False, compute_uv=False)
        sigma = w_svd[0]
        sigma = torch.max(torch.ones_like(sigma),sigma/self.ln_lambda)
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def get_kernel(kernel_width=5, sigma=0.5):

    kernel = np.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.)/2.
    sigma_sq =  sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center)/2.
            dj = (j - center)/2.
            kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
            kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)

    kernel /= kernel.sum()

    return kernel

class gaussian(nn.Module):
    def __init__(self, n_planes,  kernel_width=5, sigma=0.5):
        super().__init__()
        self.n_planes = n_planes
        self.kernel = get_kernel(kernel_width=kernel_width,sigma=sigma)

        convolver = nn.ConvTranspose2d(n_planes, n_planes, kernel_size=5, stride=2, padding=2, output_padding=1, groups=n_planes)
        convolver.weight.data[:] = 0
        convolver.bias.data[:] = 0
        convolver.weight.requires_grad = False
        convolver.bias.requires_grad = False

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            convolver.weight.data[i, 0] = kernel_torch
        
        self.upsampler_ = convolver

    def forward(self, x):
        x = self.upsampler_(x)
        return x
   
# 模型
import torch.nn.functional as F


# 仿照上面的自定义一套conv， bn， up
def Conv(input_channels, output_channels,kernel_size=3, ln_lambda=2, stride=1, bias=True, pad='Replication'):
    """ 
        定义两种类型:
        1. ln_lamdba>0, lipschiz-controlled conv
        2. ln_lamdba=0, normale conv
    """
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'Replication':
        padder = nn.ReplicationPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=to_pad, bias=bias)

    if ln_lambda>0:
        convolver = SpectralNorm(convolver, ln_lambda)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def Up(upsample_mode,input_channels,sigma):
    """
        定义三种upsample：
        1. 一般deconv 一般转置卷积
        2. gaussian-controlled deconv 
        3. 非转置卷积 bilinear nearest
    """
    if upsample_mode == 'deconv':
        up= nn.ConvTranspose2d(input_channels, input_channels, 4, stride=2, padding=1)
    elif upsample_mode=='bilinear' or upsample_mode=='nearest':
        up = nn.Upsample(scale_factor=2, mode=upsample_mode,align_corners=False)
    elif upsample_mode == 'gaussian':
        up = gaussian(input_channels, kernel_width=5, sigma=sigma)
    else:
        assert False
        
    return up



