import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as fnn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.GRUCell(input_size, hidden_size)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs = []
        for i in range(1):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()

class Generator(torch.nn.Module):
    def __init__(self, ngpu, z_dim, ngf, ndf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.z_dim = z_dim
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc 
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(z_dim, ngf * 4*int((nc+1)/2), 4, 1, 0, bias=False),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 4*int((nc+1)/2), ngf * 2*int((nc+1)/2), 4, 2,1, bias=False),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngf * 2*int((nc+1)/2), ngf * 1*int((nc+1)/2), 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(ngf * 1*int((nc+1)/2), ngf*int((nc+1)/4), 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(ngf*int((nc+1)/4), nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

