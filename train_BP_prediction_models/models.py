## ===========================================================================
## Copyright (C) 2024 Infineon Technologies AG
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
## 2. Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
## 3. Neither the name of the copyright holder nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.
## ===========================================================================

from torch import Tensor
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, dimModel: int, dropout: float = 0.1, maxLen: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(maxLen).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dimModel, 2) * (-math.log(10000.0) / dimModel))
        pe = torch.zeros(maxLen, 1, dimModel)
        pe[:, 0, 0::2] = torch.sin(position * divTerm)
        pe[:, 0, 1::2] = torch.cos(position * divTerm)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Define the RegressionTransformer model
class RegressionTransformer(nn.Module):
    def __init__(self, dimModel, inputFeatureDim, demographicFeatureDim, dropout, nHeads, hiddenDim):
        super(RegressionTransformer, self).__init__()

        # INFO
        self.modelType = 'Transformer'
        self.dimModel = dimModel

        # LAYERS
        self.embedding = nn.Linear(inputFeatureDim, dimModel) #nn.linear since we have continuous, non-categorical features
        self.posEncoder = PositionalEncoding(dimModel, dropout)
        self.transformer = nn.Transformer(d_model=dimModel, nhead=nHeads, dropout=dropout, batch_first=True, num_encoder_layers=2, num_decoder_layers=2)

        self.fcForSbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)
        self.fcForDbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.sys = nn.Linear(hiddenDim, 1)
        self.dias = nn.Linear(hiddenDim, 1)


    def forward(self, src, tgt, demographicFeatures, srcMask=None):
        src = self.embedding(src)
        src = self.posEncoder(src)

        tgt = self.embedding(tgt)
        tgt = self.posEncoder(tgt)

        if srcMask is not None:
            output = self.transformer(src=src, tgt=tgt, src_mask=srcMask)
        else:
            output = self.transformer(src=src, tgt=tgt)

        combinedFeatures = torch.cat((output[:,0,:], demographicFeatures), dim=-1)

        outputSbp = self.relu1(self.fcForSbp(combinedFeatures))
        outputDbp = self.relu2(self.fcForDbp(combinedFeatures))



        systole = self.sys(outputSbp)
        diastole = self.dias(outputDbp)
        return systole, diastole


class Baseline(nn.Module):
    def __init__(self, inputFeatureDim, demographicFeatureDim, dimModel, hiddenDim):
        super(Baseline, self).__init__()

        # INFO
        self.modelType = 'Baseline'
        # LAYERS
        self.embedding = nn.Linear(inputFeatureDim, dimModel) #nn.linear since we have continuous, non-categorical features
        self.fcForSbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)
        self.fcForDbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sys = nn.Linear(hiddenDim, 1)
        self.dias = nn.Linear(hiddenDim, 1)


    def forward(self, src, demographicFeatures):
        src = self.embedding(src)

        combinedFeatures = torch.cat((torch.squeeze(src, dim=1), demographicFeatures), dim=-1)

        outputSbp = self.relu1(self.fcForSbp(combinedFeatures))
        outputDbp = self.relu2(self.fcForDbp(combinedFeatures))

        systole = self.sys(outputSbp)
        diastole = self.dias(outputDbp)
        return systole, diastole
