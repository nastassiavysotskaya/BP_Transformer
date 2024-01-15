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

from train_BP_prediction_models.dataloader import MultisourceTimeSeriesDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import torch


def calcRegressionMetricsBP(sbpGT, dbpGT, sbpPredicted, dbpPredicted, regressionMetric):

    if regressionMetric == 'mae':
        sbpMetric = mean_absolute_error(sbpGT, sbpPredicted)
        dbpMetric = mean_absolute_error(dbpGT, dbpPredicted)

    elif regressionMetric == 'rmse':
        sbpMetric = torch.sqrt(torch.tensor(mean_squared_error(sbpGT, sbpPredicted)))
        dbpMetric = torch.sqrt(torch.tensor(mean_squared_error(dbpGT, dbpPredicted)))

    elif regressionMetric == 'r2':
        sbpMetric = r2_score(sbpGT, sbpPredicted)
        dbpMetric = r2_score(dbpGT, dbpPredicted)

    metric = sbpMetric + dbpMetric

    return metric


def abpReturnTrainAndValDataloader(device, filename, batchSize, sequenceLength, useScaler, useHeightAndWeight, shuffle, modelType):
    # Read in training dataset
    dfTotal = pd.read_csv(filename)
    dfX = dfTotal.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
    dfy = dfTotal[['Ground Truth SBP', 'Ground Truth DBP']]

    trainFeatures, valFeatures, trainTarget, valTarget = train_test_split(dfX, dfy, test_size=0.2,
                                                                              random_state=42)  # Adjust test_size and random_state as needed

    # Create train and validation DataFrames
    trainDf = pd.concat([trainFeatures, trainTarget], axis=1)
    valDf = pd.concat([valFeatures, valTarget], axis=1)



    # Create a MultisourceTimeSeriesDataset
    trainDataset = MultisourceTimeSeriesDataset(device, trainDf, sequenceLength, useScaler=useScaler,
                                                 useHeightAndWeight=useHeightAndWeight, modelType=modelType)
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle)


    # Create a Validation DataLoader from MultisourceTimeSeriesDataset
    valDataset = MultisourceTimeSeriesDataset(device, valDf, sequenceLength, useScaler=useScaler,
                                               useHeightAndWeight=useHeightAndWeight, modelType=modelType)
    valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle)



    return trainDataloader, valDataloader

def radarReturnTrainValTestDataloader(device, featuresetFile, batchSize, sequenceLength, useScaler, useHeightAndWeight, shuffle, modelType):
    # Read in training dataset
    dfTotal = pd.read_csv(featuresetFile)
    dfX = dfTotal.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
    dfy = dfTotal[['Ground Truth SBP', 'Ground Truth DBP']]

    trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(dfX,
                                                                                dfy,
                                                                                test_size=0.2,
                                                                                random_state=42)

    trainFeatures, valFeatures, trainTarget, valTarget = train_test_split(trainFeatures,
                                                                              trainTarget,
                                                                              test_size=0.15,
                                                                              random_state=42)

    # Create train and validation DataFrames
    trainDf = pd.concat([trainFeatures, trainTarget], axis=1)
    valDf = pd.concat([valFeatures, valTarget], axis=1)
    testDf = pd.concat([testFeatures, testTarget], axis=1)



    # Create a Train DataLoader from MultisourceTimeSeriesDataset
    trainDataset = MultisourceTimeSeriesDataset(device, trainDf, sequenceLength, useScaler=useScaler,
                                                 useHeightAndWeight=useHeightAndWeight, modelType=modelType)
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle)


    # Create a Validation DataLoader from MultisourceTimeSeriesDataset
    valDataset = MultisourceTimeSeriesDataset(device, valDf, sequenceLength, useScaler=useScaler,
                                               useHeightAndWeight=useHeightAndWeight, modelType=modelType)
    valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle)

    # Create a Test DataLoader from MultisourceTimeSeriesDataset
    testDataset = MultisourceTimeSeriesDataset(device, testDf, sequenceLength, useScaler=useScaler,
                                                useHeightAndWeight=useHeightAndWeight, modelType=modelType)
    testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=shuffle)

    return trainDataloader, valDataloader, testDataloader


# copied from attention-is-all-you-need-pytorch/transformer/Optim.py by jadore801120
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=2.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
