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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import torch
import os



class MultisourceTimeSeriesDataset(Dataset):
    def __init__(self, device, dataframe, sequenceLength, useScaler=False, useHeightAndWeight=True, modelType=None):
        self.sequenceLength = sequenceLength
        self.modelType = modelType
        self.useScaler = useScaler
        self.useHeightAndWeight = useHeightAndWeight
        self.data, self.demographics, self.targets = self.preprocessData(dataframe, device)
        self.subjectIds = list(self.data.keys())


    def preprocessData(self, dataframe, device):
        dataDict = {}
        targetDict = {}
        demographicDict = {}

        targetColumns = ['Ground Truth SBP', 'Ground Truth DBP']
        if self.useHeightAndWeight:
            demographicColumns = ['Age', 'Gender', 'Height', 'Weight']
        else:
            demographicColumns = ['Age', 'Gender']


        if self.useScaler:
            demographicscalerPath ="train_BP_prediction_models/" + self.modelType + '/demographicscaler.save'
            featurescalerPath =    "train_BP_prediction_models/" + self.modelType + '/featurescaler.save'
            sbpScalerPath =        "train_BP_prediction_models/" + self.modelType + '/SBPscaler.save'
            dbpScalerPath =        "train_BP_prediction_models/" + self.modelType + '/DBPscaler.save'

            if not os.path.exists(featurescalerPath):
                labelfreeData = dataframe.drop(columns=['Subject', 'Ground Truth SBP', 'Ground Truth DBP', 'Length', 'Age', 'Gender', 'Height', 'Weight']).values
                featureScaler = MinMaxScaler()
                featureScaler = featureScaler.fit(labelfreeData)
                joblib.dump(featureScaler, featurescalerPath)

            else:
                featureScaler = joblib.load(featurescalerPath)

            if not os.path.exists(sbpScalerPath):
                labelData = dataframe['Ground Truth SBP'].values
                sbpScaler = MinMaxScaler()
                sbpScaler = sbpScaler.fit(labelData.reshape(-1, 1))
                joblib.dump(sbpScaler, sbpScalerPath)

            else:
                sbpScaler = joblib.load(sbpScalerPath)

            if not os.path.exists(dbpScalerPath):
                labelData = dataframe['Ground Truth DBP'].values
                dbpScaler = MinMaxScaler()
                dbpScaler = dbpScaler.fit(labelData.reshape(-1, 1))
                joblib.dump(dbpScaler, dbpScalerPath)

            else:
                dbpScaler = joblib.load(dbpScalerPath)



            if not os.path.exists(demographicscalerPath):
                demographicData = dataframe[demographicColumns].values
                demographicScaler = MinMaxScaler()
                demographicScaler = demographicScaler.fit(demographicData)
                joblib.dump(demographicScaler, demographicscalerPath)

            else:
                demographicScaler = joblib.load(demographicscalerPath)



        


        for subjectId, group in dataframe.groupby('Subject'):
            if (self.useHeightAndWeight and not (np.isnan(torch.FloatTensor(group[demographicColumns].values)).any())) or not self.useHeightAndWeight:

                data = group.drop(columns=['Subject', 'Ground Truth SBP', 'Ground Truth DBP', 'Length', 'Age', 'Gender', 'Height', 'Weight']).values

                sbp = group['Ground Truth SBP'].values
                dbp = group['Ground Truth DBP'].values
      
                
                demographic = group[demographicColumns].values

                if self.use_scaler:
                    data = featureScaler.transform(data)
                    scaledSbp = sbpScaler.transform(sbp.reshape(-1,1))
                    scaledDbp = dbpScaler.transform(dbp.reshape(-1,1))
                    target = np.hstack([scaledSbp, scaledDbp])
                    demographic = demographicScaler.transform(demographic)


                targetDict[subjectId] = torch.FloatTensor(target).to(device)

                demographicDict[subjectId] = torch.FloatTensor(demographic).to(device)

                
                dataDict[subjectId] = torch.FloatTensor(data).to(device)

        return dataDict, demographicDict, targetDict

    def __len__(self):
        return sum(len(self.data[subjectId]) - self.sequenceLength +1 for subjectId in self.subjectIds)

    def __getitem__(self, idx):

        startIdx = idx
        for subjectId in self.subjectIds:
            if startIdx < len(self.data[subjectId])-self.sequenceLength+1:
                #idx belongs to this subjectId
                #print("idx ", idx, " belongs to subject ", subjectId, " with ", len(self.data[subjectId]),
                #      "entries. The current subject-specific index is: ", startIdx)
                # use self.sequenceLength -1 inputs for forecasting the self.sequenceLengths values
                dataFromSubject = self.data[subjectId][startIdx:startIdx + self.sequenceLength-1]

                demographicsFromSubject = self.demographics[subjectId][startIdx]

                featuresFromSubject = self.data[subjectId][startIdx + self.sequenceLength-1]

                groundTruth = self.targets[subjectId][startIdx + self.sequenceLength-1]

                return dataFromSubject, demographicsFromSubject, featuresFromSubject, groundTruth
            else:
                #idx doesn't belong to this subject, iterate to next subject after adapting the startIdx
                startIdx=startIdx - len(self.data[subjectId]) + self.sequenceLength -1
