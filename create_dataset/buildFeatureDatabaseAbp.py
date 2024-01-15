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

from create_dataset.utils.featureUtils import (scaleCurveUsingGroundTruthBpAndRatioToFirstCurve,
                                               findPeaksAndValleysWithMinPeakProminenceFilter,
                                               featureDetection,
                                               )
from create_dataset.utils.AbpDatasetCreationUtils import *
import pandas as pd
import numpy as np
import operator
import os.path
import scipy
import tqdm
import sys

sys.path.append('..')



def buildDataset(data, storagePath):

    subjects, uniqueSubjects = extractUniqueSubjects(data)

    print("we have ", len(uniqueSubjects), " unique subjects in this dataset.")
    for subject in tqdm.tqdm(uniqueSubjects):
        firstOccurence = subjects.index(subject)
        lastOccurence = len(subjects) - operator.indexOf(reversed(subjects), subject) - 1

        extractDataPerSubject(data, subject, firstOccurence, lastOccurence, storagePath)

def extractDataPerSubject(data, subjectId, firstOccurence, lastOccurence, storagePath):

    fieldNames = ['Subject', 'Gender', 'Age', 'Height', 'Weight',
                   'Ground Truth SBP', 'Ground Truth DBP', 'PPG SBP', 'PPG DBP',
                   'PR', 'DicroticNotch', 'AP', 'PP', 'TR',
                   'Systolic_Upstroke_Time', 'Diastolic_Time',
                   'SW10', 'DW10', 'SW25', 'DW25',
                   'SW33', 'DW33', 'SW50', 'DW50',
                   'SW66', 'DW66', 'SW75', 'DW75', 'Length']
    featureDf = pd.DataFrame(columns=fieldNames)

    referencePulsewave = pd.read_csv('referencePulsewaveForCorrelationBasedFiltering-[0,1]scaled.csv',
                                     delimiter=',')
    referencePulsewave = np.array(referencePulsewave).flatten()
    referencePpForScaling = 1
    ppOfFirstCurveForRatioComparison = 1
    pulsewaveCounter = 0



    refSYS = data['SBP'][firstOccurence]
    refDIA = data['DBP'][firstOccurence]


    for i in range(firstOccurence, lastOccurence):
        testsubjectDataSignals = data['Signals'][i]
        abpSignal = testsubjectDataSignals[2]
        abpPeaks, abpPeakIndices, abpValleys, abpValleyIndices = findPeaksAndValleysWithMinPeakProminenceFilter(abpSignal)

        averageSbpForSegment = data['SBP'][i]
        averageDbpForSegment = data['DBP'][i]
        age = data['Age'][i]
        gender = np.array(data['Gender'][i]).squeeze()
        gender = (gender == 'M').astype(float)
        height = data['Height'][i]
        weight = data['Weight'][i]


        for iCurve in tqdm.tqdm(range(0, len(abpValleyIndices)-1)):

            singleAbpCurve = abpSignal[abpValleyIndices[iCurve]:abpValleyIndices[iCurve+1]]

            # since the sampling rate for the ABP data was 125Hz, 125 samples correspond to 1 second
            # 25 samples thus correspond to 0.2 seconds, equal to a pulse rate of 300 bpm
            # we use this to discard noise immediately
            if len(singleAbpCurve)>25:
                resampledReference = scipy.signal.resample(referencePulsewave, 100)
                resampledAbpCurve = scipy.signal.resample(np.array(singleAbpCurve), 100)
                corrcoef = np.corrcoef(resampledReference, resampledAbpCurve)


                if np.abs(corrcoef[0][1]) > 0.8:

                    if referencePpForScaling == 1:
                        # This gives us the Pulse Pressure of the first high quality ABP curve of the patient
                        # we use this factor to scale the curve to the expect range after we [0,1]-scale it
                        referencePpForScaling = (np.max(singleAbpCurve) - np.min(singleAbpCurve))
                        ppOfFirstCurveForRatioComparison = referencePpForScaling


                    curveForFeatureExtraction = scaleCurveUsingGroundTruthBpAndRatioToFirstCurve(singleAbpCurve, referencePpForScaling, ppOfFirstCurveForRatioComparison)

                    (pulsewaveBeginning, SBP, DBP,
                     PR, DicroticNotch, AP, PP, TR,
                     SystolicUpstrokeTime, DiastolicTime,
                     SW10, DW10, SW25, DW25, SW33, DW33,
                     SW50, DW50, SW66, DW66, SW75, DW75) = featureDetection(curveForFeatureExtraction,
                                                                            iPulse=pulsewaveCounter,
                                                                            plotForTest=False)

                    pulsewaveCounter+=1

                    groundTruthSbp = int(np.round(np.max(singleAbpCurve)))
                    groundTruthDbp = int(np.round(np.min(singleAbpCurve)))


                    rowDict = {'Subject': subjectId,
                                'Gender': gender,
                                'age': age,
                                'Height': height,
                                'Weight': weight,
                                'Ground Truth SBP': groundTruthSbp,
                                'Ground Truth DBP': groundTruthDbp,
                                'PPG SBP': SBP, 'PPG DBP': DBP,
                                'PR': PR, 'DicroticNotch': DicroticNotch,
                                'AP': AP, 'PP': PP, 'TR': TR,
                                'Systolic_Upstroke_Time': SystolicUpstrokeTime,
                                'Diastolic_Time': DiastolicTime,
                                'SW10': SW10, 'DW10': DW10,
                                'SW25': SW25, 'DW25': DW25,
                                'SW33': SW33, 'DW33': DW33,
                                'SW50': SW50, 'DW50': DW50,
                                'SW66': SW66, 'DW66': SW66,
                                'SW75': SW75, 'DW75': DW75, 'Length': len(curveForFeatureExtraction[pulsewaveBeginning:])}
                    currentFeatures = pd.DataFrame([rowDict])
                    featureDf = pd.concat([featureDf, currentFeatures], ignore_index=True)
                    currentFeatures.to_csv(storagePath, index=False, mode="a",
                                            header=not os.path.isfile(storagePath))

def useDatasetToExtractFeatures(datasetName):
    data = loadMatlabData(matDataPath + datasetName + '.mat')
    data = data['Subset']

    storagepathToSubset = "data/ABP/" + datasetName + '.csv'
    if os.path.exists(storagepathToSubset):
        os.remove(storagepathToSubset)

    buildDataset(data=data, storagePath=storagepathToSubset)


if __name__ == "__main__":
    matDataPath = 'PulseDB/Subset_Files/'

    useDatasetToExtractFeatures('Train_Subset')
    useDatasetToExtractFeatures('AAMI_Cal_Subset')
    useDatasetToExtractFeatures('AAMI_Test_Subset')
    useDatasetToExtractFeatures('CalBased_Test_Subset')
    useDatasetToExtractFeatures('CalFree_Test_Subset')