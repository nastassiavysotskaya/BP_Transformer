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

from models import RegressionTransformer, Baseline
from modelUtils import *
import numpy as np
import argparse
import joblib
import tqdm
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", default='todo/', type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["Baseline", "RegressionTransformer"])
    parser.add_argument("--data_path", default='../data/radar/', type=str)
    parser.add_argument("--shuffle", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_scaler", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_height_and_weight_demographics", default=True, type=bool,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--max_patience", default=10, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--correlation_threshold", default=0.8, type=float)

    args = parser.parse_args()
    trainedModelPath = args.trained_model_path
    dataPath = args.data_path
    correlationThreshold = args.correlation_threshold

    # PARSE BOOLEAN INPUTS FOR DATALOADER
    shuffle = args.shuffle
    useScaler = args.use_scaler
    useHeightAndWeightDemographics = args.use_height_and_weight_demographics

    # PARSE OTHER DATALOADER HYPERPARAMETERS
    batchSize = args.batch_size
    sequenceLength = args.sequence_length

    # PARSE MODEL HYPERPARAMETERS
    modelType = args.model_type
    dropout = args.dropout
    nHeads = args.num_heads
    nEpochs = args.num_epochs
    hiddenDim = args.hidden_dim
    maxPatience = args.max_patience
    learningRate = args.learning_rate

    featuresetFile = dataPath + "correlationthreshold_" + str(correlationThreshold) + "/radar_database_BP.csv"

    assert os.path.exists(
        featuresetFile), featuresetFile + "\nData for correlation threshold %.1f was not extracted yet." % (
        correlationThreshold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are using the following device: ", device)

    # create data loaders
    (trainDataloader,
     valDataloader,
     testDataloader) = radarReturnTrainValTestDataloader(device=device,
                                                         featuresetFile=featuresetFile,
                                                         batchSize=batchSize,
                                                         sequenceLength=sequenceLength,
                                                         useScaler=useScaler,
                                                         useHeightAndWeight=useHeightAndWeightDemographics,
                                                         shuffle=shuffle,
                                                         modelType=modelType)

    print("len(train_dataloader): ", len(trainDataloader))
    print("len(val_dataloader): ", len(valDataloader))
    print("len(test_dataloader): ", len(testDataloader))
    # train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), leave=False)

    # input feature dimensions
    inputDim = 21
    if useHeightAndWeightDemographics:
        demographicDim = 4
    else:
        demographicDim = 2


    # Instantiate the model
    if modelType == "Baseline":
        model = Baseline(inputFeatureDim=inputDim, demographicFeatureDim=demographicDim, dimModel=512,
                         hiddenDim=hiddenDim)
    elif modelType == "RegressionTransformer":
        model = RegressionTransformer(dimModel=512,
                                      inputFeatureDim=inputDim,
                                      demographicFeatureDim=demographicDim,
                                      dropout=dropout,
                                      nHeads=nHeads,
                                      hiddenDim=hiddenDim)
    else:
        raise NotImplementedError("We only support Baseline and RegressionTransformer as model types currently.")

    trainedModel = torch.load(trainedModelPath + 'best_model_checkpoint.pth', map_location=device)
    model.load_state_dict(trainedModel['model_state_dict'])
    model.eval()
    model.to(device)
    print("We have successfully loaded the pretrained model...", trainedModelPath + 'best_model_checkpoint.pth')




    if useScaler:
        SBPscaler = joblib.load(modelType + 'SBPscaler.save')
        DBPscaler = joblib.load(modelType + 'DBPscaler.save')


    testingResults = pd.DataFrame(columns=["SBP GT", "DBP GT", "SBP Pred", "DBP Pred", "SBP Diff", "DBP Diff"])

    with torch.no_grad():
        for batch in tqdm.tqdm(testDataloader):
            inputFeatures, demographics, featuresToPredict, targets = batch

            featuresToPredict = featuresToPredict.reshape(
                [featuresToPredict.shape[0], 1, featuresToPredict.shape[1]])

            sbpGT = targets[:, 0]
            dbpGT = targets[:, 1]


            # Forward pass
            if modelType == "Baseline":
                sbpPred, dbpPred = model(src=featuresToPredict,
                                         demographic_features=demographics)
            elif modelType == "RegressionTransformer":
                sbpPred, dbpPred = model(src=inputFeatures,
                                         tgt=featuresToPredict,
                                         demographic_features=demographics)
            else:
                raise NotImplementedError(
                    "We only support Baseline and RegressionTransformer as model types currently.")

            sbpPred, dbpPred = sbpPred[:, 0], dbpPred[:, 0]



            # Convert to original scale
            sbpGT_OriginalScale = SBPscaler.inverse_transform(sbpGT.reshape(-1, 1).cpu().numpy())
            dbpGT_OriginalScale = DBPscaler.inverse_transform(dbpGT.reshape(-1, 1).cpu().numpy())
            sbpPred_OriginalScale = SBPscaler.inverse_transform(sbpPred.reshape(-1, 1).detach().cpu().numpy())
            dbpPred_OriginalScale = DBPscaler.inverse_transform(dbpPred.reshape(-1, 1).detach().cpu().numpy())

            sbpGT_NanMask = np.isnan(sbpGT_OriginalScale)
            dbpGT_NanMask = np.isnan(dbpGT_OriginalScale)
            sbpPred_NanMask = np.isnan(sbpPred_OriginalScale)
            dbpPred_NanMask = np.isnan(dbpPred_OriginalScale)

            joint_nan_mask = sbpGT_NanMask | dbpGT_NanMask | sbpPred_NanMask | dbpPred_NanMask

            sbpGT_OriginalScale = np.round(sbpGT_OriginalScale[~joint_nan_mask][0])
            dbpGT_OriginalScale = np.round(dbpGT_OriginalScale[~joint_nan_mask][0])
            sbpPred_OriginalScale = np.round(sbpPred_OriginalScale[~joint_nan_mask][0])
            dbpPred_OriginalScale = np.round(dbpPred_OriginalScale[~joint_nan_mask][0])


            currentResults = pd.DataFrame([{"SBP GT": sbpGT_OriginalScale,
                                             "DBP GT": dbpGT_OriginalScale,
                                             "SBP Pred": sbpPred_OriginalScale,
                                             "DBP Pred": dbpPred_OriginalScale,
                                             "SBP Diff": (sbpPred_OriginalScale - sbpGT_OriginalScale),
                                             "DBP Diff": (dbpPred_OriginalScale - dbpGT_OriginalScale)}])

            if not os.path.exists(trainedModelPath + "testing_results.csv"):
                testingResults.to_csv(trainedModelPath + "testing_results.csv", index=False, mode="a",
                                      header=not os.path.isfile(trainedModelPath + "testing_results.csv"))
            else:
                currentResults.to_csv(trainedModelPath + "testing_results.csv", index=False, mode="a",
                                  header=not os.path.isfile(trainedModelPath + "testing_results.csv"))



       
    # examplary result.
    # for more results check analyseResults.py with the testing_results.csv that is stored in your modelpath
    sbpError = testingResults["SBP Diff"]

    sbpMe = np.mean(sbpError)
    sbpMae = np.mean(np.abs(sbpError))

    print(sbpMe, sbpMae)
