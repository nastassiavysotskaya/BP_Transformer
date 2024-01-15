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
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from modelUtils import *
import torch.nn as nn
import numpy as np
import datetime
import argparse
import joblib
import tqdm
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='data/ABP/', type=str)
    parser.add_argument("--shuffle", default=True, type=bool,action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_scaler", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_height_and_weight_demographics", default=True, type=bool,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_type", default="RegressionTransformer",
                        type=str,  choices=["Baseline", "RegressionTransformer"])

    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--max_patience", default=10, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)

    args = parser.parse_args()
    data_path= args.data_path

    # PARSE BOOLEAN INPUTS FOR DATALOADER
    shuffle = args.shuffle
    useScaler = args.use_scaler
    useHeightAndWeightDemographics = args.use_height_and_weight_demographics

    # PARSE OTHER DATALOADER HYPERPARAMETERS
    batchSize = args.batch_size
    sequenceLength = args.sequence_length
    modelType = args.model_type
    if modelType== "Baseline":
        assert sequenceLength == 1, "The baseline predicts on a single curve, so the sequence length must be 1 too."

    # PARSE MODEL HYPERPARAMETERS
    dropout = args.dropout
    nHeads = args.num_heads
    nEpochs = args.num_epochs
    hiddenDim = args.hidden_dim
    maxPatience = args.max_patience
    learningRate = args.learning_rate

    # create model name to distinguish between different models
    modelName = "Pretrained_" + modelType + "_"
    currentDatetime = datetime.datetime.now()
    formattedDatetime = currentDatetime.strftime("%Y-%m-%d %H-%M")
    modelName = modelName + formattedDatetime

    modelDir = "train_BP_prediction_models/" + modelType + "/Pretrained Models/" + modelName
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)


    trainingConfig = {}
    trainingConfig[
        'demographics'] = 'age, gender, height, weight' if useHeightAndWeightDemographics else 'age, gender'
    trainingConfig['number_of_attention_heads'] = nHeads
    trainingConfig['batch_size'] = batchSize
    trainingConfig['sequence_length'] = sequenceLength
    trainingConfig['num_epochs'] = nEpochs
    trainingConfig['learning_rate'] = learningRate
    trainingConfig['max_patience'] = maxPatience

    with open(str(modelDir + '/modelconfig.json'), 'w') as outfile:
        json.dump(trainingConfig, outfile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # create data loaders
    trainDataloader, valDataloader = abpReturnTrainAndValDataloader(device=device,
                                                                    filename=data_path + "Train_Subset.csv",
                                                                    batchSize=batchSize,
                                                                    sequenceLength=sequenceLength,
                                                                    useScaler=useScaler,
                                                                    useHeightAndWeight=useHeightAndWeightDemographics,
                                                                    shuffle=shuffle,
                                                                    modelType=modelType)


    # input feature dimensions
    inputDim = 21
    if useHeightAndWeightDemographics:
        demographic_dim = 4
    else:
        demographic_dim = 2

    # Instantiate the model
    if modelType == "Baseline":
        model = Baseline(inputFeatureDim=inputDim, demographicFeatureDim=demographic_dim, dimModel=512,
                         hiddenDim=hiddenDim)
    elif modelType == "RegressionTransformer":
        model = RegressionTransformer(dimModel=512,
                                      inputFeatureDim=inputDim,
                                      demographicFeatureDim=demographic_dim,
                                      dropout=dropout,
                                      nHeads=nHeads,
                                      hiddenDim=hiddenDim)
    else:
        raise NotImplementedError("We only support Baseline and RegressionTransformer as model types currently.")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    sched = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=4000)

    trainLossHistory, trainRmseHistory, trainMaeHistory, trainR2History = [], [], [], []
    valLossHistory, valRmseHistory, valMaeHistory, valR2History = [], [], [], []

    bestValLoss = float('inf')  # Initialize with a high value
    patience = maxPatience  # Number of epochs to wait for validation loss improvement
    patienceCounter = 0  # Counter for patience

    if useScaler:
        sbpScaler = joblib.load("train_BP_prediction_models/" + modelType + '/SBPscaler.save')
        dbpScaler = joblib.load("train_BP_prediction_models/" + modelType + '/DBPscaler.save')

    writer = SummaryWriter(modelDir + "/logs/")
    history = pd.DataFrame(
        columns=["Epoch", "Train_Loss", "Train_RMSE", "Train_MAE", "Train_R2", "Val_Loss", "Val_RMSE", "Val_MAE",
                 "Val_R2"])

    for epoch in tqdm.tqdm(range(nEpochs), position=0, leave=True):
        # Training loop
        # print("The current epoch is....", epoch)
        model.train()

        # Initialize metrics for training set
        trainMae = 0.0
        trainRmse = 0.0
        trainR2 = 0.0
        trainLoss = 0.0

        for batch in tqdm.tqdm(trainDataloader, position=0, leave=True):
            sched.zero_grad()

            inputFeatures, demographics, featuresToPredict, targets = batch
            featuresToPredict = featuresToPredict.reshape(
                [featuresToPredict.shape[0], 1, featuresToPredict.shape[1]])

            sbpGT = targets[:, 0]
            dbpGT = targets[:, 1]

            # Forward pass
            if modelType == "Baseline":
                systolePred, diastolePred = model(src=featuresToPredict,
                                                  demographic_features=demographics)
            elif modelType == "RegressionTransformer":
                systolePred, diastolePred = model(src=inputFeatures,
                                                  tgt=featuresToPredict,
                                                  demographic_features=demographics)
            else:
                raise NotImplementedError(
                    "We only support Baseline and RegressionTransformer as model types currently.")
            systolePred, diastolePred = systolePred[:, 0], diastolePred[:, 0]
            # Compute the loss for systole and diastole separately
            lossSystole = torch.sqrt(criterion(systolePred, sbpGT))
            lossDiastole = torch.sqrt(criterion(diastolePred, dbpGT))

            # Total loss is the sum of both losses
            loss = lossSystole + lossDiastole

            # Backpropagation
            loss.backward()
            sched.step_and_update_lr()
            trainLoss += loss.item()

            # print("In the Batch loop... The loss is now: ", loss.item())

            sbpGT_OriginalScale = sbpScaler.inverse_transform(sbpGT.reshape(-1, 1).cpu().numpy())
            dbpGT_OriginalScale = dbpScaler.inverse_transform(dbpGT.reshape(-1, 1).cpu().numpy())
            sbpPred_OriginalScale = sbpScaler.inverse_transform(systolePred.reshape(-1, 1).detach().cpu().numpy())
            dbpPred_OriginalScale = dbpScaler.inverse_transform(
                diastolePred.reshape(-1, 1).detach().cpu().numpy())

            sbpGT_NanMask = np.isnan(sbpGT_OriginalScale)
            dbpGT_NanMask = np.isnan(dbpGT_OriginalScale)
            sbpPred_NanMask = np.isnan(sbpPred_OriginalScale)
            dbpPred_NanMask = np.isnan(dbpPred_OriginalScale)

            joint_nan_mask = sbpGT_NanMask | dbpGT_NanMask | sbpPred_NanMask | dbpPred_NanMask

            sbpGT_OriginalScale = sbpGT_OriginalScale[~joint_nan_mask]
            dbpGT_OriginalScale = dbpGT_OriginalScale[~joint_nan_mask]
            sbpPred_OriginalScale = sbpPred_OriginalScale[~joint_nan_mask]
            dbpPred_OriginalScale = dbpPred_OriginalScale[~joint_nan_mask]

            # Calculate metrics for this batch
            batchMae = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                               sbpPredicted=sbpPred_OriginalScale,
                                               dbpPredicted=dbpPred_OriginalScale, regressionMetric='mae')
            batchRmse = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                                sbpPredicted=sbpPred_OriginalScale,
                                                dbpPredicted=dbpPred_OriginalScale, regressionMetric='rmse')
            batchR2 = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                              sbpPredicted=sbpPred_OriginalScale,
                                              dbpPredicted=dbpPred_OriginalScale, regressionMetric='r2')

            trainMae += batchMae
            trainRmse += batchRmse
            trainR2 += batchR2

        avgTrainLoss = trainLoss / len(trainDataloader)
        avgTrainMae = trainMae / len(trainDataloader)
        avgTrainRmse = trainRmse / len(trainDataloader)
        avgTrainR2 = trainR2 / len(trainDataloader)

        trainLossHistory.append(avgTrainLoss)
        trainMaeHistory.append(avgTrainMae)
        trainRmseHistory.append(avgTrainRmse)
        trainR2History.append(avgTrainR2)

        # Validation phase
        model.eval()
        valLoss = 0.0
        valMae = 0.0
        valRmse = 0.0
        valR2 = 0.0

        with torch.no_grad():
            for batch in valDataloader:
                inputFeatures, demographics, featuresToPredict, targets = batch
                featuresToPredict = featuresToPredict.reshape(
                    [featuresToPredict.shape[0], 1, featuresToPredict.shape[1]])

                sbpGT = targets[:, 0]
                dbpGT = targets[:, 1]

                # Forward pass
                if modelType == "Baseline":
                    systolePred, diastolePred = model(src=featuresToPredict,
                                                      demographic_features=demographics)
                elif modelType == "RegressionTransformer":
                    systolePred, diastolePred = model(src=inputFeatures,
                                                      tgt=featuresToPredict,
                                                      demographic_features=demographics)
                else:
                    raise NotImplementedError(
                        "We only support Baseline and RegressionTransformer as model types currently.")
                systolePred, diastolePred = systolePred[:, 0], diastolePred[:, 0]

                # Compute the loss for systole and diastole separately
                lossSystole = torch.sqrt(criterion(systolePred, sbpGT))
                lossDiastole = torch.sqrt(criterion(diastolePred, dbpGT))
                # Total loss is the sum of both losses
                loss = lossSystole + lossDiastole

                valLoss += loss.item()

                # Convert to original scale
                sbpGT_OriginalScale = sbpScaler.inverse_transform(sbpGT.reshape(-1, 1).cpu().numpy())
                dbpGT_OriginalScale = dbpScaler.inverse_transform(dbpGT.reshape(-1, 1).cpu().numpy())
                sbpPred_OriginalScale = sbpScaler.inverse_transform(
                    systolePred.reshape(-1, 1).detach().cpu().numpy())
                dbpPred_OriginalScale = dbpScaler.inverse_transform(
                    diastolePred.reshape(-1, 1).detach().cpu().numpy())

                sbpGT_NanMask = np.isnan(sbpGT_OriginalScale)
                dbpGT_NanMask = np.isnan(dbpGT_OriginalScale)
                sbpPred_NanMask = np.isnan(sbpPred_OriginalScale)
                dbpPred_NanMask = np.isnan(dbpPred_OriginalScale)

                joint_nan_mask = sbpGT_NanMask | dbpGT_NanMask | sbpPred_NanMask | dbpPred_NanMask

                sbpGT_OriginalScale = sbpGT_OriginalScale[~joint_nan_mask]
                dbpGT_OriginalScale = dbpGT_OriginalScale[~joint_nan_mask]
                sbpPred_OriginalScale = sbpPred_OriginalScale[~joint_nan_mask]
                dbpPred_OriginalScale = dbpPred_OriginalScale[~joint_nan_mask]

                # Calculate metrics for this batch
                batchMae = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                                   sbpPredicted=sbpPred_OriginalScale,
                                                   dbpPredicted=dbpPred_OriginalScale, regressionMetric='mae')
                batchRmse = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                                    sbpPredicted=sbpPred_OriginalScale,
                                                    dbpPredicted=dbpPred_OriginalScale, regressionMetric='rmse')
                batchR2 = calcRegressionMetricsBP(sbpGT=sbpGT_OriginalScale, dbpGT=dbpGT_OriginalScale,
                                                  sbpPredicted=sbpPred_OriginalScale,
                                                  dbpPredicted=dbpPred_OriginalScale, regressionMetric='r2')

                valMae += batchMae
                valRmse += batchRmse
                valR2 += batchR2

        avgValLoss = valLoss / len(valDataloader)
        avgValMae = valMae / len(valDataloader)
        avgValRmse = valRmse / len(valDataloader)
        avgValR2 = valR2 / len(valDataloader)

        currentMetrics = pd.DataFrame([{"Epoch": epoch + 1, "Train_Loss": avgTrainLoss, "Train_RMSE": avgTrainRmse,
                                         "Train_MAE": avgTrainMae, "Train_R2": avgTrainR2, "Val_Loss": avgValLoss,
                                         "Val_RMSE": avgValRmse, "Val_MAE": avgValMae, "Val_R2": avgValR2}])
        # history = pd.concat([history, current_metrics], ignore_index=True)

        if not os.path.exists(modelDir + "/metrics.csv"):
            history.to_csv(modelDir + "/metrics.csv", index=False, mode="a",
                           header=not os.path.isfile(modelDir + "/metrics.csv"))
        else:
            currentMetrics.to_csv(modelDir + "/metrics.csv", index=False, mode="a",
                                  header=not os.path.isfile(modelDir + "/metrics.csv"))

        valLossHistory.append(avgValLoss)
        valMaeHistory.append(avgValMae)
        valRmseHistory.append(avgValRmse)
        valR2History.append(avgValR2)

        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train", avgTrainLoss, epoch + 1)
        writer.add_scalar("Loss/Validation", avgValLoss, epoch + 1)

        # Log metrics to TensorBoard
        writer.add_scalar("MAE/Train", avgTrainMae, epoch + 1)
        writer.add_scalar("MAE/Validation", avgValMae, epoch + 1)
        writer.add_scalar("RMSE/Train", avgTrainRmse, epoch + 1)
        writer.add_scalar("RMSE/Validation", avgValRmse, epoch + 1)
        writer.add_scalar("R2/Train", avgTrainR2, epoch + 1)
        writer.add_scalar("R2/Validation", avgValR2, epoch + 1)

        # print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if epoch == 0 or avgValLoss < bestValLoss:
            bestValLoss = avgValLoss
            patienceCounter = 0
            # Save the best model checkpoint

            torch.save(model.state_dict(), modelDir + "/best_model_checkpoint.pth")
        else:
            patienceCounter += 1
            if patienceCounter >= patience:
                print("Early stopping: Validation loss hasn't improved in a while (patience=", patience, ").")
                break

    print("Training finished!")

    # Save the trained model
    torch.save(model.state_dict(), modelDir + "/final_model.pth")

    # Save the training history
    training_history = {
        "train_losses": trainLossHistory,
        "train_RMSEs": trainRmseHistory,
        "train_MAEs": trainMaeHistory,
        "train_R2s": trainR2History,
        "val_losses": valLossHistory,
        "val_RMSEs": valRmseHistory,
        "val_MAEs": valMaeHistory,
        "val_R2s": valR2History
    }
    torch.save(training_history, modelDir + "/training_history.pth")

    # Close the SummaryWriter for the TensorBoard
    writer.close()