from torch.utils.tensorboard import SummaryWriter
from models import RegressionTransformer, Baseline
import torch.optim as optim
from model_utils import *
import torch.nn as nn
import numpy as np
import datetime
import argparse
import joblib
import tqdm
import math
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository_path", default="TODO/", type=str)
    parser.add_argument("--data_path", default='data/ABP/', type=str)
    parser.add_argument("--shuffle", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_scaler", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_height_and_weight_demographics", default=True, type=bool,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_type", default="RegressionTransformer",
                        type=str, choices=["Baseline", "RegressionTransformer"])

    parser.add_argument("--resume_training", default=False, type=bool)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--max_patience", default=10, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)

    args = parser.parse_args()
    repo_path = args.repository_path
    data_path = args.data_path
    checkpointDir = args.checkpoint_dir

    # PARSE BOOLEAN INPUTS FOR DATALOADER
    shuffle = args.shuffle
    print(shuffle)
    use_scaler = args.use_scaler
    use_height_and_weight_demographics = args.use_height_and_weight_demographics
    resume_training = args.resume_training

    # PARSE OTHER DATALOADER HYPERPARAMETERS
    d_model = args.d_model
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    model_type = args.model_type
    if model_type == "Baseline":
        assert sequence_length == 1, "The baseline predicts on a single curve, so the sequence length must be 1 too."

    # PARSE MODEL HYPERPARAMETERS
    dropout = args.dropout
    num_heads = args.num_heads
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    max_patience = args.max_patience
    learning_rate = args.learning_rate

    # create model name to distinguish between different models
    if resume_training:
        modelname = "Pretrained_" + model_type + "_" + "dmodel[" + str(d_model) + "]_nheads[" + str(num_heads) + "]"
    else:
        modelname = "Pretrained_" + model_type + "_" + "dmodel[" + str(d_model) + "]_nheads[" + str(num_heads) + "]_"
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M")
        modelname = modelname + formatted_datetime

    if resume_training:
        model_dir = checkpointDir
    else:
        model_dir = repo_path + "train_BP_prediction_models/" + model_type + "/Pretrained Models/" + modelname
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    training_config = {}
    training_config[
        'demographics'] = 'age, gender, height, weight' if use_height_and_weight_demographics else 'age, gender'
    training_config['number_of_attention_heads'] = num_heads
    training_config['batch_size'] = batch_size
    training_config['sequence_length'] = sequence_length
    training_config['num_epochs'] = num_epochs
    training_config['learning_rate'] = learning_rate
    training_config['max_patience'] = max_patience
    training_config['d_model'] = d_model
    training_config['num_heads'] = num_heads

    with open(str(model_dir + '/modelconfig.json'), 'w') as outfile:
        json.dump(training_config, outfile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # create data loaders
    train_dataloader, val_dataloader = ABP_return_train_val_dataloader(device=device,
                                                                       filename=repo_path + data_path + "Train_Subset.csv",
                                                                       batch_size=batch_size,
                                                                       sequence_length=sequence_length,
                                                                       use_scaler=use_scaler,
                                                                       use_height_and_weight=use_height_and_weight_demographics,
                                                                       shuffle=shuffle,
                                                                       model_type=model_type)

    # input feature dimensions
    input_dim = 21
    if use_height_and_weight_demographics:
        demographic_dim = 4
    else:
        demographic_dim = 2

    # Instantiate the model
    if model_type == "Baseline":
        model = Baseline(input_feature_dim=input_dim, demographic_feature_dim=demographic_dim, d_model=512,
                         hidden_dim=hidden_dim)
    elif model_type == "RegressionTransformer":
        model = RegressionTransformer(d_model=512,
                                      input_feature_dim=input_dim,
                                      demographic_feature_dim=demographic_dim,
                                      dropout=dropout,
                                      num_heads=num_heads,
                                      hidden_dim=hidden_dim)
    else:
        raise NotImplementedError("We only support Baseline and RegressionTransformer as model types currently.")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses_history, train_rmse_history, train_mae_history, train_r2_history = [], [], [], []
    val_losses_history, val_rmse_history, val_mae_history, val_r2_history = [], [], [], []

    best_val_loss = float('inf')  # Initialize with a high value
    patience = max_patience  # Number of epochs to wait for validation loss improvement
    patience_counter = 0  # Counter for patience

    if use_scaler:
        SBPscaler = joblib.load("train_BP_prediction_models/" + model_type + '/SBPscaler.save')
        DBPscaler = joblib.load("train_BP_prediction_models/" + model_type + '/DBPscaler.save')

    writer = SummaryWriter(model_dir + "/logs/")
    history = pd.DataFrame(
        columns=["Epoch", "Train_Loss", "Train_RMSE", "Train_MAE", "Train_R2", "Val_Loss", "Val_RMSE", "Val_MAE",
                 "Val_R2"])

    if resume_training:
        startingEpoch = -1

        checkpointFile = ""
        for file in os.listdir(model_dir):
            # print(file)
            if "checkpoint_epoch" in file:
                # epoch = int(file.split("_")[-1].split(".")[0])
                epoch = int(file.lstrip("checkpoint_epoch")[:-4])
                if epoch > startingEpoch:
                    startingEpoch = epoch + 1
                    checkpointFile = file

        checkpoint = torch.load(model_dir + checkpointFile, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        num_train_batches = len(train_dataloader)
        sched = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=4000,
                               n_steps=num_train_batches * checkpoint['epoch'])
        patience_counter = checkpoint['patience_counter']
        best_val_loss = checkpoint['best_val_loss']
    else:
        startingEpoch = 0
        sched = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=4000)

    for epoch in tqdm.tqdm(range(startingEpoch, num_epochs), position=0, leave=True):
        # Training loop
        model.train()

        # Initialize metrics for training set
        train_mae = 0.0
        train_rmse = 0.0
        train_r2 = 0.0
        train_loss = 0.0

        for batch in tqdm.tqdm(train_dataloader, position=0, leave=True):
            sched.zero_grad()

            input_features, demographics, features_to_predict, targets = batch
            features_to_predict = features_to_predict.reshape(
                [features_to_predict.shape[0], 1, features_to_predict.shape[1]])
            # print(targets.shape)
            # print(input_features.shape, demographics.shape)
            gt_SBP = targets[:, 0]
            gt_DBP = targets[:, 1]

            # Forward pass
            if model_type == "Baseline":
                systole_pred, diastole_pred = model(src=features_to_predict,
                                                    demographic_features=demographics)
            elif model_type == "RegressionTransformer":
                systole_pred, diastole_pred = model(src=input_features,
                                                    tgt=features_to_predict,
                                                    demographic_features=demographics)
            else:
                raise NotImplementedError(
                    "We only support Baseline and RegressionTransformer as model types currently.")
            systole_pred, diastole_pred = systole_pred[:, 0], diastole_pred[:, 0]
            # Compute the loss for systole and diastole separately
            loss_systole = torch.sqrt(criterion(systole_pred, gt_SBP))
            loss_diastole = torch.sqrt(criterion(diastole_pred, gt_DBP))

            # Total loss is the sum of both losses
            loss = loss_systole + loss_diastole

            # Backpropagation
            loss.backward()
            sched.step_and_update_lr()
            train_loss += loss.item()

            # print("In the Batch loop... The loss is now: ", loss.item())

            gt_SBP_originalscale = SBPscaler.inverse_transform(gt_SBP.reshape(-1, 1).cpu().numpy())
            gt_DBP_originalscale = DBPscaler.inverse_transform(gt_DBP.reshape(-1, 1).cpu().numpy())
            systole_pred_originalscale = SBPscaler.inverse_transform(systole_pred.reshape(-1, 1).detach().cpu().numpy())
            diastole_pred_originalscale = DBPscaler.inverse_transform(
                diastole_pred.reshape(-1, 1).detach().cpu().numpy())

            nan_mask_gt_SBP = np.isnan(gt_SBP_originalscale)
            nan_mask_gt_DBP = np.isnan(gt_DBP_originalscale)
            nan_mask_pred_SBP = np.isnan(systole_pred_originalscale)
            nan_mask_pred_DBP = np.isnan(diastole_pred_originalscale)

            joint_nan_mask = nan_mask_gt_SBP | nan_mask_gt_DBP | nan_mask_pred_SBP | nan_mask_pred_DBP

            gt_SBP_originalscale = gt_SBP_originalscale[~joint_nan_mask]
            gt_DBP_originalscale = gt_DBP_originalscale[~joint_nan_mask]
            systole_pred_originalscale = systole_pred_originalscale[~joint_nan_mask]
            diastole_pred_originalscale = diastole_pred_originalscale[~joint_nan_mask]

            # Calculate metrics for this batch
            batch_mae = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                   pred_SBP=systole_pred_originalscale,
                                                   pred_DBP=diastole_pred_originalscale, regression_metric='mae')
            batch_rmse = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                    pred_SBP=systole_pred_originalscale,
                                                    pred_DBP=diastole_pred_originalscale, regression_metric='rmse')
            batch_r2 = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                  pred_SBP=systole_pred_originalscale,
                                                  pred_DBP=diastole_pred_originalscale, regression_metric='r2')

            train_mae += batch_mae
            train_rmse += batch_rmse
            train_r2 += batch_r2

        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_mae = train_mae / len(train_dataloader)
        avg_train_rmse = train_rmse / len(train_dataloader)
        avg_train_r2 = train_r2 / len(train_dataloader)

        train_losses_history.append(avg_train_loss)
        train_mae_history.append(avg_train_mae)
        train_rmse_history.append(avg_train_rmse)
        train_r2_history.append(avg_train_r2)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                input_features, demographics, features_to_predict, targets = batch
                features_to_predict = features_to_predict.reshape(
                    [features_to_predict.shape[0], 1, features_to_predict.shape[1]])

                gt_SBP = targets[:, 0]
                gt_DBP = targets[:, 1]

                # Forward pass
                if model_type == "Baseline":
                    systole_pred, diastole_pred = model(src=features_to_predict,
                                                        demographic_features=demographics)
                elif model_type == "RegressionTransformer":
                    systole_pred, diastole_pred = model(src=input_features,
                                                        tgt=features_to_predict,
                                                        demographic_features=demographics)
                else:
                    raise NotImplementedError(
                        "We only support Baseline and RegressionTransformer as model types currently.")
                systole_pred, diastole_pred = systole_pred[:, 0], diastole_pred[:, 0]

                # Compute the loss for systole and diastole separately
                loss_systole = torch.sqrt(criterion(systole_pred, gt_SBP))
                loss_diastole = torch.sqrt(criterion(diastole_pred, gt_DBP))
                # Total loss is the sum of both losses
                loss = loss_systole + loss_diastole

                val_loss += loss.item()

                # Convert to original scale
                gt_SBP_originalscale = SBPscaler.inverse_transform(gt_SBP.reshape(-1, 1).cpu().numpy())
                gt_DBP_originalscale = DBPscaler.inverse_transform(gt_DBP.reshape(-1, 1).cpu().numpy())
                systole_pred_originalscale = SBPscaler.inverse_transform(
                    systole_pred.reshape(-1, 1).detach().cpu().numpy())
                diastole_pred_originalscale = DBPscaler.inverse_transform(
                    diastole_pred.reshape(-1, 1).detach().cpu().numpy())

                nan_mask_gt_SBP = np.isnan(gt_SBP_originalscale)
                nan_mask_gt_DBP = np.isnan(gt_DBP_originalscale)
                nan_mask_pred_SBP = np.isnan(systole_pred_originalscale)
                nan_mask_pred_DBP = np.isnan(diastole_pred_originalscale)

                joint_nan_mask = nan_mask_gt_SBP | nan_mask_gt_DBP | nan_mask_pred_SBP | nan_mask_pred_DBP

                gt_SBP_originalscale = gt_SBP_originalscale[~joint_nan_mask]
                gt_DBP_originalscale = gt_DBP_originalscale[~joint_nan_mask]
                systole_pred_originalscale = systole_pred_originalscale[~joint_nan_mask]
                diastole_pred_originalscale = diastole_pred_originalscale[~joint_nan_mask]

                # Calculate metrics for this batch
                batch_mae = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                       pred_SBP=systole_pred_originalscale,
                                                       pred_DBP=diastole_pred_originalscale, regression_metric='mae')
                batch_rmse = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                        pred_SBP=systole_pred_originalscale,
                                                        pred_DBP=diastole_pred_originalscale, regression_metric='rmse')
                batch_r2 = calc_regression_metrics_BP(gt_SBP=gt_SBP_originalscale, gt_DBP=gt_DBP_originalscale,
                                                      pred_SBP=systole_pred_originalscale,
                                                      pred_DBP=diastole_pred_originalscale, regression_metric='r2')

                val_mae += batch_mae
                val_rmse += batch_rmse
                val_r2 += batch_r2

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_mae = val_mae / len(val_dataloader)
        avg_val_rmse = val_rmse / len(val_dataloader)
        avg_val_r2 = val_r2 / len(val_dataloader)

        current_metrics = pd.DataFrame([{"Epoch": epoch + 1, "Train_Loss": avg_train_loss, "Train_RMSE": avg_train_rmse,
                                         "Train_MAE": avg_train_mae, "Train_R2": avg_train_r2, "Val_Loss": avg_val_loss,
                                         "Val_RMSE": avg_val_rmse, "Val_MAE": avg_val_mae, "Val_R2": avg_val_r2}])
        # history = pd.concat([history, current_metrics], ignore_index=True)

        if not os.path.exists(model_dir + "/metrics.csv"):
            history.to_csv(model_dir + "/metrics.csv", index=False, mode="a",
                           header=not os.path.isfile(model_dir + "/metrics.csv"))
        else:
            current_metrics.to_csv(model_dir + "/metrics.csv", index=False, mode="a",
                                   header=not os.path.isfile(model_dir + "/metrics.csv"))

        val_losses_history.append(avg_val_loss)
        val_mae_history.append(avg_val_mae)
        val_rmse_history.append(avg_val_rmse)
        val_r2_history.append(avg_val_r2)

        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)

        # Log metrics to TensorBoard
        writer.add_scalar("MAE/Train", avg_train_mae, epoch + 1)
        writer.add_scalar("MAE/Validation", avg_val_mae, epoch + 1)
        writer.add_scalar("RMSE/Train", avg_train_rmse, epoch + 1)
        writer.add_scalar("RMSE/Validation", avg_val_rmse, epoch + 1)
        writer.add_scalar("R2/Train", avg_train_r2, epoch + 1)
        writer.add_scalar("R2/Validation", avg_val_r2, epoch + 1)

        # print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if epoch % 5 == 0:
            # torch.save(model.state_dict(), model_dir + "/checkpoint_epoch"+str(epoch)+".pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'loss': train_loss,
            }, model_dir + "/checkpoint_epoch" + str(epoch) + ".pth")
        # Check for early stopping
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model checkpoint

            # torch.save(model.state_dict(), model_dir + "/best_model_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, model_dir + "/best_model_checkpoint.pth")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping: Validation loss hasn't improved in a while (patience=", patience, ").")
                break

    print("Training finished!")

    # Save the trained model
    torch.save(model.state_dict(), model_dir + "/final_model.pth")

    # Save the training history
    training_history = {
        "train_losses": train_losses_history,
        "train_RMSEs": train_rmse_history,
        "train_MAEs": train_mae_history,
        "train_R2s": train_r2_history,
        "val_losses": val_losses_history,
        "val_RMSEs": val_rmse_history,
        "val_MAEs": val_mae_history,
        "val_R2s": val_r2_history
    }
    torch.save(training_history, model_dir + "/training_history.pth")

    # Close the SummaryWriter for the TensorBoard
    writer.close()