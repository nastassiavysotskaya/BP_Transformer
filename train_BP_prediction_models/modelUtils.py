import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataloader import MultisourceTimeSeriesDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time
import os


def calc_regression_metrics_BP(gt_SBP, gt_DBP, pred_SBP, pred_DBP, regression_metric):
    # print("gt_SBP:shpae", gt_SBP.shape)
    # print("Let's compare to the first sample:", gt_SBP[0], pred_SBP[0], gt_SBP[0] - pred_SBP[0])
    if regression_metric == 'mae':
        SBP_metric = mean_absolute_error(gt_SBP, pred_SBP)
        DBP_metric = mean_absolute_error(gt_DBP, pred_DBP)

    elif regression_metric == 'rmse':
        SBP_metric = torch.sqrt(torch.tensor(mean_squared_error(gt_SBP, pred_SBP)))
        DBP_metric = torch.sqrt(torch.tensor(mean_squared_error(gt_DBP, pred_DBP)))

    elif regression_metric == 'r2':
        SBP_metric = r2_score(gt_SBP, pred_SBP)
        DBP_metric = r2_score(gt_DBP, pred_DBP)

    metric = SBP_metric + DBP_metric

    return metric


def ABP_return_train_val_dataloader(device, filename, batch_size, sequence_length, use_scaler, use_height_and_weight,
                                    shuffle, model_type):
    pathToExtractedTrainDF = filename[:-4] + "_trainDF.pkl"
    pathToExtractedValDF = filename[:-4] + "_valDF.pkl"
    print("Checking if the train and val dataframes have been extracted yet for %s..." % (filename))

    if os.path.exists(pathToExtractedTrainDF):
        print("Train/val dataframes have been created before! \nLoading them now...")
        tic = time.time()
        with open(pathToExtractedTrainDF, 'rb') as trainfile:
            train_df = pickle.load(trainfile)
        with open(pathToExtractedValDF, 'rb') as valfile:
            val_df = pickle.load(valfile)
        print("Loading took ", time.time() - tic, " seconds.")
    else:
        print("Train/val dataframes have not been created before! \nStart extracting...")
        tic = time.time()
        # READ IN TRAINING DATASET
        df_total = pd.read_csv(filename)
        print("In total, we have %d training samples." % len(df_total))
        df_X = df_total.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
        df_y = df_total[['Ground Truth SBP', 'Ground Truth DBP']]

        train_features, train_target, val_features, val_target = [], [], [], []

        for subject_id, group in df_total.groupby('Subject'):
            # print("For subject ", subject_id, " we have ", len(group), " entries.")
            df_X_forSubject = group.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
            df_y_forSubject = group[['Ground Truth SBP', 'Ground Truth DBP']]

            n_val = max(1, int(len(group) * 0.2))
            n_train = len(group) - n_val

            subject_train_features = df_X_forSubject[:n_train]
            subject_train_target = df_y_forSubject[:n_train]

            subject_val_features = df_X_forSubject[n_train:]
            subject_val_target = df_y_forSubject[n_train:]

            print("Len(train/val): (%d/%d)" % (len(subject_train_features), len(subject_val_features)))
            assert (len(subject_train_features) + len(subject_val_features) == len(group))

            train_features.append(subject_train_features)
            train_target.append(subject_train_target)
            val_features.append(subject_val_features)
            val_target.append(subject_val_target)

            # if train_features is None:

            #     train_features, train_target= subject_train_features, subject_train_target
            #     val_features, val_target = subject_val_features, subject_val_target
            # else:
            #     train_features = pd.concat([train_features, subject_train_features], ignore_index=True)
            #     train_target = pd.concat([train_target, subject_train_target], ignore_index=True)

            #     val_features = pd.concat([val_features, subject_val_features], ignore_index=True)
            #     val_target = pd.concat([val_target, subject_val_target], ignore_index=True)

        train_features_DF = pd.concat(train_features, ignore_index=True)
        train_target_DF = pd.concat(train_target, ignore_index=True)
        val_features_DF = pd.concat(val_features, ignore_index=True)
        val_target_DF = pd.concat(val_target, ignore_index=True)
        # print("Total df length: ", len(df_total))
        print("Len(train/val): (%d/%d)" % (len(train_features_DF), len(val_features_DF)))
        assert (len(train_features_DF) + len(val_features_DF) == len(df_total))

        # train_features, val_features, train_target, val_target = train_test_split(df_X, df_y, test_size=0.2,
        #                                                                          random_state=42)  # Adjust test_size and random_state as needed

        # Create train and validation DataFrames
        train_df = pd.concat([train_features_DF, train_target_DF], axis=1)
        val_df = pd.concat([val_features_DF, val_target_DF], axis=1)

        with open(pathToExtractedTrainDF, 'wb') as trainDfFile:
            pickle.dump(train_df, trainDfFile)
        with open(pathToExtractedValDF, 'wb') as valDfFile:
            pickle.dump(val_df, valDfFile)
        print("Extracting took ", time.time() - tic, " seconds.")

    # Create a MultisourceTimeSeriesDataset
    train_dataset = MultisourceTimeSeriesDataset(device, train_df, sequence_length, use_scaler=use_scaler,
                                                 use_height_and_weight=use_height_and_weight, model_type=model_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # Create a Validation DataLoader from MultisourceTimeSeriesDataset
    val_dataset = MultisourceTimeSeriesDataset(device, val_df, sequence_length, use_scaler=use_scaler,
                                               use_height_and_weight=use_height_and_weight, model_type=model_type)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return train_dataloader, val_dataloader


def radar_return_train_val_test_dataloader(device, featureset_file, batch_size, sequence_length, use_scaler,
                                           use_height_and_weight, shuffle, model_type):
    # READ IN TRAINING DATASET
    df_total = pd.read_csv(featureset_file)
    df_X = df_total.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
    df_y = df_total[['Ground Truth SBP', 'Ground Truth DBP']]

    train_features, train_target, val_features, val_target, test_features, test_target = None, None, None, None, None, None

    for subject_id, group in df_total.groupby('Subject'):
        # print("For subject ", subject_id, " we have ", len(group), " entries.")
        df_X_forSubject = group.drop(columns=['Ground Truth SBP', 'Ground Truth DBP'])
        df_y_forSubject = group[['Ground Truth SBP', 'Ground Truth DBP']]

        n_train = int(len(group) * 0.8)
        n_val = max(1, int(n_train * 0.15))

        subject_train_features = df_X_forSubject[:n_train - n_val]
        subject_train_target = df_y_forSubject[:n_train - n_val]

        subject_val_features = df_X_forSubject[n_train - n_val:n_train]
        subject_val_target = df_y_forSubject[n_train - n_val:n_train]

        subject_test_features = df_X_forSubject[n_train:]
        subject_test_target = df_y_forSubject[n_train:]

        # print("Len(train/val/test): (%d/%d/%d)" %(len(subject_train_features), len(subject_val_features), len(subject_test_features)))
        assert (len(subject_train_features) + len(subject_val_features) + len(subject_test_features) == len(group))

        if train_features is None:
            train_features, train_target = subject_train_features, subject_train_target
            val_features, val_target = subject_val_features, subject_val_target
            test_features, test_target = subject_test_features, subject_test_target
        else:
            train_features = pd.concat([train_features, subject_train_features], ignore_index=True)
            train_target = pd.concat([train_target, subject_train_target], ignore_index=True)

            val_features = pd.concat([val_features, subject_val_features], ignore_index=True)
            val_target = pd.concat([val_target, subject_val_target], ignore_index=True)

            test_features = pd.concat([test_features, subject_test_features], ignore_index=True)
            test_target = pd.concat([test_target, subject_test_target], ignore_index=True)

    # print("Total df length: ", len(df_total))
    # print("Len(train/val/test): (%d/%d/%d)" %(len(train_features), len(val_features), len(test_features)))
    assert (len(train_features) + len(val_features) + len(test_features) == len(df_total))

    # train_features, test_features, train_target, test_target = train_test_split(df_X,
    #                                                                             df_y,
    #                                                                             test_size=0.2,
    #                                                                             random_state=42)

    # train_features, val_features, train_target, val_target = train_test_split(train_features,
    #                                                                           train_target,
    #                                                                           test_size=0.15,
    #                                                                           random_state=42)

    # Create train and validation DataFrames
    train_df = pd.concat([train_features, train_target], axis=1)
    val_df = pd.concat([val_features, val_target], axis=1)
    test_df = pd.concat([test_features, test_target], axis=1)

    # Create a Train DataLoader from MultisourceTimeSeriesDataset
    train_dataset = MultisourceTimeSeriesDataset(device, train_df, sequence_length, use_scaler=use_scaler,
                                                 use_height_and_weight=use_height_and_weight, model_type=model_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # print("in modelutils: len(df, dataset, dataloader)", len(train_df), len(train_dataset), len(train_dataloader))

    # # Create a Validation DataLoader from MultisourceTimeSeriesDataset
    val_dataset = MultisourceTimeSeriesDataset(device, val_df, sequence_length, use_scaler=use_scaler,
                                               use_height_and_weight=use_height_and_weight, model_type=model_type)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create a Test DataLoader from MultisourceTimeSeriesDataset
    test_dataset = MultisourceTimeSeriesDataset(device, test_df, sequence_length, use_scaler=use_scaler,
                                                use_height_and_weight=use_height_and_weight, model_type=model_type)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, test_dataloader


def returnShapDataloader(device, featureset_file, batch_size, sequence_length, use_scaler, use_height_and_weight,
                         shuffle, model_type):
    # READ IN TRAINING DATASET
    df_total = pd.read_csv(featureset_file)

    totalDataset = MultisourceTimeSeriesDataset(device, df_total, sequence_length, use_scaler=use_scaler,
                                                use_height_and_weight=use_height_and_weight, model_type=model_type)
    totalDataloader = DataLoader(totalDataset, batch_size=batch_size, shuffle=shuffle)

    return totalDataloader


# copied from attention-is-all-you-need-pytorch/transformer/Optim.py by jadore801120
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=2.0, n_steps=0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = n_steps

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
