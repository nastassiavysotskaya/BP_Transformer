import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os


class MultisourceTimeSeriesDataset(Dataset):
    def __init__(self, device, dataframe, sequence_length, use_scaler=False, use_height_and_weight=True,
                 model_type=None):
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.use_scaler = use_scaler
        self.device = device
        self.use_height_and_weight = use_height_and_weight
        self.data, self.demographics, self.targets = self.preprocess_data(dataframe)
        # self.data, self.demographics, self.targets = self.data.to(device), self.demographics.to(device), self.targets.to(device)
        self.subject_ids = list(self.data.keys())
        # print(len(self.subject_ids),"SubjectIDs in Dataset:", self.subject_ids)

    def preprocess_data(self, dataframeWithNaN):

        # print("Total number of expected samples: ", len(dataframe))
        data_dict = {}
        target_dict = {}
        demographic_dict = {}

        target_columns = ['Ground Truth SBP', 'Ground Truth DBP']
        if self.use_height_and_weight:
            demographic_columns = ['Age', 'Gender', 'Height', 'Weight']
            print("DF len before dropping nan: ", len(dataframeWithNaN))
            dataframe = dataframeWithNaN.dropna()
            print("DF len after dropping nan: ", len(dataframe))
        else:
            demographic_columns = ['Age', 'Gender']
            dataframe = dataframeWithNaN

        if self.use_scaler:
            demographicscaler_path = "train_BP_prediction_models/" + self.model_type + '/demographicscaler.save'
            featurescaler_path = "train_BP_prediction_models/" + self.model_type + '/featurescaler.save'
            SBPscaler_path = "train_BP_prediction_models/" + self.model_type + '/SBPscaler.save'
            DBPscaler_path = "train_BP_prediction_models/" + self.model_type + '/DBPscaler.save'

            if not os.path.exists(featurescaler_path):
                labelfree_data = dataframe.drop(
                    columns=['Subject', 'Ground Truth SBP', 'Ground Truth DBP', 'Length', 'Age', 'Gender', 'Height',
                             'Weight']).values
                featurescaler = MinMaxScaler()
                featurescaler = featurescaler.fit(labelfree_data)
                joblib.dump(featurescaler, featurescaler_path)

            else:
                featurescaler = joblib.load(featurescaler_path)

            if not os.path.exists(SBPscaler_path):
                label_data = dataframe['Ground Truth SBP'].values
                SBPscaler = MinMaxScaler()
                SBPscaler = SBPscaler.fit(label_data.reshape(-1, 1))
                joblib.dump(SBPscaler, SBPscaler_path)

            else:
                SBPscaler = joblib.load(SBPscaler_path)

            if not os.path.exists(DBPscaler_path):
                label_data = dataframe['Ground Truth DBP'].values
                DBPscaler = MinMaxScaler()
                DBPscaler = DBPscaler.fit(label_data.reshape(-1, 1))
                joblib.dump(DBPscaler, DBPscaler_path)

            else:
                DBPscaler = joblib.load(DBPscaler_path)

            if not os.path.exists(demographicscaler_path):
                dem_data = dataframe[demographic_columns].values
                # print(dem_data)
                # (np.min(dem_data, axis=0))
                demographicscaler = MinMaxScaler()
                demographicscaler = demographicscaler.fit(dem_data)
                joblib.dump(demographicscaler, demographicscaler_path)

            else:
                demographicscaler = joblib.load(demographicscaler_path)

        for subject_id, group in dataframe.groupby('Subject'):
            # print("Subject/ len(group): ", subject_id, len(group))
            if (self.use_height_and_weight and not (
            np.isnan(torch.FloatTensor(group[demographic_columns].values)).any())) or not self.use_height_and_weight:
                # print("We include this subject! ", subject_id)

                data = group.drop(
                    columns=['Subject', 'Ground Truth SBP', 'Ground Truth DBP', 'Length', 'Age', 'Gender', 'Height',
                             'Weight']).values
                # target = group[target_columns].values

                # print("Expected target shape...", target)
                SBP = group['Ground Truth SBP'].values
                DBP = group['Ground Truth DBP'].values

                demographic = group[demographic_columns].values

                if self.use_scaler:
                    data = featurescaler.transform(data)
                    scaled_SBP = SBPscaler.transform(SBP.reshape(-1, 1))
                    scaled_DBP = DBPscaler.transform(DBP.reshape(-1, 1))
                    target = np.hstack([scaled_SBP, scaled_DBP])
                    demographic = demographicscaler.transform(demographic)

                # print("Returned target shape...", target)

                target_dict[subject_id] = target  # torch.FloatTensor(target).to(device)

                demographic_dict[subject_id] = demographic  # torch.FloatTensor(demographic).to(device) #.iloc[0]

                data_dict[subject_id] = data  # torch.FloatTensor(data).to(device)

        # print("len of returned dicts: ", pd.DataFrame.from_dict(data_dict).shape, len(demographic_dict), len(target_dict))
        return data_dict, demographic_dict, target_dict

    def __len__(self):
        # print("self.len: ", sum(len(self.data[subject_id]) - self.sequence_length +1 for subject_id in self.subject_ids))
        return sum(len(self.data[subject_id]) - self.sequence_length + 1 for subject_id in self.subject_ids)

    def __getitem__(self, idx):
        # self.__len__

        start_idx = idx
        for subject_ID in self.subject_ids:
            # print(idx, start_idx, len(self.data[subject_ID]))
            if start_idx < len(self.data[subject_ID]) - self.sequence_length + 1:
                # idx belongs to this subject_ID
                # print("idx ", idx, " belongs to subject ", subject_ID, " with ", len(self.data[subject_ID]),
                #      "entries. The current subject-specific index is: ", start_idx)
                # use self.sequence_length -1 inputs for forecasting the self.sequence_lengths values
                data_from_subject = self.data[subject_ID][start_idx:start_idx + self.sequence_length - 1]
                data_from_subject = torch.Tensor(data_from_subject).to(self.device)
                # print(data_from_subject.shape, start_idx, start_idx + self.sequence_length-1)
                # print(data_from_subject)

                demographics_from_subject = self.demographics[subject_ID][start_idx]
                demographics_from_subject = torch.Tensor(demographics_from_subject).to(self.device)
                # print(demographics_from_subject)

                targets_from_subject = self.data[subject_ID][start_idx + self.sequence_length - 1]
                targets_from_subject = torch.Tensor(targets_from_subject).to(self.device)
                # print(targets_from_subject.shape)
                # print(targets_from_subject)

                ground_truth_SBP_DBP = self.targets[subject_ID][start_idx + self.sequence_length - 1]
                ground_truth_SBP_DBP = torch.Tensor(ground_truth_SBP_DBP).to(self.device)
                # print(ground_truth_SBP_DBP)

                return data_from_subject, demographics_from_subject, targets_from_subject, ground_truth_SBP_DBP
            else:
                # idx doesn't belong to this subject, iterate to next subject after adapting the start_idx
                # print("jumping to next subject..")
                start_idx = start_idx - len(self.data[subject_ID]) + self.sequence_length - 1
