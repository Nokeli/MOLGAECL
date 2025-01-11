import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import shutil
import sys
import copy
import time
import random
import yaml
import csv
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn.metrics import roc_curve, precision_recall_curve
import sklearn.metrics as m

from dataset.dataset_test_classify_drugbank import MolTestDatasetWrapper, MolTestDataset

# 模型在DDI任务上微调，针对特定DDI任务
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class classification_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(classification_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, 128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(128, num_classes),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  ])

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)

        for layer in self.fc2:
            out = layer(out)
        # out = self.fc2(out)
        return out


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        # self.device = self._get_device()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        dir_name = current_time + '_' + config['task_name']
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss().cuda()
            # self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data1, data2, ddi_label, n_iter):
        # get the prediction
        feature, pred = model(data1, data2)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            ddi_label = torch.from_numpy(np.array(ddi_label)).long().to(self.device)
            loss = self.criterion(pred, ddi_label)
            # print("train_batch loss:", loss)
            # outputs = classification_model(ddi_feature_train)
            # loss = self.criterion(outputs, ddi_label_train)
        # elif self.config['dataset']['task'] == 'regression':
        #     if self.normalizer:
        #         loss = self.criterion(pred, self.normalizer.norm(data.y))
        #     else:
        #         loss = self.criterion(pred, data.y)

        return pred, loss

    def train(self, save_path, num_epochs):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        ## drug pretrain feature
        # drug_feature = []
        # smiles_smile = []
        # ddi_dict = {}
        # num = 1
        # smiles_data = pd.read_csv("data/drugbank/drug_smiles_list.csv")
        # smiles_index = smiles_data['smiles'].tolist()
        # # lennn = len(all_ddi_data)

        ## ddi data
        if config['task_name'] == 'drugbank':
            ddi_data = pd.read_csv("data/drugbank/classify_finetune/filtered_ddi_smiles_label.csv", header=None,
                                   index_col=False)
        elif config['task_name'] == 'ogbl':
            ddi_data = pd.read_csv("data/ogbl/classify_finetune/filtered_ddi_smiles_label.csv", header=None,
                                   index_col=False)
        elif config['task_name'] == 'ZhangDDI':
            ddi_data = pd.read_csv("data/ZhangDDI/classify_finetune/filtered_ddi_smiles_label.csv", header=None,
                                   index_col=False)

        ddi_len = len(ddi_data.iloc[:, 0])
        kf_index = np.arange(0, ddi_len)
        random.shuffle(kf_index)
        # drug1_smile = ddi_data.iloc[:, 1]
        # drug2_smile = ddi_data.iloc[:, 2]

        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        train_index = []
        valid_index = []
        test_index = []
        for train_idx, test_idx in kf.split(kf_index):
            random.shuffle(train_idx)
            train_index.append(train_idx)

            random.shuffle(test_idx)
            valid_index.append(test_idx[0:int(0.5 * len(test_idx))])
            test_index.append(test_idx[int(0.5 * len(test_idx)):])

        acc_result = []
        auc_result = []
        precision_result = []
        f1_result = []
        recall_result = []
        mcc_result = []
        aupr_result = []

        for i in range(5):
            # load model
            if self.config['model_type'] == 'gin':
                from models.ginet_finetune import GINet
                model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gcn':
                from models.gcn_finetune import GCN
                model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gae':
                # 直接将分类任务放到finetune中，使用GAE_finetune2
                from models.GAE_finetune2 import GraphAutoencoder
                # from models.GAE_finetune import GraphAutoencoder
                model = GraphAutoencoder(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
                print(model)

            layer_list = []
            for name, param in model.named_parameters():
                # print("model.named_parameters:", model.named_parameters())
                if 'pred_head' in name:
                    print(name, param.requires_grad)
                    layer_list.append(name)

            params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
            base_params = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

            optimizer = torch.optim.Adam(
                [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
                self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
            )

            if apex_support and self.config['fp16_precision']:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
                )

            # data processing

            # 提取train药物smile和ddi label
            drug_one_smile_list = ddi_data.iloc[:, 1].tolist()
            drug_one_smile = [drug_one_smile_list[index] for index in train_index[i]]
            drug_two_smile_list = ddi_data.iloc[:, 2].tolist()
            drug_two_smile = [drug_two_smile_list[index] for index in train_index[i]]
            label_list = ddi_data.iloc[:, 3].tolist()
            label = [label_list[index] for index in train_index[i]]
            drug_data1 = MolTestDataset(drug_one_smile)
            drug_data2 = MolTestDataset(drug_two_smile)

            data1_loader = DataLoader(drug_data1, batch_size=256, num_workers=4, drop_last=True)
            data2_loader = DataLoader(drug_data2, batch_size=256, num_workers=4, drop_last=True)
            label_loader = DataLoader(label, batch_size=256, num_workers=4, drop_last=True)

            # 提取valid药物smile和ddi label
            drug_one_smile_valid = [drug_one_smile_list[index] for index in valid_index[i]]
            drug_two_smile_valid = [drug_two_smile_list[index] for index in valid_index[i]]
            ddi_label_valid = [label_list[index] for index in valid_index[i]]
            drug_data1_valid = MolTestDataset(drug_one_smile_valid)
            drug_data2_valid = MolTestDataset(drug_two_smile_valid)

            data1_loader_valid = DataLoader(drug_data1_valid, batch_size=256, num_workers=4, drop_last=True)
            data2_loader_valid = DataLoader(drug_data2_valid, batch_size=256, num_workers=4, drop_last=True)
            label_loader_valid = DataLoader(ddi_label_valid, batch_size=256, num_workers=4, drop_last=True)

            # 提取test药物smile和ddi label
            drug_one_smile_test = [drug_one_smile_list[index] for index in test_index[i]]
            drug_two_smile_test = [drug_two_smile_list[index] for index in test_index[i]]
            ddi_label_test = [label_list[index] for index in test_index[i]]
            drug_data1_test = MolTestDataset(drug_one_smile_test)
            drug_data2_test = MolTestDataset(drug_two_smile_test)

            data1_loader_test = DataLoader(drug_data1_test, batch_size=256, num_workers=4, drop_last=True)
            data2_loader_test = DataLoader(drug_data2_test, batch_size=256, num_workers=4, drop_last=True)
            label_loader_test = DataLoader(ddi_label_test, batch_size=256, num_workers=4, drop_last=True)

            # save test data
            test_ddi = pd.DataFrame()
            test_ddi['drug1'] = drug_one_smile_test
            test_ddi['drug2'] = drug_two_smile_test
            test_ddi['label'] = ddi_label_test
            test_ddi_outfile = str(save_path) + '/Fold=' + str(i + 1) + ' test_ddi_data' + '.csv'
            test_ddi.to_csv(test_ddi_outfile, index=False)

            best_valid_cls = 0
            min_loss = 200
            best_epoch = 2000

            print("---------------" + f'Fold [{i}]' + '--------------------')
            for epoch_counter in range(num_epochs):
                num = 0
                num_valid = 0
                num_test = 0
                loss_all = 0.0
                loss_all_valid = 0.0
                loss_all_test = 0.0
                # train
                for data1, data2, label_train in zip(data1_loader, data2_loader, label_loader):
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    optimizer.zero_grad()
                    __, loss = self._step(model, data1, data2, label_train, n_iter)
                    num += 1
                    loss.backward()
                    optimizer.step()
                    loss_all += loss.item()

                loss_all = loss_all / num
                print("train loss:", loss_all)

                # valid
                model.eval()
                with torch.no_grad():
                    predictions = []
                    validation_label = []
                    for data1_valid, data2_valid, label_valid in zip(data1_loader_valid, data2_loader_valid,
                                                                     label_loader_valid):
                        data1_valid = data1_valid.to(self.device)
                        data2_valid = data2_valid.to(self.device)
                        pred, valid_loss = self._step(model, data1_valid, data2_valid, label_valid, n_iter)
                        _, valid_predicted = torch.max(pred, 1)

                        predictions.extend(valid_predicted.cpu())
                        validation_label.extend(label_valid)

                        num_valid += 1
                        # print("valid batch loss:", valid_loss.item())
                        loss_all_valid += valid_loss.item()

                    loss_all_valid = loss_all_valid / num_valid
                    # print("valid loss:", loss_all_valid)

                    acc_val = accuracy_score(validation_label, predictions)
                    print(f'valid loss: {loss_all_valid},valid Accuracy: {acc_val}')

                    if loss_all_valid <= min_loss:
                        # if acc_val >= max_auc and valid_loss.item() <= min_loss:
                        model_max = copy.deepcopy(model)
                        min_loss = loss_all_valid
                        best_epoch = epoch_counter
                        print("best model is {}".format(epoch_counter))

                    # 写出结果
                    fold_outfile_name = str(save_path) + '/Fold=' + str(i + 1) + 'valid loss & valid accuracy' + '.txt'
                    with open(fold_outfile_name, 'a') as file:
                        file.write(
                            f'best_epoch:{best_epoch}\n')
                        file.write(
                            f'epoch_counter:{epoch_counter},valid loss: {loss_all_valid},valid Accuracy: {acc_val}\n')

                # switch the model to training mode
                model.train()

            # save best model
            best_model_outfile = str(save_path) + '/Fold=' + str(i + 1) + ' best_model' + '.pth'
            torch.save(model_max.state_dict(), best_model_outfile)

            # test
            model.eval()
            with torch.no_grad():
                test_predictions = []
                test_label = []
                test_scores = []
                for data1_test, data2_test, label_test in zip(data1_loader_test, data2_loader_test,
                                                              label_loader_test):
                    data1_test = data1_test.to(self.device)
                    data2_test = data2_test.to(self.device)
                    test_pred, test_loss = self._step(model_max, data1_test, data2_test, label_test, n_iter)
                    _, test_predicted = torch.max(test_pred, 1)
                    # test_score_true = test_pred[:, 1]  # probability of positive category

                    test_score_all = F.softmax(test_pred)
                    test_score_true = test_score_all[:, 1]  # probability of positive category

                    test_predictions.extend(test_predicted.cpu())
                    test_label.extend(label_test)
                    test_scores.extend(test_score_true.cpu())

                    num_test += 1
                    # print("test batch loss:", valid_loss.item())
                    loss_all_test += test_loss.item()

                loss_all_test = loss_all_test / num_test
                print("test loss:", loss_all_test)

                acc_test = accuracy_score(test_label, test_predictions)
                # auc = roc_auc_score(test_label, test_predictions)
                auc = roc_auc_score(test_label, test_scores)
                precision = precision_score(test_label, test_predictions)
                f1 = f1_score(test_label, test_predictions)
                recall = recall_score(test_label, test_predictions)
                mcc = matthews_corrcoef(test_label, test_predictions)
                fpr, tpr, thr = roc_curve(y_true=test_label, y_score=test_predictions)
                p, r, t = precision_recall_curve(y_true=test_label, probas_pred=test_scores)
                # p, r, t = precision_recall_curve(y_true=test_label, probas_pred=test_predictions)
                aupr = m.auc(r, p)

                print(f'Accuracy: {acc_test}')
                print(f'AUC-ROC: {auc}')
                print(f'Precision: {precision}')
                print(f'F1 Score: {f1}')
                print(f'Recall: {recall}')
                print(f'MCC: {mcc}')
                print(f'AUPR: {aupr}')

            acc_result.append(acc_test)
            auc_result.append(auc)
            precision_result.append(precision)
            f1_result.append(f1)
            recall_result.append(recall)
            mcc_result.append(mcc)
            aupr_result.append(aupr)

            fold_results = {"Accuracy": acc_test,
                            "Auc": auc,
                            "Precision": precision,
                            "Recall": recall,
                            "F1-score": f1,
                            "MCC": mcc,
                            "AUPR": aupr}
            
            fold_outfile_name = str(save_path) + '/Fold=' + str(i + 1) + self.config[
                'task_name'] + ' finetune test metrics' + '.txt'

            with open(fold_outfile_name, 'w') as file:
                for key, value in fold_results.items():
                    file.write(f'"{key}"={value}\n')

        # Calculate average results
        results = {"Accuracy mean, Accuracy std": [np.mean(acc_result), np.std(acc_result)],
                   "Auc mean, Auc std": [np.mean(auc_result), np.std(auc_result)],
                   "Precision mean, Precision std": [np.mean(precision_result), np.std(precision_result)],
                   "Recall mean, Recall std": [np.mean(recall_result), np.std(recall_result)],
                   "F1-score mean, F1-score std": [np.mean(f1_result), np.std(f1_result)],
                   "MCC mean, MCC std": [np.mean(mcc_result), np.std(mcc_result)],
                   "AUPR mean, AUPR std": [np.mean(aupr_result), np.std(aupr_result)],
                   }

        outfile_name = str(save_path) + '/average ' + self.config['task_name'] + ' finetune test metrics' + '.txt'

        with open(outfile_name, 'w') as file:
            for key, value in results.items():
                file.write(f'"{key}"={value}\n')

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                print(model)
                print(pred)

                # # 从模型的参数中提取特定层（在layer_list中指定）的参数，并将它们存储在params列表中
                # layer_list = []
                # for name, param in model.named_parameters():
                #     print(model.named_parameters())
                #     if 'pred_head' in name:
                #         print(name, param.requires_grad)
                #         layer_list.append(name)
                #
                # params = list(
                #     map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))

                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    num_epochs = config['epochs']
    save_path = config['task_name'] + "+" + str(current_time) + "+num_epochs=" + str(num_epochs)
    if os.path.exists(save_path):
        print("error")
    else:
        os.mkdir(save_path)

    fine_tune = FineTune(dataset, config)
    fine_tune.train(save_path=save_path, num_epochs=num_epochs)

    # if config['dataset']['task'] == 'classification':
    #     return fine_tune.roc_auc
    # if config['dataset']['task'] == 'regression':
    #     if config['task_name'] in ['qm7', 'qm8', 'qm9']:
    #         return fine_tune.mae
    #     else:
    #         return fine_tune.rmse


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    # DDI prediction
    if config['task_name'] == 'ogbl':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/ogbl/drug_smiles_mix.txt'
        target_list = []  # 分类标签：无

    elif config['task_name'] == 'drugbank':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/drugbank/drug_smiles_list.csv'
        target_list = []  # 分类标签：无

    elif config['task_name'] == 'ZhangDDI':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/ZhangDDI/drug_list_zhang.csv'
        target_list = []  # 分类标签：无

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    results_list = []
    # for target in target_list:
    #     config['dataset']['target'] = target
    #     result = main(config)
    #     results_list.append([target, result])
    result = main(config)
