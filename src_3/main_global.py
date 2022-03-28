import wandb
from data_loaders import make_train_data, make_test_val_data, make_model
from train import train, test, precision, recall, f1_score
import torch
import time
import gc
import config_file
from clean_data_seq2point import load_all_houses_with_device
import random
import optuna
from data_loaders import PecanStreetDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lstm_seq2point import LSTM
from clean_data_seq2point import normalize_y
from sklearn import metrics
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.transforms

wandb.login()

print(torch.cuda.is_available())

gc.collect()
torch.cuda.empty_cache()

print(torch.__version__)

config_ = config_file.load_hyperparameters("refrigerator1")

homes = load_all_houses_with_device(config_file.path, config_['appliance'])


def model_pipeline(hyperparameters, train_months, test_month, appliance, window_length, train_buildings,
                   test_buildings, patience):
    with wandb.init(project="global_models_march25", config=hyperparameters):
        wandb.run.name = str(config_['appliance']) + "_Test:" + str(test_buildings) + "_Train:" + str(train_buildings)

        config = wandb.config

        # lengths = [85320, 132480, 132480, 132480, 132480, 132480, 132480]

        model, criterion, optimizer = make_model(config)

        print(model)
        print("Window Length: ", window_length)

        wandb.watch(model, criterion, log="all", log_freq=10)

        example_ct = 0
        batch_ct = 0
        all_epochs = 0

        # Scheduler for training on single building
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1, verbose=True)

        # #base_lr: 0.001*lr, max_lr: 4*lr, step_size_up:50, step_size_down:2000
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr = 0.99999999999999999999999*config_['learning_rate'], #0.01
        #     max_lr = config_['learning_rate'], #1
        #     step_size_up = 1, #200
        #     step_size_down = 1, #1000
        #     gamma = 1, #1
        #     cycle_momentum=False,
        #     verbose=False
        # )

        validation_loader, test_loader, test_val_seq_std, test_val_seq_mean = make_test_val_data(
            config,
            test_month,
            appliance,
            window_length,
            test_buildings
        )

        time_log = time.time()
        train_loader, train_seq_std, train_seq_mean = make_train_data(
            config,
            train_months,
            appliance,
            window_length,
            train_buildings
        )

        model, example_ct, batch_ct, all_epochs, best_model, train_results = train(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            config,
            example_ct,
            batch_ct,
            all_epochs,
            scheduler,
            test_val_seq_std,
            test_val_seq_mean,
            train_seq_std,
            train_seq_mean,
            patience
        )

        print("Time to train on one home: ", time.time() - time_log)

        test_results = test(best_model, test_loader, criterion, test_val_seq_std, test_val_seq_mean)

    return model, train_results, test_results, best_model

home_ids = homes.dataid.unique()

home_ids_train = [x for x in home_ids if x!=3383]

PATH = "/home/Alfredo/private_nilm/models_power_ratio_filter_global"

final_test_results = {}
final_train_results = {}
random.seed(3)
train_homes = []
best_models = []
max_patience = 200
min_patience = 50
#training_homes = [3383]
#
for i in range(1):
    gc.collect()
    torch.cuda.empty_cache()
    #training_homes.append(i)
    #training_homes = train_homes_from_fl[-1]
    training_homes = home_ids_train
    testing_homes = [3383]
    #patience = int((max_patience-min_patience)/(1-random_select[-1])*len(training_homes)+max_patience+(max_patience-min_patience)/(1-random_select[-1]))
    patience = 50
    print("patience: ", patience)
    print("training_home: ", training_homes)
    print("test_home: ", testing_homes)
    model, train_results, test_results, best_model = model_pipeline(
    config_,
    'sept_oct_nov',
    'sept_oct_nov',
    config_['appliance'],
    config_['window_size'],
    training_homes,
    testing_homes,
    patience)
    test_results = {str(config_["appliance"])+"_Train_home_"+str(training_homes)+"_Test_home_"+str(testing_homes)+"_test_results": test_results}
    train_results = {str(config_["appliance"]) + "_Train_home_" + str(training_homes) + "_Test_home_" + str(
        testing_homes) + "_train_results": train_results}
    final_test_results.update(test_results)
    final_train_results.update(train_results)
    print(final_train_results)
    print(final_test_results)
    #model.cpu()
    #torch.save(model.state_dict(), PATH+"\\refrigerator_model_total_houses_"+str(random_select[i])+"_trial_3.pth")
    best_model.cpu()
    model.cpu()
    best_models.append(best_model)
    torch.save(best_model.state_dict(), PATH+"seq2point_global_refrigerator_model_"+str(testing_homes)+"test_train"+str(training_homes)+"_trial2.pth")