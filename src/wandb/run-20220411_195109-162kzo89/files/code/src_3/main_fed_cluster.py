import wandb
from data_loaders import make_train_data, make_test_val_data, make_model
from train_fed import train, test, precision, recall, f1_score
import torch
import time
import gc
#import lstm
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
import copy

wandb.login()

print(torch.cuda.is_available())

gc.collect()
torch.cuda.empty_cache()

print(torch.__version__)


config_ = config_file.load_hyperparameters("refrigerator1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

homes = load_all_houses_with_device(config_file.path, config_['appliance'])

def client_update(client_model, optimizer, train_loader, epoch=config_['epochs']):
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.MSELoss(output, target)
            loss.backward()
            optimizer.step()
    return float(loss.item())

def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def model_pipeline(hyperparameters, train_months, test_month, appliance, window_length, train_buildings,
                   test_buildings, patience):
    with wandb.init(project="march7_FL_trials", config=hyperparameters):
        wandb.run.name = str(config_['appliance']) + "_Test:" + str(test_buildings) + "_Train:" + str(train_buildings)

        config = wandb.config

        # lengths = [85320, 132480, 132480, 132480, 132480, 132480, 132480]

        global_model, criterion, optimizer = make_model(config)

        client_models = [LSTM(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            config.hidden_size_1,
            config.hidden_size_2,
            config.fc1,
            config.batch_size,
            config.window_size
        ).to(device) for _ in range(len(train_buildings))]

        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        initial_weights = [copy.deepcopy(model.state_dict()) for model in client_models]

        optimizers = [torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        ) for model in client_models]

        print(global_model)
        print("Window Length: ", window_length)

        wandb.watch(global_model, criterion, log="all", log_freq=10)

        validation_loader, test_loader, test_val_seq_std, test_val_seq_mean = make_test_val_data(
            config,
            test_month,
            appliance,
            window_length,
            test_buildings
        )

        test_results = []
        train_results = []
        example_ct = 0
        batch_ct = 0
        all_epochs = 0
        for r in range(20):
            client_losses = 0.0
            gc.collect()
            torch.cuda.empty_cache()
            for i in range(len(train_buildings)):
                # wandb.watch(client_models[i], criterion, log="all", log_freq=10)
                time_log = time.time()
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizers[i], step_size=50, gamma=0.9, verbose=True)
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizers[i],
                    base_lr=.95 * config_['learning_rate'],
                    max_lr=1 * config_['learning_rate'],
                    #step_size_up=50,
                    #step_size_down=6000,
                    gamma=1,
                    cycle_momentum=False,
                    verbose=False
                )

                train_loader, train_seq_std, train_seq_mean = make_train_data(
                    config,
                    train_months,
                    appliance,
                    window_length,
                    [train_buildings[i]]
                )

                # loss += client_update(client_models[i], optimizer[i], train_loader, epochs = config.epochs)
                example_ct, batch_ct, all_epochs, _, client_train_loss = train(
                    client_models[i].to(device),
                    train_loader,
                    validation_loader,
                    criterion,
                    optimizers[i],
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


                client_losses += client_train_loss

                print("Time to train on one home: ", time.time() - time_log)


            new_weights = [copy.deepcopy(model.state_dict()) for model in client_models]
            print('weight_deltas: ', [torch.sub(new_weights[i].items(), initial_weights[i].items()) for i in range(len(new_weights))])

            client_losses = client_losses / len(train_buildings)
            server_aggregate(global_model, client_models)
            test_results.append(test(global_model, test_loader, criterion, test_val_seq_std, test_val_seq_mean))
            train_results.append(client_losses)

            print("train_results: ", train_results)
            print("test_results: ", test_results)
            print("Round_" + str(r) + "_results: ",
                  test(global_model, test_loader, criterion, test_val_seq_std, test_val_seq_mean))

    return train_results, test_results, global_model

home_ids = homes.dataid.unique()[0:4]

home_ids_train = [x for x in home_ids if x!=3383]

PATH = "/home/Alfredo/private_nilm/models_power_ratio_filter_with_fl"

final_results = {}
random.seed(3)
global_models = []
best_models = []
training_homes = list(home_ids)

for i in range(1):
    gc.collect()
    torch.cuda.empty_cache()
    #training_homes.append(i)
    testing_homes = [3383]

    patience = 20
    print("patience: ", patience)
    print("training_home: ", training_homes)
    print("test_home: ", testing_homes)
    train_results, test_results, global_model = model_pipeline(
        config_,
        'sept_oct_nov',
        'dec',
        config_['appliance'],
        config_['window_size'],
        training_homes,
        testing_homes,
        patience)
    global_models.append(global_model)
    result = {"Train_home_" + str(training_homes) + "_Test_home_" + str(testing_homes): test_results}
    final_results.update(result)
    print(final_results)
    global_model.cpu()
    torch.save(global_model.state_dict(), PATH + "/seq2point_fl_refrigerator_model_3383test_train" + str(len(training_homes)) + "_homes_trial1.pth")