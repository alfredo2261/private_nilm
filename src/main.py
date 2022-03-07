import wandb
from data_loaders import make_train_data, make_test_val_data, make_model
from train import train, test
import torch
import time
import gc
import config_file
from clean_data import load_all_houses_with_device
import random

wandb.login()

torch.cuda.is_available()

gc.collect()
torch.cuda.empty_cache()

torch.__version__  # should be


config_ = config_file.config


def model_pipeline(hyperparameters, train_months, test_month, appliance, window_length, train_buildings,
                   test_buildings):
    with wandb.init(project="Drye1_single_houses", config=hyperparameters):
        wandb.run.name = "Test1706normalize_over_dataset_hs1:32_hs2:200_fc1:200_fc2:100_1200epochs100batch_StepLR.0004_step50_gamma0.9_3dropout_weightdecay.00005_kern:7_outchan:16_4sigmoid_0maxpool_Trainbldgs:" + str(
            train_buildings)

        config = wandb.config

        # lengths = [85320, 132480, 132480, 132480, 132480, 132480, 132480]

        model, criterion, optimizer = make_model(config)

        print(model)

        wandb.watch(model, criterion, log="all", log_freq=10)

        example_ct = 0
        batch_ct = 0
        all_epochs = 0
        # all step_size=30 tests had gamma=0.5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9, verbose=True)

        validation_loader, test_loader = make_test_val_data(config, test_month, appliance, window_length,
                                                            test_buildings)

        time_log = time.time()
        train_loader = make_train_data(config, train_months, appliance, window_length, train_buildings)
        model, example_ct, batch_ct, all_epochs = train(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            config,
            example_ct,
            batch_ct,
            all_epochs,
            scheduler)

        print("Time to train on one home: ", time.time() - time_log)

        results = test(model, test_loader, criterion)

    return model, results


homes = load_all_houses_with_device(config_file.path, 'drye1')


drye1_homes = homes['dataid'].loc[homes['dataid'] != 1706].unique()


final_results = {}
random.seed(5)
for j in drye1_homes:
    for i in range(10):
        model, per_house_result = model_pipeline(
        config_,
        'may_june_july',
        'may_june_july',
        'drye1',
        100,
        [j],
        [1706])
        result = {j: per_house_result}
        final_results.update(result)
