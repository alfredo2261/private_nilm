import wandb
from data_loaders_optimization import make_train_data, make_test_val_data, make_model
from train import train, test
import torch
import time
import gc
import config_file
from clean_data import load_all_houses_with_device
import random
import optuna

wandb.login()

print(torch.cuda.is_available())

gc.collect()
torch.cuda.empty_cache()

print(torch.__version__)


config_ = config_file.load_hyperparameters("refrigerator1")

homes = load_all_houses_with_device(config_file.path, config_['appliance'])

refrigerator_homes = homes['dataid'].unique()


def objective(trial):
    params = {
        "hidden_size_1": trial.suggest_int("hidden_size_1", 50, 150),
        "hidden_size_2": trial.suggest_int("hidden_size_2", 100, 250),
        "fc1": trial.suggest_int("fc1", 50, 400),
        "weight_decay": trial.suggest_uniform("weight_decay", 0.01, 0.12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
        "window_size": trial.suggest_int("window_size", 50, 200)
    }

    model, result, best_model = model_pipeline(
        config_,
        'may_june_july',
        'aug',
        config_['appliance'],
        params['window_size'],
        list(drye1_homes),
        [random.choice(drye1_homes)],
        params)

    return result[-1]


def model_pipeline(hyperparameters, train_months, test_month, appliance, window_length, train_buildings,
                   test_buildings, params):
    with wandb.init(project="refrigerator1_optimization_jan6", config=hyperparameters, dir=r"F:\alfredo_rodriguez"):
        wandb.run.name = "attention_Jan_9:Test:" + str(test_buildings) + "_Train:" + str(
            train_buildings) + "_Windowlength:" + str(window_length)

        gc.collect()
        torch.cuda.empty_cache()

        config = wandb.config

        # lengths = [85320, 132480, 132480, 132480, 132480, 132480, 132480]

        model, criterion, optimizer = make_model(config, params)

        print(model)
        print("Window Length: ", window_length)

        wandb.watch(model, criterion, log="all", log_freq=10)

        example_ct = 0
        batch_ct = 0
        all_epochs = 0
        # all step_size=30 tests had gamma=0.5
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.85, verbose=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, verbose=True)

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=.01 * params['learning_rate'],
            max_lr=2 * params['learning_rate'],
            step_size_up=20,
            step_size_down=200,
            gamma=0.9,
            cycle_momentum=False,
            verbose=True)

        validation_loader, test_loader, test_val_seq_min, test_val_seq_max = make_test_val_data(
            config,
            test_month,
            appliance,
            window_length,
            test_buildings
        )

        time_log = time.time()
        train_loader, train_seq_min, train_seq_max = make_train_data(
            config,
            train_months,
            appliance,
            window_length,
            train_buildings
        )

        model, example_ct, batch_ct, all_epochs, best_model = train(
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
            test_val_seq_min,
            test_val_seq_max,
            train_seq_min,
            train_seq_max
        )

        print("Time to train on one home: ", time.time() - time_log)

        results = test(best_model, test_loader, criterion, test_val_seq_min, test_val_seq_max)

    return model, results, best_model


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("best trial:")
    trial_ = study.best_trial

    print(trial_.values)
    print(trial_.params)