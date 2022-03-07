import wandb
import numpy as np
import torch
from sklearn import metrics
from data_loaders import device


def precision(prediction, true):
    numerator = np.sum([min(l) for l in zip(prediction, true)])
    return numerator / np.sum(prediction)


def recall(prediction, true):
    numerator = np.sum([min(l) for l in zip(prediction, true)])
    return numerator / np.sum(true)


def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)


def train(model, loader, validation_loader, criterion, optimizer, config, example_ct, batch_ct, all_epochs,
          scheduler):  # , privacy_engine):
    home_total_loss = 0.0
    total_validation_loss = 0.0
    # the_last_loss = 100
    # patience = 10
    # trigger_times = 0
    # total_home_r2 = 0.0
    for epoch in (range(config.epochs)):
        model.train()
        epoch_total_loss = 0.0
        epoch_predictions = []
        epoch_true_vals = []
        epoch_r2 = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        epoch_abs_diff = 0.0
        epoch_squ_diff = 0.0
        epoch_true_sum = 0.0
        batch_number = 0
        all_epochs += 1
        for _, (features, labels) in enumerate(loader):
            train_predictions, train_true_vals, loss = train_batch(features, labels, model, optimizer, criterion)
            epoch_total_loss += float(loss.item())
            epoch_predictions.append(train_predictions)
            epoch_true_vals.append(train_true_vals)
            example_ct += len(features)
            batch_ct += 1
            batch_number += 1
            # scheduler.step()

        scheduler.step()

        # validation process
        model.eval()
        with torch.no_grad():
            validation_total_loss = 0.0
            # val_predictions = []
            # val_true_vals = []
            val_r2 = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_abs_diff = 0.0
            val_squ_diff = 0.0
            val_true_sum = 0.0
            val_batch = 0

        val_preds = []
        val_true = []
        for features, labels in validation_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            validation_total_loss += float(loss.item())

            val_preds.append(outputs.cpu().detach().numpy())
            val_true.append(labels.cpu().detach().numpy())

            val_batch += 1
            del features, labels

        home_total_loss += epoch_total_loss / batch_number
        total_validation_loss += validation_total_loss / val_batch
        epoch_predictions = [item for sublist in epoch_predictions for item in sublist]
        epoch_true_vals = [item for sublist in epoch_true_vals for item in sublist]
        all_val_preds = [item for sublist in val_preds for item in sublist]
        all_val_true_vals = [item for sublist in val_true for item in sublist]

        train_tracker = 0
        for i in range(len(epoch_predictions)):
            if len(epoch_true_vals[i]) != 1 and len(epoch_predictions[i]) != 1:
                for j in range(len(epoch_predictions[i])):
                    if len(epoch_true_vals[i][j].shape) != 0 and len(epoch_predictions[i][j].shape) != 0:
                        epoch_true_vals[i][j] = np.exp(epoch_true_vals[i][j]) - 1
                        epoch_predictions[i][j] = np.exp(epoch_predictions[i][j]) - 1
                        epoch_r2 += metrics.r2_score(epoch_true_vals[i][j], epoch_predictions[i][j])
                        epoch_precision += precision(epoch_predictions[i][j], epoch_true_vals[i][j])
                        epoch_recall += recall(epoch_predictions[i][j], epoch_true_vals[i][j])
                        epoch_abs_diff += np.sum(abs(np.subtract(epoch_true_vals[i][j], epoch_predictions[i][j])))
                        epoch_squ_diff += np.sum(np.square(np.subtract(epoch_true_vals[i][j], epoch_predictions[i][j])))
                        epoch_true_sum += np.sum(epoch_true_vals[i][j])
                        train_tracker += 1

        val_tracker = 0
        for i in range(len(all_val_preds)):
            if len(all_val_true_vals[i]) > 1 and len(all_val_preds[i]) > 1:
                all_val_true_vals[i] = np.exp(all_val_true_vals[i]) - 1
                all_val_preds[i] = np.exp(all_val_preds[i]) - 1
                val_r2 += metrics.r2_score(all_val_true_vals[i], all_val_preds[i])
                val_precision += precision(all_val_preds[i], all_val_true_vals[i])
                val_recall += recall(all_val_preds[i], all_val_true_vals[i])
                val_abs_diff += np.sum(abs(np.subtract(all_val_true_vals[i], all_val_preds[i])))
                val_squ_diff += np.sum(np.square(np.subtract(all_val_true_vals[i], all_val_preds[i])))
                val_true_sum += np.sum(all_val_true_vals[i])
                val_tracker += 1

        total_precision = epoch_precision / train_tracker
        total_recall = epoch_recall / train_tracker

        total_val_precision = val_precision / val_tracker
        total_val_recall = val_recall / val_tracker

        wandb.log({
            'Training_MSE': epoch_total_loss / batch_number,
            'Validation_MSE': validation_total_loss / val_batch,
            'Training_R2': epoch_r2 / train_tracker,
            'Validation_R2': val_r2 / val_tracker,
            'Training_F1': f1_score(total_precision, total_recall),
            'Validation_F1': f1_score(total_val_precision, total_val_recall),
            'Training_NEP': epoch_abs_diff / epoch_true_sum,
            'Validation_NEP': val_abs_diff / val_true_sum,
            'Training_NDE': epoch_squ_diff / epoch_true_sum,
            'Validation_NDE': val_squ_diff / val_true_sum,
            'Training_MAE': epoch_abs_diff / train_tracker,
            'Validation_MAE': val_abs_diff / val_tracker},
            step=all_epochs)

        # Early stopping
        # current_loss = validation_total_loss/val_batch
        # if current_loss > the_last_loss:
        #    trigger_times += 1
        #    print('trigger times:', trigger_times)

        #    if trigger_times >= patience:
        #        print('Early stopping!\nStart to test process.')
        #        print(f"Loss after " + str(example_ct).zfill(5) + f" batches: {epoch_total_loss / batch_number:.4f}")
        #        return model, example_ct, batch_ct, all_epochs

        # else:
        #    print('trigger times: 0')
        #    trigger_times = 0

        # the_last_loss = current_loss

        print(f"Loss after " + str(example_ct).zfill(5) + f" batches: {epoch_total_loss / batch_number:.4f}")

        # train_log(epoch_total_loss / batch_number,
        #        validation_total_loss / val_batch,
        #        example_ct,
        #        all_epochs)

    return model, example_ct, batch_ct, all_epochs


def train_batch(features, labels, model, optimizer, criterion):
    train_predictions = []
    train_true_vals = []

    features = features.to(device)
    labels = labels.to(device)

    outputs = model(features)
    loss = criterion(outputs, labels)

    train_predictions.append(outputs.cpu().detach().numpy())
    train_true_vals.append(labels.cpu().detach().numpy())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    del features, labels

    return train_predictions, train_true_vals, loss


def train_log(loss, val_loss, example_ct, epoch):
    loss = float(loss)
    val_loss = float(val_loss)

    wandb.log({
        "Rounds_on_all_homes": epoch,
        "All_Homes_Training_MSE": loss,
        "All_Homes_Validation_MSE": val_loss})
    # "All_Homes_Training_R2": training_r2})
    # step = example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" total epochs: {loss:.4f}")


def test(model, test_loader, criterion):  # , privacy_engine):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_steps = 0
        r_squared = 0.0
        local_precision = 0.0
        local_recall = 0.0
        abs_diff = 0.0
        squ_diff = 0.0
        true_sum = 0.0
        predictions = []
        true_vals = []
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            prediction = model(features)
            loss = criterion(prediction, labels)
            total_loss += float(loss.item())
            total_steps += 1
            # r_squared += float(metrics.r2_score(labels.cpu().detach().numpy(), prediction.cpu().detach().numpy()))
            predictions.append(prediction.cpu().detach().numpy())
            true_vals.append(labels.cpu().detach().numpy())
            del features, labels

        tracker = 0
        for i in range(len(predictions)):
            if len(true_vals[i]) != 1 and len(predictions[i]) != 1:
                for j in range(len(predictions[i])):
                    if len(true_vals[i][j].shape) != 0 and len(predictions[i][j].shape) != 0:
                        r_squared += metrics.r2_score(true_vals[i][j], predictions[i][j])
                        local_precision += precision(predictions[i][j], true_vals[i][j])
                        local_recall += recall(predictions[i][j], true_vals[i][j])
                        abs_diff += np.sum(abs(np.subtract(true_vals[i][j], predictions[i][j])))
                        squ_diff += np.sum(np.square(np.subtract(true_vals[i][j], predictions[i][j])))
                        true_sum += np.sum(true_vals[i][j])
                        tracker += 1

        total_precision = local_precision / tracker
        total_recall = local_recall / tracker

        wandb.log({
            'Test_MSE': total_loss / total_steps,
            'Test_R2_Value': r_squared / tracker,
            'Test_F1_Score': f1_score(total_precision, total_recall),
            'Test_NEP': abs_diff / true_sum,
            'Test_NDE': squ_diff / true_sum,
            'Test_MAE': abs_diff / tracker})
    # if privacy_engine:
    # epsilon, best_alpha = privacy_engine.get_privacy_spent()
    # print("Testing epsilon: ", epsilon)
    # print("Testing alpha: ", best_alpha)

    wandb.save("model_final.h5")

    results = [total_loss / total_steps, r_squared / tracker, f1_score(total_precision, total_recall),
               abs_diff / true_sum, squ_diff / true_sum, abs_diff / tracker]

    return results
