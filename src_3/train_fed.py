import wandb
import numpy as np
import torch
from sklearn import metrics
from data_loaders import device
import gc
import copy


def precision(prediction, true):
    numerator = np.sum([min(l) for l in zip(prediction, true)])
    return numerator / np.sum(prediction)


def recall(prediction, true):
    numerator = np.sum([min(l) for l in zip(prediction, true)])
    return numerator / np.sum(true)


def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)


def train(model, loader, validation_loader, criterion, optimizer, config, example_ct, batch_ct, all_epochs,
          scheduler, val_seq_min, val_seq_max, train_seq_min, train_seq_max, patience):  # , privacy_engine):

    the_last_loss = 1000
    trigger_times = 0
    mae_compare = 1000.0

    for epoch in (range(config.epochs)):
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        epoch_total_loss = 0.0
        epoch_predictions = []
        epoch_true_vals = []
        epoch_abs_diff = 0.0
        epoch_squ_diff = 0.0
        epoch_true_sum = 0.0
        epoch_squ_sum = 0.0
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
            #When using cyclicLR
            scheduler.step()

        #When using stepLR
        #scheduler.step()

        # validation process
        model.eval()
        with torch.no_grad():
            validation_total_loss = 0.0
            val_abs_diff = 0.0
            val_squ_diff = 0.0
            val_true_sum = 0.0
            val_squ_sum = 0.0
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

        for i in range(len(epoch_predictions)):
            if len(epoch_predictions[i].shape) == 1:
                epoch_predictions[i] = [epoch_predictions[i]]

        for i in range(len(epoch_true_vals)):
            if len(epoch_true_vals[i].shape) == 1:
                epoch_true_vals[i] = [epoch_true_vals[i]]

        epoch_predictions = [item for sublist in epoch_predictions for item in sublist]
        epoch_true_vals = [item for sublist in epoch_true_vals for item in sublist]

        for i in range(len(val_preds)):
            if len(val_preds[i].shape) == 1:
                val_preds[i] = [val_preds[i]]

        for i in range(len(val_true)):
            if len(val_true[i].shape) == 1:
                val_true[i] = [val_true[i]]

        val_preds = [item for sublist in val_preds for item in sublist]
        val_true = [item for sublist in val_true for item in sublist]

        train_tracker = 0
        train_examples = 0

        #if any(isinstance(el, list) for el in epoch_predictions):
        for i in range(len(epoch_predictions)):
            if len(epoch_true_vals[i]) > 1 and len(epoch_predictions[i]) > 1: #check implementation
                epoch_true_vals[i] = np.add(np.multiply(epoch_true_vals[i], train_seq_max - train_seq_min), train_seq_min)
                epoch_true_vals[i] = np.exp(epoch_true_vals[i]) - 1
                epoch_predictions[i] = np.add(np.multiply(epoch_predictions[i], train_seq_max - train_seq_min), train_seq_min)
                epoch_predictions[i] = np.exp(epoch_predictions[i]) - 1
                epoch_abs_diff += np.sum(abs(np.subtract(epoch_true_vals[i], epoch_predictions[i])))
                epoch_squ_diff += np.sum(np.square(np.subtract(epoch_true_vals[i], epoch_predictions[i])))
                epoch_true_sum += np.sum(epoch_true_vals[i])
                epoch_squ_sum += np.sum(np.square(epoch_true_vals[i]))
                train_tracker += 1
                train_examples += len(epoch_predictions[i])

        epoch_predictions = [item for sublist in epoch_predictions for item in sublist]
        epoch_true_vals = [item for sublist in epoch_true_vals for item in sublist]

        epoch_precision = precision(epoch_predictions, epoch_true_vals)
        epoch_recall = recall(epoch_predictions, epoch_true_vals)
        epoch_r2 = metrics.r2_score(epoch_true_vals, epoch_predictions)
        epoch_mse = np.mean(np.square(np.subtract(np.array(epoch_true_vals), np.array(epoch_predictions))))

        val_tracker = 0
        val_examples = 0
        for i in range(len(val_preds)):
            if len(val_true[i]) > 1 and len(val_preds[i]) > 1:
                val_true[i] = np.add(np.multiply(val_true[i], val_seq_max - val_seq_min), val_seq_min)
                val_true[i] = np.exp(val_true[i]) - 1
                val_preds[i] = np.add(np.multiply(val_preds[i], val_seq_max - val_seq_min), val_seq_min)
                val_preds[i] = np.exp(val_preds[i]) - 1
                val_abs_diff += np.sum(abs(np.subtract(val_true[i], val_preds[i])))
                val_squ_diff += np.sum(np.square(np.subtract(val_true[i], val_preds[i])))
                val_true_sum += np.sum(val_true[i])
                val_squ_sum += np.sum(np.square(val_true[i]))
                val_tracker += 1
                val_examples += len(val_preds[i])

        val_preds = [item for sublist in val_preds for item in sublist]
        val_true = [item for sublist in val_true for item in sublist]

        val_precision = precision(val_preds, val_true)
        val_recall = recall(val_preds, val_true)
        val_r2 = metrics.r2_score(val_true, val_preds)
        val_mse = np.mean(np.square(np.subtract(np.array(val_true), np.array(val_preds))))

        validation_loss = val_abs_diff/val_examples
        #PATH = r"C:\Users\aar245\Desktop\privacy_preserving_nn\models"
        if validation_loss < mae_compare:
            best_model = copy.deepcopy(model)
            #torch.save(model.state_dict(), PATH+"\\"+str(config.appliance)+"_checkpoint.pth")
            mae_compare = validation_loss

        wandb.log({
            'Training_Loss': epoch_total_loss / batch_number, #good
            'Validation_Loss': validation_total_loss / val_batch, #good
            'Training_R2': epoch_r2, #good
            'Validation_R2': val_r2, #good
            'Training_F1': f1_score(epoch_precision, epoch_recall), #good
            'Validation_F1': f1_score(val_precision, val_recall), #good
            'Training_NEP': epoch_abs_diff / epoch_true_sum, #good
            'Validation_NEP': val_abs_diff / val_true_sum, #good
            'Training_NDE': epoch_squ_diff / epoch_squ_sum, #good
            'Validation_NDE': val_squ_diff / val_squ_sum, #good
            'Training_MAE': epoch_abs_diff / train_examples, #good
            'Validation_MAE': val_abs_diff / val_examples,
            'Training_MSE': epoch_mse,
            'Validation_MSE': val_mse}, #good
            step=all_epochs)

        # Early stopping
        current_loss = val_abs_diff / val_examples
        if current_loss > the_last_loss:
           trigger_times += 1
           print('trigger times:', trigger_times)

           if trigger_times >= patience:
               print('Early stopping!\nStart to test process.')
               print(f"Loss after " + str(example_ct).zfill(5) + f" batches: {epoch_total_loss/batch_number:.4f}")
               return example_ct, batch_ct, all_epochs, best_model, epoch_total_loss/batch_number

        else:
           print('trigger times: 0')
           #based on global minimum validation MAE achieved
           the_last_loss = current_loss
           trigger_times = 0

        # based on local minimum validation MAE achieved
        #the_last_loss = current_loss
        print(f"Loss after " + str(example_ct).zfill(5) + f" batches: {epoch_total_loss/batch_number:.4f}")
        # for param in best_model.parameters():
        #     print(param.data)
    return example_ct, batch_ct, all_epochs, best_model, epoch_total_loss/batch_number


def train_batch(features, labels, model, optimizer, criterion):

    features = features.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(features)
    loss = criterion(outputs, labels)

    y_pred = outputs.cpu().detach().numpy()
    y_true = labels.cpu().detach().numpy()

    loss.backward()

    optimizer.step()
    #print("Conv_gradient: ", model.conv1.weight.grad)
    #print("LSTM1_gradient_ih: ", model.lstm1.weight_ih_l0.grad)
    #print("LSTM1_gradient_hh: ", model.lstm1.weight_hh_l0.grad)
    #print("LSTM2_gradient_ih: ", model.lstm2.weight_ih_l0.grad)
    #print("LSTM2_gradient_hh: ", model.lstm2.weight_hh_l0.grad)
    del features, labels

    return y_pred, y_true, loss


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


def test(model, test_loader, criterion, test_seq_min, test_seq_max):  # , privacy_engine):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_steps = 0
        abs_diff = 0.0
        squ_diff = 0.0
        squ_sum = 0.0
        true_sum = 0.0

        predictions = []
        true_vals = []
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            prediction = model(features)
            loss = criterion(prediction, labels)
            total_loss += float(loss.item())
            total_steps += 1
            predictions.append(prediction.cpu().detach().numpy())
            true_vals.append(labels.cpu().detach().numpy())
            del features, labels

        for i in range(len(predictions)):
            if len(predictions[i].shape) == 1:
                predictions[i] = [predictions[i]]

        for i in range(len(true_vals)):
            if len(true_vals[i].shape) == 1:
                true_vals[i] = [true_vals[i]]

        predictions = [item for sublist in predictions for item in sublist]
        true_vals = [item for sublist in true_vals for item in sublist]

        # if not any(isinstance(el, list) for el in predictions):
        #     predictions = [predictions]
        #     true_vals = [true_vals]

        tracker = 0
        test_examples = 0
        for i in range(len(predictions)):
            if len(true_vals[i]) > 1 and len(predictions[i]) > 1:
                true_vals[i] = np.add(np.multiply(true_vals[i], test_seq_max - test_seq_min), test_seq_min)
                true_vals[i] = np.exp(true_vals[i]) - 1
                predictions[i] = np.add(np.multiply(predictions[i], test_seq_max - test_seq_min), test_seq_min)
                predictions[i] = np.exp(predictions[i]) - 1
                abs_diff += np.sum(abs(np.subtract(true_vals[i], predictions[i])))
                squ_diff += np.sum(np.square(np.subtract(true_vals[i], predictions[i])))
                squ_sum += np.sum(np.square(true_vals[i]))
                true_sum += np.sum(true_vals[i])
                tracker += 1
                test_examples += len(predictions[i])

        predictions = [item for sublist in predictions for item in sublist]
        true_vals = [item for sublist in true_vals for item in sublist]

        local_precision = precision(predictions, true_vals)
        local_recall = recall(predictions, true_vals)
        r_squared = metrics.r2_score(true_vals, predictions)
        mse = np.mean(np.square(np.subtract(np.array(true_vals), np.array(predictions))))

        wandb.log({
            'Test_Loss': total_loss / total_steps,
            'Test_R2_Value': r_squared,
            'Test_F1_Score': f1_score(local_precision, local_recall),
            'Test_NEP': abs_diff / true_sum,
            'Test_NDE': squ_diff / squ_sum,
            'Test_MAE': abs_diff / test_examples,
            'Test_MSE': mse})
    # if privacy_engine:
    # epsilon, best_alpha = privacy_engine.get_privacy_spent()
    # print("Testing epsilon: ", epsilon)
    # print("Testing alpha: ", best_alpha)

    #wandb.save("model_final.h5")

    results = [total_loss / total_steps, r_squared, f1_score(local_precision, local_recall),
               abs_diff / true_sum, squ_diff / squ_sum, abs_diff / test_examples, mse]

    return results
