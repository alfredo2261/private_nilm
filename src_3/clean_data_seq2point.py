import numpy as np
import pandas as pd
import random
import time
from scipy import stats
from datetime import timedelta


def floor_outliers(df, cap):
    df['appliance_power'] = df['appliance_power'].mask(df['appliance_power'] > cap, 0)
    return df


def load_all_houses_with_device(path, appliance):
    df = pd.read_csv(path, usecols=['dataid', 'localminute', 'grid', 'solar', 'solar2', str(appliance)])
    df['localminute'] = pd.to_datetime(df['localminute'])
    df = df.fillna(0)
    df = df.set_index(['localminute'])
    grouped = df[['dataid', str(appliance)]].groupby(df.dataid).sum()
    house_ids = grouped[grouped[str(appliance)] > 0].dataid.index.unique()
    df = df[df.dataid.isin(house_ids)]
    #df = df.reset_index(drop=True)

    df['net_power'] = df.grid + df.solar + df.solar2
    df['net_power'] = np.clip(df['net_power']*1000, a_min=0, a_max=None)
    df['net_power'] = df['net_power'].mask(df['net_power'] > 30000, 0)

    df[str(appliance)] = np.clip(df[str(appliance)]*1000, a_min=0, a_max=None)
    df['appliance_power'] = df[str(appliance)]

    df = df.drop(columns=['grid', 'solar', 'solar2', str(appliance)])
    df['appliance_power'] = df['appliance_power'].values.astype('float32')
    df['net_power'] = df['net_power'].values.astype('float32')

    remove = []
    cap = []
    if appliance == "air1":
        cap = 5000
        # remove = [1249, 7365, 8849]
    if appliance == 'clotheswasher1':
        cap = 1300
        # remove = [6240]
    # if appliance == 'drye1':
    #    remove = [183]
    if appliance == 'furnace1':
        cap = 1000
        # remove = [6703, 3403, 7365, 2126, 10089]
    if appliance == 'oven1':
        cap = 5000
        # remove = [792]
    if appliance == 'range1':
        cap = 8000
        # remove = [7367]
    if appliance == 'dishwasher1':
        cap = 1500
        # remove = [8849, 6240, 7367]
    # if appliance == 'freezer1':
    # remove = [9973, 7159]
    # if appliance == 'heater1':
    # remove = [2561]
    if appliance == 'waterheater1':
        cap = 6000
        # remove = [10983, 9973, 8627, 2096]
    if appliance == 'waterheater2':
        cap = 5000
        # remove = [8627]
    if appliance == 'wellpump1':
        cap = 1600
        # remove = [10983]
    if appliance == 'clotheswasher_dryg1':
        cap = 800
    if appliance == 'refrigerator1':
        cap = 2000
        remove = [984]

        #removing top 20 in terms of power ratio
        #remove = [145, 183, 335, 387, 526, 1417, 2358, 3383, 4628, 6240, 6526, 6672, 7021, 7069, 7365, 9004, 10554, 10811, 10983, 11878]

        # remove = [2561,
        #          3976,
        #          11785,
        #          142,
        #          3344,
        #          8849,
        #          3996,
        #          11421,
        #          3488,
        #          6178,
        #          6564,
        #          9002,
        #          6703,
        #          690,
        #          8627,
        #          10164,
        #          6069,
        #          950,
        #          5058,
        #          6594,
        #          10182,
        #          5192,
        #          9290,
        #          2126,
        #          984,
        #          1240,
        #          9053,
        #          5982,
        #          1249,
        #          8162,
        #          3700,
        #          9973,
        #          5367,
        #          8825,
        #          6907,
        #          7935]

        # #Based on k-means clustering
        # remove = [145,
        #           3344,
        #           4628,
        #           10554,
        #           950,
        #           984,
        #           1240,
        #           1249,
        #           5192,
        #           5367,
        #           5982,
        #           6069,
        #           6564,
        #           6594,
        #           6703,
        #           6907,
        #           7935,
        #           8162,
        #           8627,
        #           8849,
        #           9002,
        #           11421,
        #           11785
        #           ]

        # #Based on homes with incomplete recording length
        #remove = [145, 3344, 4628, 10554]

        # #Based on building square footage
        # remove = [
        #     145,
        #     690,
        #     950,
        #     984,
        #     1249,
        #     2561,
        #     3344,
        #     3700,
        #     4628,
        #     5367,
        #     5982,
        #     6907,
        #     7021,
        #     8162,
        #     8849,
        #     9973,
        #     10164,
        #     10554
        # ]
        # Based on building square footage
        # keep = [  142,   183,   335,   387,   526,  1240,  1417,  2126,  2358,
        #      3383,  3488,  3976,  3996,  5058,  5192,  6069,  6178,  6240,
        #      6526,  6564,  6594,  6672,  6703,  7069,  7365,  7935,  8627,
        #      8825,  9002,  9004,  9053,  9290, 10182, 10811, 10983, 11421,
        #     11785, 11878]
    if remove:
        for house in remove:
            df = df[df['dataid'] != house]

    if cap:
        df = floor_outliers(df, cap)
    return df


def normalize_x(x_sequence):
    std = np.std(x_sequence)
    return (x_sequence - np.mean(x_sequence)) / std


def normalize_y(y_sequence):
    std = np.std(y_sequence)
    y_mean = np.mean(y_sequence)
    # y_sequence = np.log(np.add(y_sequence, 1))
    # sequence_min = np.min(y_sequence)
    # sequence_max = np.max(y_sequence)
    # output = (y_sequence - sequence_min) / (sequence_max - sequence_min)
    return (y_sequence-y_mean)/std, std, y_mean


# def normalize_y(y_sequence):
#     temp_y = [item for subitem in y_sequence for item in subitem]
#     std = np.std(temp_y)
#     y_mean = np.mean(temp_y)
#     # y_sequence = np.log(np.add(y_sequence, 1))
#     # sequence_min = np.min(y_sequence)
#     # sequence_max = np.max(y_sequence)
#     # output = (y_sequence - sequence_min) / (sequence_max - sequence_min)
#     return [item for subitem in (y_sequence-y_mean)/std for item in subitem], std, y_mean


def add_padding(data, window_length):
    padding = np.zeros(int(window_length/2))
    data = np.append(data, padding)
    data = np.insert(data, 0, padding)
    return data


def split_data(data, window_length):
    data = add_padding(data, window_length)
    steps = len(data) - window_length
    data_splits = []
    for i in range(steps+1):
        data_splits.append(data[i:window_length+i])
    return list(data_splits)


def synthetic_data(data, ratio, threshold):
    threshold = min(data.appliance_power.values) + threshold
    off_ratio = (data.appliance_power.values < threshold).sum() / len(data.appliance_power.values)

    aggregate_data = data.net_power.values

    data = data.appliance_power.values

    differences = []
    for i in range(len(data) - 1):
        differences.append(data[i + 1] - data[i])

    chunks = [[]]
    indices = [[]]
    time_step = 0
    chunk_count = 0
    chunks[chunk_count].append(data[0])
    indices[chunk_count].append(0)
    while time_step < len(data) - 1:
        if differences[time_step] > threshold:
            time_step += 1
            chunk_count += 1
            chunks.append([])
            indices.append([])
            chunks[chunk_count].append(data[time_step])
            indices[chunk_count].append(time_step)
        elif differences[time_step] < - threshold:
            time_step += 1
            chunk_count += 1
            chunks.append([])
            indices.append([])
            chunks[chunk_count].append(data[time_step])
            indices[chunk_count].append(time_step)
        else:
            time_step += 1
            chunks[chunk_count].append(data[time_step])
            indices[chunk_count].append(time_step)

    for i in range(len(chunks) - 1):
        if len(chunks[i]) < 2:
            chunks[i + 1].insert(0, chunks[i][0])
            indices[i + 1].insert(0, indices[i][0])

    chunks = [i for i in chunks if len(i) > 1]
    indices = [i for i in indices if len(i) > 1]

    datapoints_ratio = 2 * abs(off_ratio - ratio)
    datapoints = int(datapoints_ratio * len(data))

    on_sequences = [(index, value) for index, value in enumerate(chunks) if np.mean(value) > threshold]
    off_sequences = [(index, value) for index, value in enumerate(chunks) if np.mean(value) <= threshold]

    on_seq_vals = list(zip(*on_sequences))[1]
    off_seq_vals = list(zip(*off_sequences))[1]

    on_seq_indices = list(zip(*on_sequences))[0]
    off_seq_indices = list(zip(*off_sequences))[0]

    on_seq_indices = [indices[i] for i in on_seq_indices]
    off_seq_indices = [indices[i] for i in off_seq_indices]

    if off_ratio > ratio:
        datapoint_count = 0
        while datapoint_count < datapoints:
            random_location = random.randint(0, len(chunks) - 1)
            random_on_sequence = random.randint(0, len(on_sequences) - 1)
            chunks.insert(random_location, on_seq_vals[random_on_sequence])
            indices.insert(random_location, on_seq_indices[random_on_sequence])
            datapoint_count += len(on_seq_vals[random_on_sequence])

    if off_ratio < ratio:
        datapoint_count = 0
        while datapoint_count < datapoints:
            random_location = random.randint(0, len(chunks) - 1)
            random_off_sequence = random.randint(0, len(off_sequences) - 1)
            chunks.insert(random_location, off_seq_vals[random_off_sequence])
            indices.insert(random_location, off_seq_indices[random_off_sequence])
            datapoint_count += len(off_sequences[random_off_sequence])

    chunks = [i for subitem in chunks for i in subitem]
    indices = [i for subitem in indices for i in subitem]

    aggregate_data = [aggregate_data[i] for i in indices]
    appliance_data = chunks

    return aggregate_data, appliance_data


def create_activations(path, appliance, window_length, buildings):
    data = load_all_houses_with_device(path, appliance)
    data = [data.loc[data['dataid'] == i] for i in buildings]

    x_sets = []
    y_sets = []
    for i in data:
        i = i.loc[~i.index.duplicated(), :]
        unique_days = i.index.map(lambda t: t.date()).unique()
        i['net_power'] = split_data(normalize_x(i['net_power'].values), window_length)
        for j in unique_days[0:-1]:
            start_date = j.strftime("%Y-%m-%d")
            end_date = (j + timedelta(days = 1)).strftime("%Y-%m-%d")
            day_data = i.loc[(i.index >= start_date) & (i.index < end_date)]
            X_padded = []
            for sequence in day_data.net_power:
                X_padded.append(list(sequence))
            if len(X_padded) == 1440:
                x_sets.append(X_padded)
                y_sets.append(day_data.appliance_power)

    y_data = [i for subset in y_sets for i in subset]

    y_data, std, y_mean = normalize_y(y_sets)
    y_data = np.reshape(y_data, (-1,1440))
    x_data = np.array(x_sets, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    examples = list(zip(x_data, y_data))
    random.shuffle(examples)

    shuffled_x = []
    shuffled_y = []
    for i in examples:
        shuffled_x.append(i[0])
        shuffled_y.append(i[1])

    x_data = [item for subitem in shuffled_x for item in subitem]
    y_data = [item for subitem in shuffled_y for item in subitem]
    y_data = np.reshape(y_data, (-1,1))

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32), np.array(std, dtype=np.float32), np.array(y_mean, dtype=np.float32)
