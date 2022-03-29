path = "/home/Alfredo/input/1min_real_sept_oct_nov_dec2019.csv"

def load_hyperparameters(appliance):

    if str(appliance) == "refrigerator1":
        config = dict(
            appliance=str(appliance),
            epochs=2000,
            batch_size=2500,
            learning_rate=3.3180899699083407e-04,#e-06,
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            hidden_size_1=64, #74
            hidden_size_2=85, #223,
            fc1=211,#52,
            fc2=109,
            weight_decay=0,
            #weight_decay=0.08044407443013193,
            #window_size=136
            window_size=249
        )

    if str(appliance) == "drye1":
        # hyperparameters for drye1
        config = dict(
            appliance=str(appliance),
            epochs = 1500,
            batch_size = 250,
            learning_rate = 0.00022534369918008793,
            in_channels = 1,
            out_channels = 16,
            kernel_size = 7,
            hidden_size_1 = 44,
            hidden_size_2 = 129,
            fc1 = 340,
            fc2 = 22,
            weight_decay = 0.021543979275305797,
            window_size = 169
        )

    if str(appliance) == "waterheater1":
        #hyperparameters for waterheater1
        config = dict(
            appliance=str(appliance),
            epochs = 200,
            batch_size = 100,
            learning_rate = 0.0006067466151695117,
            in_channels = 1,
            out_channels = 16,
            kernel_size = 7,
            hidden_size_1 = 170,
            hidden_size_2 = 150,
            fc1 = 237,
            fc2 = 152,
            weight_decay = 0.07172563654435048,
            window_size = 167
        )

    if str(appliance) == "clotheswasher1":
        #hyperparameters for clotheswasher1
        config = dict(
            appliance=str(appliance),
            epochs=200,
            batch_size=100,
            learning_rate=0.0006070989478443628,
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            hidden_size_1=65,
            hidden_size_2=142,
            fc1=231,
            fc2=105,
            weight_decay=0.10735237530643969,
            window_size=184
        )

    if str(appliance) == "dishwasher1":
        #hyperparameters for dishwasher1
        config = dict(
            appliance=str(appliance),
            epochs=200,
            batch_size=100,
            learning_rate=0.00038521759959249434,
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            hidden_size_1=132,
            hidden_size_2=136,
            fc1=183,
            fc2=167,
            weight_decay=0.051832935309959846,
            window_size=67
        )

    return config


# original config
# config = dict(
#     epochs=750,
#     batch_size=500,
#     learning_rate=0.0004,
#     in_channels=1,
#     out_channels=16,
#     kernel_size=7,
#     hidden_size_1=32,
#     hidden_size_2=200,
#     fc1=200,
#     fc2=100)
