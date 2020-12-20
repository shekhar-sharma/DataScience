  
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from MvLDAN import MvLDAN_gneral
def create_mnist_full_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)

    models.append(Sequential())
    models[0].add(Dense(256, input_shape=input_size[0], activation='relu'))
    models[0].add(Dense(256, activation='relu'))
    models[0].add(Dense(128, activation='relu'))
    models[0].add(Dense(output_size))

    models.append(Sequential())
    models[1].add(Dense(256, input_shape=input_size[1], activation='relu'))
    models[1].add(Dense(256, activation='relu'))
    models[1].add(Dense(128, activation='relu'))
    models[1].add(Dense(output_size))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output) 
