#This is a simple implementation of LSTM
from keras.layers import Input,Embedding,RepeatVector,Reshape,LSTM,merge,Dense
from keras.models import Model
from keras.optimizers import SGD,RMSprop
def lstm_model(max_len=200,dof=70,embd_dim=2,
               hidden_dim_list=[100,20],activation_list=['tanh','tanh'],
               optimizer ='sgd',lr='0.01',momentum=0.0):
    input = Input(shape = (max_len,dof),name='input')
    label_input  =Input(shape=(1,),name='label_input')
    embd_label = Embedding(input_dim=8,output_dim=embd_dim)(label_input)
    embd_label = Reshape(target_shape=(embd_dim,))(embd_label)
    embd_label = RepeatVector(max_len)(embd_label)
    encoded = merge([input, embd_label], mode='concat',concat_axis=2)
    for i, (dim, activation) in enumerate(zip(hidden_dim_list, activation_list)):
        if i ==len(hidden_dim_list)-1:
            encoded = LSTM(output_dim=dim, activation=activation, return_sequences=False)(encoded)
        else:
            encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

    encoded = Dense(output_dim=8, activation='sigmoid')(encoded)
    model = Model(input=[input, label_input], output = encoded, name='Encoder')

    if optimizer=='sgd':
        optimizer_model = SGD(lr=lr,momentum=momentum)
    elif optimizer=='rmsprop':
        optimizer_model = RMSprop(lr=lr)

    else:
        raise ValueError('No such kind optimizer')

    #compile model
    model.compile(optimizer=optimizer_model,loss='binary_crossentropy',metrics='accuracy')
    model.summary()
    return model