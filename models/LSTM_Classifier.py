#This is a simple implementation of LSTM
from keras.layers import Input,Embedding,RepeatVector,Reshape,LSTM,merge,Dense
from keras.models import Model
def lstm_model(max_len=200,dof=70,embd_dim=2,
               hidden_dim_list=[100,20],activation_list=['tanh','tanh'],):
    input = Input(shape = (max_len,dof),name='input')
    label_input  =Input(shape=(1,),name='label_input')
    embd_label = Embedding(input_dim=8,output_dim=embd_dim)(label_input)
    embd_label = Reshape(target_shape=(embd_dim,))(embd_label)
    embd_label = RepeatVector(max_len)(embd_label)
    encoded = merge([input, embd_label], mode='concat',concat_axis=2)
    for i, (dim, activation) in enumerate(zip(hidden_dim_list, activation_list)):
        encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
        if i ==len(hidden_dim_list)-1:
            encoded = LSTM(output_dim=dim, activation=activation, return_sequences=False)(encoded)
    encoded = Dense(output_dim=1, activation='sigmoid')(encoded)
    return Model(input=[input, label_input], output = encoded, name='Encoder')
