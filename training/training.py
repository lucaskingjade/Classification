from Classification.models.rnn_classifier import *
from Classification.data.dataset import Emilya_Dataset
from keras.constraints import max_norm,unit_norm,nonneg
constraint = unit_norm(axis=-1)
rnn_model = RNN_Classifier(embd_dim=3, hidden_dim_list=[100, 20], activation_list=['tanh', 'tanh'],
                        batch_norm_list=[False,False],
                     batch_size=300,
                     max_epoch=200, optimiser='adam', lr=0.005, decay=0.0,
                     momentum=0.0,remove_pairs=True,
                     rm_activities=['Simple Walk','Sitting Down', 'Move Books'],
                     rm_emotions=['Panic Fear','Sadness', 'Joy'],
                           constant_initializer=True,constant_value=0.01,
                           constraint=constraint,threshold_embd=None)
data_obj = Emilya_Dataset()
rnn.training(data_obj)
##initialise the weight from other model
# model_path = './rnn.h5'
# rnn_model.rnn.load_weights(model_path,by_name=True)
#
#
#
# rnn_model.training(data_obj)
