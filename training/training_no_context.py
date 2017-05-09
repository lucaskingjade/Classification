from Classification.models.rnn_without_context import *
from Classification.data.dataset import Emilya_Dataset

rnn = RNN_without_Context(hidden_dim_list=[100, 20], activation_list=['tanh', 'tanh'],
                          batch_size=300,
                          max_epoch=200, optimiser='rmsprop', lr=0.005, decay=0.0,
                          momentum=0.0,remove_pairs=True,
                          rm_activities=['Simple Walk','Sitting Down', 'Move Books'],
                          rm_emotions=['Panic Fear','Sadness', 'Joy'])
data_obj = Emilya_Dataset()
rnn.training(data_obj)
