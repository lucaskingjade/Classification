from Classification.models.rnn_classifier import *
from Classification.data.dataset import Emilya_Dataset

rnn = RNN_Classifier(hidden_dim_list=[100, 100], activation_list=['tanh', 'tanh'],
                     dropout_dis_list=[0.0, 0.0], batch_size=300,
                     max_epoch=200, optimiser='rmsprop', lr=0.001, decay=0.0,
                     momentum=0.0)
data_obj = Emilya_Dataset()
rnn.training(data_obj)
