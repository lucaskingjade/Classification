

#from Classification.models.rnn_classifier import *
from Classification.data.dataset import Emilya_Dataset
from Classification.models.LSTM_Classifier import lstm_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# rnn = RNN_Classifier(hidden_dim_list=[100, 100], activation_list=['tanh', 'tanh'],
#                      dropout_dis_list=[0.0, 0.0], batch_size=300,
#                      max_epoch=200, optimiser='rmsprop', lr=0.001, decay=0.0,
#                      momentum=0.0)
data_obj = Emilya_Dataset()
# rnn.training(data_obj)


classifier = KerasClassifier(build_fn=lstm_model)
max_len=[200]
dof=[70]
embd_dim=[2]
hidden_dim_list=[[100,20]]
activation_list=[['tanh','tanh']]
params = dict(max_len=max_len,dof=dof,embd_dim=embd_dim,
            hidden_dim_list=hidden_dim_list,activation_list=activation_list)

grid_search = GridSearchCV(classifier,param_grid=params,cv=5)
grid_search.fit(X=[data_obj.train_X,data_obj.train_Y2],y=data_obj.train_Y1)
