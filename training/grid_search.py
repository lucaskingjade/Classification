

from Classification.models.rnn_classifier import *
from Classification.data.dataset import Emilya_Dataset
from Classification.models.LSTM_Classifier import lstm_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def data_generator(train_X,valid_X):
    #yield the indices for training and valid sets
    indices_train = np.arange(start=0,stop=len(train_X),step=1,dtype=int)
    indices_valid = np.arange(start=len(train_X),stop=(len(train_X)+len(valid_X)),step=1,dtype=int)
    yield  indices_train,indices_valid
# rnn = RNN_Classifier(hidden_dim_list=[100, 100], activation_list=['tanh', 'tanh'],
#                      dropout_dis_list=[0.0, 0.0], batch_size=300,
#                      max_epoch=200, optimiser='rmsprop', lr=0.001, decay=0.0,
#                      momentum=0.0)
data_obj = Emilya_Dataset()
# rnn.training(data_obj)


#classifier = KerasClassifier(build_fn=lstm_model)
# max_len=[200]
# dof=[70]
embd_dim=[2]
hidden_dim_list=[[100,20]]
activation_list=[['tanh','tanh']]
batch_size=[300]
max_epoch=[20]
params = dict(embd_dim=embd_dim,batch_size=batch_size,max_epoch=max_epoch,
            hidden_dim_list=hidden_dim_list,activation_list=activation_list)
params.update(data_obj=[data_obj])
X = np.concatenate((data_obj.train_X,data_obj.valid_X),axis=0)
data_indices = data_generator(data_obj.train_X,data_obj.valid_X)
grid_search = GridSearchCV(RNN_Classifier(),param_grid=params,cv=data_indices)
print "data_obj: train_X:{0}, train_Y2:{1},train_Y1:{2}".format(data_obj.train_X.shape,data_obj.train_Y2.shape,data_obj.train_Y1.shape)
grid_search.fit(X=X)

