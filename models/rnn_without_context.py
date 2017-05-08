#This is a classifier model without any context information


from keras.models import Model
from keras.layers import LSTM,Dense,merge,Input,Embedding,RepeatVector,Reshape
import numpy as np
from keras.optimizers import SGD,RMSprop
from sklearn.base import BaseEstimator
from Classification.data.Emilya_Dataset.EmilyData_utils import get_label_by_name
from keras.utils.np_utils import to_categorical

class RNN_without_Context(BaseEstimator):

    def __init__(self,
                 hidden_dim_list=[100,100],
                 activation_list=['tanh','tanh'],
                 batch_size=300,max_epoch=200,
                 optimiser='rmsprop',
                 lr=0.001,decay=0.0,
                 momentum=0.0,data_obj=None,remove_pairs=False,
                 rm_activities = ["Simple Walk"],
                rm_emotions = ["Panic Fear"],set_up_data=True):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        print args.keys()
        self.save_configuration(args)

    def set_up_model(self):
        self.rnn = self.rnn()
        self.compile()
        #summary of model
        self.rnn.summary()

    def compile(self):
        if self.optimiser=='sgd':
            optimizer = SGD(lr=self.lr)
        elif self.optimiser=='rmsprop':
            optimizer = RMSprop(lr=self.lr)
        else:
            pass
        self.rnn.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    def save_configuration(self,arguments):
        with open('meta_data.txt','w') as file:
            file.writelines("========Meta Data========\r\n")
            for key in arguments.keys():
                file.writelines(key+' : '+ str(arguments[key])+'\r\n')
            file.writelines('===========END===========\r\n')


    def set_up_dataset(self,dataset_obj):
        #save datasource to meta_data.txt
        with open('meta_data.txt','a') as file:
            file.writelines(dataset_obj.file_source)

        if hasattr(dataset_obj,'max_len'):
            self.max_len = dataset_obj.max_len
        else:
            raise ValueError("Attribute 'max_len' doesn't exist in dataset obj ")

        if hasattr(dataset_obj,'dof'):
            self.dof = dataset_obj.dof
        else:
            raise ValueError("Attribute 'dof' doesn't exist in dataset obj ")

        self.train_X = dataset_obj.train_X
        self.train_Y1 = dataset_obj.train_Y1
        self.train_Y2 = dataset_obj.train_Y2
        self.train_Y3 = dataset_obj.train_Y3

        self.valid_X = dataset_obj.valid_X
        self.valid_Y1 = dataset_obj.valid_Y1
        self.valid_Y2 = dataset_obj.valid_Y2
        self.valid_Y3 = dataset_obj.valid_Y3

        self.test_X = dataset_obj.test_X
        self.test_Y1 = dataset_obj.test_Y1
        self.test_Y2 = dataset_obj.test_Y2
        self.test_Y3 = dataset_obj.test_Y3

        #remove {"Simple Walk","Panic Fear"} pairs from training set
        if self.remove_pairs == True:
            # self.activities = ["Simple Walk"]
            # emotions = ["Panic Fear"]

            activities = self.rm_activities
            emotions = self.rm_emotions
            print "remove paris:{0},{1}".format(activities,emotions)
            self.train_X, self.train_Y1,self.train_Y2,\
            self.train_missing_X,self.train_missing_Y1,self.train_missing_Y2\
                = self.remove_pairs_fn(self.train_X,self.train_Y1,self.train_Y2,activities,emotions)
            self.valid_X, self.valid_Y1,self.valid_Y2,\
            self.valid_missing_X,self.valid_missing_Y1,self.valid_missing_Y2\
                = self.remove_pairs_fn(self.valid_X, self.valid_Y1, self.valid_Y2, activities,
                                                                     emotions)
            self.test_X, self.test_Y1,self.test_Y2,\
            self.test_missing_X,self.test_missing_Y1,self.test_missing_Y2\
                = self.remove_pairs_fn(self.test_X, self.test_Y1, self.test_Y2, activities,
                                                                     emotions)
            #concate valid data
            print "shape of valid_x and valid_missing_X is {0},{1}".format(self.valid_X.shape,self.valid_missing_X.shape)
            self.valid_X = np.concatenate((self.valid_X,self.valid_missing_X),axis=0)
            self.valid_Y1 = np.concatenate((self.valid_Y1,self.valid_missing_Y1),axis=0)
            self.valid_Y2 = np.concatenate((self.valid_Y2, self.valid_missing_Y2), axis=0)
            del self.valid_missing_X, self.valid_missing_Y1,self.valid_missing_Y2
        else:

            #convert Y1 to categorical vector
            self.train_Y1 = to_categorical(self.train_Y1,8)
            self.valid_Y1 = to_categorical(self.valid_Y1,8)
            self.test_Y1 = to_categorical(self.test_Y1, 8)


    def remove_pairs_fn(self,X,Y1,Y2,activities,emotions):
        missing_X = []
        missing_Y1 = []
        missing_Y2 = []
        new_X = []
        new_Y1 = []
        new_Y2 = []

        for activity,emotion in zip(activities,emotions):
            #get label from name
            act_label= get_label_by_name(activity,whichlabel=1)
            em_label = get_label_by_name(emotion,whichlabel=2)
            index1 = np.where(Y1==act_label)[0]
            index2 = np.where(Y2 == em_label)[0]
            index = list(set(index1).intersection(index2))
            missing_X.extend(X[index])
            missing_Y1.extend(Y1[index])
            missing_Y2.extend(Y2[index])
            mask = np.ones(len(X), np.bool)
            mask[index] = 0
            new_X.extend(X[mask])
            new_Y1.extend(Y1[mask])
            new_Y2.extend(Y2[mask])
        new_X = np.asarray(new_X)
        new_Y1 = np.asarray(new_Y1)
        new_Y2 = np.asarray(new_Y2)
        missing_X = np.asarray(missing_X)
        missing_Y1 = np.asarray(missing_Y1)
        missing_Y2 = np.asarray(missing_Y2)
        new_Y1 = to_categorical(new_Y1,8)
        missing_Y1 = to_categorical(missing_Y1,8)
        print "shape of new_Y1 is {}".format(new_Y1.shape)
        print "shape of missing_Y1 is {}".format(missing_Y1.shape)
        return new_X,new_Y1,new_Y2,missing_X,missing_Y1,missing_Y2

    def rnn(self):
        input = Input(shape = (self.max_len,self.dof),name='input')
        encoded = input
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_list, self.activation_list)):
            if i == len(self.hidden_dim_list) - 1:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=False)(encoded)
            else:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

        encoded = Dense(output_dim=8, activation='softmax')(encoded)
        return Model(input=input, output=encoded, name='RNN')

    def batch_generator(self,iterable1,iterable2,batch_size=1,shuffle=False):
        l = len(iterable1)
        if shuffle ==True:
            indices = np.random.permutation(len(iterable1))
        else:
            indices = np.arange(0,stop=len(iterable1))
        for ndx in range(0,l,batch_size):
            cur_indices = indices[ndx:min(ndx+batch_size,l)]
            yield  iterable1[cur_indices],iterable2[cur_indices]

    def init_loss_history_list(self):

        self.loss_history = {"loss_train": [],
                             "accuracy_train": [],
                             "loss_valid": [],
                             "accuracy_valid": [],
                             }


    def training(self,data_obj=None):
        if data_obj!=None and self.set_up_data == True:
            self.set_up_dataset(data_obj)
            self.set_up_data = False
        print "shape of missing training set is {}".format(self.train_missing_X.shape)
        print "shape of training set is {}".format(self.train_X)
        self.set_up_model()
        self.init_loss_history_list()
        print "training set size: %d" % len(self.train_X)
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))

            #self.train_Y2= self.train_Y2.reshape(self.train_Y2.shape[0],1,1)
            print "shape of train_Y1 is {}".format(self.train_Y1.shape)
            self.training_loop(self.train_X,self.train_Y1,batch_size=self.batch_size)

    def training_loop(self, X,Y, batch_size):
        # batch generator
        self.data_generator = self.batch_generator(X, Y,batch_size=batch_size,shuffle=True)

        for X_batch,Y_batch in self.data_generator:
            #print "shape of Y_batch is {}".format(Y_batch.shape)
            self.rnn.train_on_batch(x=X_batch,y=Y_batch)


    def fit(self,X,y=None):
        # if self.data_obj !=None:
        #     print "Set up data_obj in __init__ function"
        #     self.set_up_dataset(self.data_obj)
        print "============Parameters==========="
        print "hidden_dim_list={}".format(self.hidden_dim_list)
        print "activation_list={}".format(self.activation_list)
        print "batch_size={}".format(self.batch_size)
        print "max_epoch={}".format(self.max_epoch)
        print "optimiser={}".format(self.optimiser)
        print "lr={}".format(self.lr)
        print "decay={}".format(self.decay)
        print "momentum={}".format(self.momentum)
        print "================================="
        self.training(self.data_obj)

    def score(self,X,y=None):
        loss,accuracy = self.rnn.evaluate(self.valid_X,y=self.valid_Y1,batch_size=1000,verbose=0)
        print 'accuracy is {}%'.format(accuracy*100.)
        print "loss is {}".format(loss)
        return accuracy

if __name__=='__main__':
    rnn = RNN_without_Context(hidden_dim_list=[100,20],activation_list=['tanh','tanh'],
                   batch_size=300,
                   max_epoch=200,optimiser='rmsprop',lr=0.001,decay=0.0,
                   momentum=0.0)
    from Classification.data.dataset import Emilya_Dataset
    data_obj = Emilya_Dataset()
    rnn.training(data_obj)