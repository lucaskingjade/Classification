from keras.models import Model
from keras.layers import LSTM,Dense,merge,Input,Embedding,RepeatVector,Reshape,BatchNormalization
import numpy as np
from keras.optimizers import SGD,RMSprop,Adam
from sklearn.base import BaseEstimator
from Classification.data.Emilya_Dataset.EmilyData_utils import get_label_by_name
from keras.utils.np_utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.initializers import Constant

class RNN_Classifier(BaseEstimator):

    def __init__(self,embd_dim=2,
                 hidden_dim_list=[100,100],
                 activation_list=['tanh','tanh'],
                 batch_norm_list=[False,False],
                 batch_size=300,max_epoch=200,
                 optimiser='rmsprop',
                 lr=0.001,decay=0.0,
                 momentum=0.0,nesterov=False,
                 data_obj=None,remove_pairs=False,
                 rm_activities = ["Simple Walk"],
                rm_emotions = ["Panic Fear"],constant_initializer=False,
                 constant_value=0.01,constraint=None,threshold_embd=None):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        print args.keys()
        self.save_configuration(args)
        self.mark = False

    def set_up_model(self):
        self.rnn = self.RNN()
        self.compile()
        #summary of model
        self.rnn.summary()

    def compile(self):
        if self.optimiser=='sgd':
            optimizer = SGD(lr=self.lr,momentum=self.momentum,decay=self.decay,nesterov=self.nesterov)
        elif self.optimiser=='rmsprop':
            optimizer = RMSprop(lr=self.lr,decay=self.decay)
        elif self.optimiser=='adam':
            optimizer = Adam(lr=self.lr,decay=self.decay)
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
        index = []
        for activity, emotion in zip(activities, emotions):
            # get label from name
            act_label = get_label_by_name(activity, whichlabel=1)
            em_label = get_label_by_name(emotion, whichlabel=2)
            index1 = np.where(Y1 == act_label)[0]
            index2 = np.where(Y2 == em_label)[0]
            cur_index = list(set(index1).intersection(index2))
            index.extend(cur_index)
        # print removed index
        print "shape of removed indexes are {}".format(len(index))
        print "shape of X is {}".format(X.shape)
        missing_X = X[index]
        missing_Y1 = Y1[index]
        missing_Y2 = Y2[index]
        mask = np.ones(len(X), np.bool)
        mask[index] = 0
        new_X = X[mask]
        new_Y1 = Y1[mask]
        new_Y2 = Y2[mask]
        new_X = np.asarray(new_X)
        new_Y1 = np.asarray(new_Y1)
        new_Y2 = np.asarray(new_Y2)
        missing_X = np.asarray(missing_X)
        missing_Y1 = np.asarray(missing_Y1)
        missing_Y2 = np.asarray(missing_Y2)
        new_Y1 = to_categorical(new_Y1, 8)
        missing_Y1 = to_categorical(missing_Y1, 8)
        assert missing_X.shape[0] == len(missing_Y1) and missing_X.shape[0] ==len(missing_Y2)
        assert new_X.shape[0]==len(new_Y1) and len(new_Y1) == len(new_Y2)
        assert missing_X.shape[0]+new_X.shape[0]==X.shape[0]
        print "shape of new_Y1 is {}".format(new_Y1.shape)
        print "shape of missing_Y1 is {}".format(missing_Y1.shape)
        return new_X, new_Y1, new_Y2, missing_X, missing_Y1, missing_Y2

    def RNN(self):
        input = Input(shape = (self.max_len,self.dof),name='input')
        label_input = Input(shape=(1,), name='label_input')
        if self.constant_initializer == True:
            init_constant = Constant(value=self.constant_value)
            embd_label = Embedding(input_dim=8, output_dim=self.embd_dim,
                                   embeddings_initializer=init_constant,
                                   embeddings_constraint=self.constraint,name='embedding_1',trainable=True)(label_input)
        else:
            embd_label = Embedding(input_dim=8, output_dim=self.embd_dim,
                                   embeddings_constraint=self.constraint,trainable=True)(label_input)

        embd_label = Reshape(target_shape=(self.embd_dim,))(embd_label)
        embd_label = RepeatVector(self.max_len)(embd_label)
        encoded = merge([input, embd_label], mode='concat', concat_axis=2)
        for i, (dim, activation,is_batch_norm) in enumerate(zip(self.hidden_dim_list, self.activation_list,self.batch_norm_list)):
            if i == len(self.hidden_dim_list) - 1:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=False)(encoded)
                if is_batch_norm==True:
                    encoded = BatchNormalization(axis=-1)(encoded)
            else:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
                if is_batch_norm==True:
                    encoded = BatchNormalization(axis=-1)(encoded)


        encoded = Dense(output_dim=8, activation='softmax')(encoded)
        return Model(input=[input, label_input], output=encoded, name='RNN')

    def batch_generator(self,iterable1,iterable2,iterable3,batch_size=1,shuffle=False):
        l = len(iterable1)
        if shuffle ==True:
            #np.random.seed(1235)
            indices = np.random.permutation(len(iterable1))
        else:
            indices = np.arange(0,stop=len(iterable1))
        for ndx in range(0,l,batch_size):
            cur_indices = indices[ndx:min(ndx+batch_size,l)]
            yield  iterable1[cur_indices],iterable2[cur_indices],iterable3[cur_indices]

    def init_loss_history_list(self):

        self.loss_history = {"loss_train": [],
                             "accuracy_train": [],
                             "loss_valid": [],
                             "accuracy_valid": [],
                             "loss_test":[],
                             "accuracy_test":[],
                             "loss_missing_test":[],
                             "accuracy_missing_test":[]
                             }
        self.embedding_history = []


    def print_loss_history(self):
        for loss_key in sorted(self.loss_history.keys()):
            print "%s:%f"%(loss_key,self.loss_history[loss_key][-1])


    def compute_loss_history(self,dataset='training'):
        if dataset == 'training':
            X = self.train_X
            Y1 = self.train_Y1
            Y2 = self.train_Y2
            str_loss = "loss_train"
            str_accuracy = "accuracy_train"
        elif dataset=='valid':
            X = self.valid_X
            Y1 = self.valid_Y1
            Y2 = self.valid_Y2
            str_loss = "loss_valid"
            str_accuracy = "accuracy_valid"
        elif dataset =='test':
            X = self.test_X
            Y1 = self.test_Y1
            Y2 = self.test_Y2
            str_loss = "loss_test"
            str_accuracy = "accuracy_test"
        elif dataset =='test_missing':
            X = self.test_missing_X
            Y1 = self.test_missing_Y1
            Y2 = self.test_missing_Y2
            str_loss = "loss_missing_test"
            str_accuracy = "accuracy_missing_test"
        else:
            raise ValueError()

        loss,accuracy = self.rnn.evaluate([X,Y2],y=Y1,batch_size = 1000,verbose =0)
        self.loss_history[str_loss].append(loss)
        self.loss_history[str_accuracy].append(accuracy)

    def plot_loss(self):
        # plot mse
        plt.figure(figsize=(5,5))
        legend_str =[]
        plt.plot(self.loss_history["loss_train"])
        legend_str.append('loss_train'+':%f' % self.loss_history["loss_train"][-1])
        plt.plot(self.loss_history["loss_valid"])
        legend_str.append('loss_valid'+':%f' % self.loss_history["loss_valid"][-1])
        plt.legend(legend_str)
        plt.savefig('./learning_curve.png')

        #plot accuracy of rnn
        plt.figure(figsize=(5,5))
        legend_str = []
        plt.plot(self.loss_history['accuracy_train'])
        legend_str.append('accuracy_train:%f'%self.loss_history['accuracy_train'][-1])
        plt.plot(self.loss_history['accuracy_valid'])
        legend_str.append('accuracy_valid:%f' % self.loss_history['accuracy_valid'][-1])
        plt.legend(legend_str,fontsize=10)
        plt.savefig('./accuracy_curve.png')

        # plot loss on test set and test_missing_set
        plt.figure(figsize=(5, 5))
        legend_str = []
        plt.plot(self.loss_history['loss_test'])
        legend_str.append('loss_test:%f' % self.loss_history['loss_test'][-1])
        plt.plot(self.loss_history['loss_missing_test'])
        legend_str.append(
            'loss_missing_test:%f' % self.loss_history['loss_missing_test'][-1])
        plt.legend(legend_str,fontsize=10)
        plt.savefig('./loss_test_and_missing.png')

        #plot accuracy on test and test_missing set
        plt.figure(figsize=(5, 5))
        legend_str = []
        plt.plot(self.loss_history['accuracy_test'])
        legend_str.append('accuracy_test:%f' % self.loss_history['accuracy_test'][-1])
        plt.plot(self.loss_history['accuracy_missing_test'])
        legend_str.append(
            'accuracy_missing_test:%f' % self.loss_history['accuracy_missing_test'][-1])
        plt.legend(legend_str, fontsize=10)
        plt.savefig('./accuracy_test_and_missing.png')

    def save_models(self):
        self.rnn.save_weights('rnn.h5')
        with open('rnn.yaml','w') as yaml_file:
            yaml_file.write(self.rnn.to_yaml())


    def training(self,data_obj=None):
        if data_obj!=None and self.mark==False:
            self.mark =True
            self.set_up_dataset(data_obj)
        self.set_up_model()
        self.init_loss_history_list()
        print "training set size: %d" % len(self.train_X)
        np.random.seed(1235)
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))
            #when accuracy of training set<85%,embedding trainable=False
            if self.threshold_embd != None:
                if epoch==0:
                    if self.rnn.get_layer('embedding_1').trainable != False:
                        self.rnn.get_layer('embedding_1').trainable = False
                        self.compile()
                else:
                    if self.loss_history['accuracy_train'][-1]>self.threshold_embd:
                        if self.rnn.get_layer('embedding_1').trainable != True:
                            self.rnn.get_layer('embedding_1').trainable=True
                            self.compile()
                        print "embedding trainable: {}".format(self.rnn.get_layer('embedding_1').trainable)
                    else:
                        if self.rnn.get_layer('embedding_1').trainable != False:
                            self.rnn.get_layer('embedding_1').trainable = False
                            self.compile()
                        print "embedding trainable: {}".format(self.rnn.get_layer('embedding_1').trainable)

            self.training_loop(self.train_X,self.train_Y1,self.train_Y2,batch_size=self.batch_size)
            #each epoch save the learned embedding
            cur_embedding = self.rnn.get_layer("embedding_1").get_weights()
            self.embedding_history.append(cur_embedding)
            #compute loss value on validation set
            self.compute_loss_history('training')
            self.compute_loss_history('valid')
            self.compute_loss_history('test')
            self.compute_loss_history('test_missing')
            #print loss value
            self.print_loss_history()

        #save embedding_history
        self.embedding_history= np.asarray(self.embedding_history)
        print "shape of embedding_history is {}".format(self.embedding_history.shape)
        np.savez('embedding_history.npz',self.embedding_history)
        #save loss and accuracy as npz file
        np.savez('loss_history.npz',self.loss_history)
        #plot training and valid set loss and accuracy
        self.plot_loss()
        #save model
        self.save_models()

    def training_loop(self, X,Y1,Y2, batch_size):
        # batch generator
        self.data_generator = self.batch_generator(X, Y1, Y2, batch_size=batch_size,shuffle=True)
        for X_batch,Y_batch, add_label in self.data_generator:

            self.rnn.train_on_batch(x=[X_batch,add_label],y=Y_batch)


    def fit(self,X,y=None):
        # if self.data_obj !=None:
        #     print "Set up data_obj in __init__ function"
        #     self.set_up_dataset(self.data_obj)
        print "============Parameters==========="
        print "embd_dim={}".format(self.embd_dim)
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
        # loss,accuracy = self.rnn.evaluate([self.valid_X,self.valid_Y2],y=self.valid_Y1,batch_size=1000,verbose=0)
        # print 'accuracy is {}%'.format(accuracy*100.)
        # print "loss is {}".format(loss)
        # return accuracy
        #find the most stable learning curve
        delta_abs = np.abs(np.asarray(self.loss_history['loss_train'][0:-1])-np.asarray(self.loss_history['loss_train'][1:]))
        mean_delta = np.mean(delta_abs)
        std_delta = np.std(delta_abs)
        print ("mean of delta_abs is",mean_delta)
        print ("std of delta_abs is", std_delta)
        return -1.*std_delta

if __name__=='__main__':
    rnn = RNN_Classifier(hidden_dim_list=[100,100],activation_list=['tanh','tanh'],
                   batch_size=300,
                   max_epoch=200,optimiser='rmsprop',lr=0.001,decay=0.0,
                   momentum=0.0)
    from Classification.data.dataset import Emilya_Dataset
    data_obj = Emilya_Dataset()
    rnn.training(data_obj)