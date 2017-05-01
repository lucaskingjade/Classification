from keras.models import Model
from keras.layers import LSTM,Dense,merge,Input,Embedding,RepeatVector,Reshape
import numpy as np
from keras.optimizers import SGD,RMSprop
class RNN_Classifier():

    def __init__(self,hidden_dim_list=[100,100],
                 activation_list=['tanh','tanh'],
                 dropout_dis_list=[0.0,0.0],
                 batch_size=300,max_epoch=200,
                 optimiser='rmsprop',
                 lr=0.001,decay=0.0,
                 momentum=0.0):

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
        self.rnn.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
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

    def rnn(self):
        input = Input(shape = (self.max_len,self.dof),name='input')
        label_input  =Input(shape=(1,),name='label_input')
        embd_label = Embedding(input_dim=8,output_dim=2)(label_input)
        embd_label = Reshape(target_shape=(2,))(embd_label)
        embd_label = RepeatVector(self.max_len)(embd_label)
        encoded = merge([input, embd_label], mode='concat',concat_axis=2)
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_list, self.activation_list)):
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
        encoded = LSTM(output_dim=10, activation='tanh',return_sequences=False)(encoded)
        encoded = Dense(output_dim=1, activation='sigmoid')(encoded)
        return Model(input=[input, label_input], output = encoded, name='Encoder')

    def batch_generator(self,iterable1,iterable2,iterable3,batch_size=1,shuffle=False):
        l = len(iterable1)
        if shuffle ==True:
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
                             }


    def training(self,data_obj):
        self.set_up_dataset(data_obj)
        self.set_up_model()
        self.init_loss_history_list()
        print "training set size: %d" % len(self.train_X)
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))

            #self.train_Y2= self.train_Y2.reshape(self.train_Y2.shape[0],1,1)
            self.training_loop(self.train_X,self.train_Y1,self.train_Y2,batch_size=self.batch_size)

    def training_loop(self, X,Y,additional_labels, batch_size):
        # batch generator
        self.data_generator = self.batch_generator(X, Y, additional_labels, batch_size=batch_size)
        for X_batch,Y_batch, add_label in self.data_generator:
            self.rnn.train_on_batch(x=[X_batch,add_label],y=Y_batch)


if __name__=='__main__':
    rnn = RNN_Classifier(hidden_dim_list=[100,100],activation_list=['tanh','tanh'],
                   dropout_dis_list=[0.0,0.0],batch_size=300,
                   max_epoch=200,optimiser='rmsprop',lr=0.001,decay=0.0,
                   momentum=0.0)
    from Classification.data.dataset import Emilya_Dataset
    data_obj = Emilya_Dataset()
    rnn.training(data_obj)