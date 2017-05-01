'''
This file will process the raw motion sequence data by:
1. normalize dataset
Note: data bvh global position names:
Xposition,     Yposition,      Zposition
frontral axis,  vertical axis, horizontal axis

'''

__author__ = 'qiwang'
import numpy as np
import numpy.linalg as LA
import os


# def get_index_by_joint_name(joint_name,axis):
#     joint_list=


def get_label_by_name(name, whichlabel=1):

    Activity_name = ['Being Seated', 'Lift', 'Simple Walk', 'Throw', 'Knocking on the Door', 'Move Books',
                     'Sitting Down', 'Walk with smth in the Hands']
    Emotion_name = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']
    Actor_name = ['Brian','Elie','Florian','Hu','Janina','Jessica', 'Maria','Muriel','Robert','Sally','Samih','Tatiana']
    if whichlabel == 1:
        for i, act in enumerate(Activity_name):
            if act == name:
                return float(i)
    elif whichlabel == 2:
        for i, em in enumerate(Emotion_name):
            if em == name:
                return float(i)
    elif whichlabel ==3:
        for i,actor in enumerate(Actor_name):
            if actor == name:
                return float(i)


def load_data_from_npz(filename):
    # load dataset
    npz_obj = np.load(filename, 'r')
    if len(npz_obj.files) ==1:
        name = npz_obj.files[0]
        dataset = npz_obj[name]
    X = dataset[0]
    Y1 = dataset[1]  # activity labels
    Y2 = dataset[2]  # emotion labels
    Y3 = dataset[3]  # actor labels

    print "=======Summary DataSet======"
    print "X.shape: (%d, None,%d)" % (len(X), X[0].shape[1])
    print "activity label size: %d" % len(Y1)
    print "emotion label size: %d" % len(Y2)
    print "actor label size: %d" % len(Y3)
    print "============END============="
    return X,Y1,Y2,Y3
    ##split data sequences into 200 frames


def get_hdf5_file_name(window_width,shift_step,sampling_interval):
    root_path = os.getenv('Classification')
    path_dataset = root_path+'data/Emilya_Dataset/'
    print 'path_dataset: %s' % path_dataset
    if sampling_interval == None or sampling_interval == 1:
        str_sampling = '_without_sampling'
    else:
        str_sampling = '_sampling' + str(sampling_interval)

    len_frames = window_width / sampling_interval
    str_frames = str(len_frames)

    if shift_step == None:
        shift_step = window_width

    str_shift = str(shift_step)

    filename = path_dataset + 'alldataset_frame' + str_frames + '_shift' + str_shift + str_sampling + '.h5'
    return filename


def normalization_vertical_position_per_subject(X,Y3,actor_labels):
    num_actors = len(np.unique(actor_labels))
    print "the number of actors = %d"%num_actors
    max_vertical_position_list = []
    min_vertical_position_list = []
    for actor in actor_labels:
        indices = np.where(Y3==actor)[0]
        #print 'indices is {}'.format(indices)
        cur_dataset = X[indices].copy()#
        posture_array = []
        for cur_seq in cur_dataset:
            posture_array.extend(cur_seq)
        posture_array = np.asarray(posture_array,dtype=np.float32)
        print "shape of posture_array = {}".format(posture_array.shape)
        max_posture = np.max(posture_array,axis=0)
        min_posture = np.min(posture_array,axis=0)
        max_posture[0] =1.0
        max_posture[2:] = 1.0
        min_posture[0] = 0.0
        min_posture[2:] = 0.0
        #using max-min normalization
        max_vertical_position_list.append(max_posture)
        min_vertical_position_list.append(min_posture)
        for i, index in enumerate(indices):
           X[index] = (cur_dataset[i] - min_posture) / (max_posture - min_posture)
    max_vertical_position_list = np.asarray(max_vertical_position_list,dtype=np.float32).\
        reshape(len(max_vertical_position_list),max_vertical_position_list[0].shape[0])
    min_vertical_position_list = np.asarray(min_vertical_position_list,dtype=np.float32).\
        reshape(len(min_vertical_position_list),min_vertical_position_list[0].shape[0])
    return X, max_vertical_position_list, min_vertical_position_list

def normalization_velocity(delta_X,max_velocity_vector = None, min_velocity_vector = None, return_max_min=False):
    X_new = []
    if max_velocity_vector ==None or min_velocity_vector ==None:
        X_concate = []
        for seq in delta_X:
            X_concate.extend(seq)
        print len(X_concate)
        print len(X_concate[1])
        max_velocity_vector = np.max(X_concate, axis=0)
        min_velocity_vector = np.min(X_concate, axis=0)
    # print max_vector-min_vector
    for i in range(len(delta_X)):
        X_new.append((delta_X[i] - min_velocity_vector) / (max_velocity_vector- min_velocity_vector))

    X_new = np.asarray(X_new,dtype=np.float32)
    if return_max_min ==True:
        return X_new, max_velocity_vector, min_velocity_vector
    else:
        return X_new


def normalization_joint_angles(X):
    X_new = []
    X_concate = []
    for seq in X:
        X_concate.extend(seq)
    print len(X_concate)
    print len(X_concate[1])
    max_vector = np.max(X_concate, axis=0)
    min_vector = np.min(X_concate, axis=0)
    max_vector[0:3] = 1.0
    min_vector[0:3] = 0.0
    # print max_vector-min_vector
    for i in range(len(X)):
        X_new.append((X[i] - min_vector) / (max_vector - min_vector))
    # X_new = np.asarray(X_new,dtype=np.float32)
    return X_new, max_vector, min_vector


def split_dataset(X, Y1, Y2, Y3, ratio_train=0.6, ratio_valid=0.2, shuffle=True):
    # split data as training set and validation set and test set.
    print "begin to split dataset into training set(%f),validation set(%f),test set(%f)"\
          %(ratio_train,ratio_valid,1-ratio_train-ratio_valid)

    train_X = []
    train_Y1 = []
    train_Y2 = []
    train_Y3 = []
    valid_X = []
    valid_Y1 = []
    valid_Y2 = []
    valid_Y3 = []
    test_X = []
    test_Y1 = []
    test_Y2 = []
    test_Y3 = []

    y1 = np.unique(Y1)
    y2 = np.unique(Y2)
    y3 = np.unique(Y3)
    print 'label index of activations is{}'.format(y1)
    print 'label index of emotions is{}'.format(y2)
    print 'label index of actors is {}'.format(y3)
    np.random.seed(345)
    nb_activity = len(np.unique(Y1))
    nb_emotion = len(np.unique(Y2))
    nb_actor = len(np.unique(Y3))
    for i in range(nb_activity):
        index_act = np.where(Y1 == y1[i])[0]
        print "index_act {}".format(index_act)
        for j in range(nb_emotion):
            index_em = np.where(Y2 == y2[j])[0]
            for k in range(nb_actor):
                index_actor = np.where(Y3==y3[k])[0]
                index1 = set(index_act).intersection(index_em)
                index = set(index1).intersection(index_actor)
                cur_X = [X[c] for c in index]
                cur_Y1 = [Y1[c] for c in index]
                cur_Y2 = [Y2[c] for c in index]
                cur_Y3 = [Y3[c] for c in index]

                indices = np.random.permutation(len(cur_X))
                num_train = int(len(cur_X) * ratio_train)
                num_valid = int(len(cur_X) * ratio_valid)
                # num_test = len(cur_X) - num_train - num_valid
                train_idx = indices[:num_train]
                valid_idx = indices[num_train:(num_train + num_valid)]
                test_idx = indices[(num_train + num_valid):]
                for c in train_idx:
                    train_X.append(cur_X[c])
                    train_Y1.append(cur_Y1[c])
                    train_Y2.append(cur_Y2[c])
                    train_Y3.append(cur_Y3[c])
                for c in valid_idx:
                    valid_X.append(cur_X[c])
                    valid_Y1.append(cur_Y1[c])
                    valid_Y2.append(cur_Y2[c])
                    valid_Y3.append(cur_Y3[c])
                for c in test_idx:
                    test_X.append(cur_X[c])
                    test_Y1.append(cur_Y1[c])
                    test_Y2.append(cur_Y2[c])
                    test_Y3.append(cur_Y3[c])

    np.random.seed(100)
    if shuffle == True:
        # shuffle training examples.
        tmp_X = []
        tmp_Y1 = []
        tmp_Y2 = []
        tmp_Y3 = []
        indices = np.random.permutation(len(train_X))
        for c in indices:
            tmp_X.append(train_X[c])
            tmp_Y1.append(train_Y1[c])
            tmp_Y2.append(train_Y2[c])
            tmp_Y3.append(train_Y3[c])
        del train_X,train_Y1,train_Y2,train_Y3

        train_X = tmp_X
        train_Y1 = tmp_Y1
        train_Y2 = tmp_Y2
        train_Y3 = tmp_Y3

        # shuffle validation set
        tmp_X = []
        tmp_Y1 = []
        tmp_Y2 = []
        tmp_Y3 = []
        indices = np.random.permutation(len(valid_X))
        for c in indices:
            tmp_X.append(valid_X[c])
            tmp_Y1.append(valid_Y1[c])
            tmp_Y2.append(valid_Y2[c])
            tmp_Y3.append(valid_Y3[c])

        del valid_X,valid_Y1,valid_Y2,valid_Y3
        valid_X = tmp_X
        valid_Y1 = tmp_Y1
        valid_Y2 = tmp_Y2
        valid_Y3 = tmp_Y3
        # shuffle test set
        tmp_X = []
        tmp_Y1 = []
        tmp_Y2 = []
        tmp_Y3 = []
        indices = np.random.permutation(len(test_X))
        for c in indices:
            tmp_X.append(test_X[c])
            tmp_Y1.append(test_Y1[c])
            tmp_Y2.append(test_Y2[c])
            tmp_Y3.append(test_Y3[c])

        del test_X,test_Y1,test_Y2,test_Y3
        test_X = tmp_X
        test_Y1 = tmp_Y1
        test_Y2 = tmp_Y2
        test_Y3 = tmp_Y3
    train_Y1 = np.asarray(train_Y1, dtype=np.float32)
    train_Y2 = np.asarray(train_Y2, dtype=np.float32)
    train_Y3 = np.asarray(train_Y3, dtype=np.float32)
    valid_Y1 = np.asarray(valid_Y1, dtype=np.float32)
    valid_Y2 = np.asarray(valid_Y2, dtype=np.float32)
    valid_Y3 = np.asarray(valid_Y3, dtype=np.float32)
    test_Y1 = np.asarray(test_Y1, dtype=np.float32)
    test_Y2 = np.asarray(test_Y2, dtype=np.float32)
    test_Y3 = np.asarray(test_Y3, dtype=np.float32)
    print 'finish splitting datasets'
    return (train_X, train_Y1, train_Y2,train_Y3), (valid_X, valid_Y1, valid_Y2,valid_Y3), \
           (test_X, test_Y1, test_Y2, test_Y3)


def truncate_long_sequence(dataset,window_size,shift_step):

    X= dataset[0]
    Y1 = dataset[1]
    Y2 = dataset[2]
    Y3 = dataset[3]
    X_new = []
    Y1_new = []
    Y2_new = []
    Y3_new = []
    for i in range(len(X)):
        seq_len = X[i].shape[0]
        j = 0
        start = 0
        end = 0
        while (j*shift_step+window_size) <= seq_len:
            start = j*shift_step
            end = start+window_size
            X_new.append(X[i][start:end,:])
            Y1_new.append(Y1[i])
            Y2_new.append(Y2[i])
            Y3_new.append(Y3[i])
            j = j + 1

    tmp_X = []
    tmp_Y1 = []
    tmp_Y2 = []
    tmp_Y3 = []
    np.random.seed(2314)
    indices = np.random.permutation(len(X_new))
    for c in indices:
        tmp_X.append(X_new[c])
        tmp_Y1.append(Y1_new[c])
        tmp_Y2.append(Y2_new[c])
        tmp_Y3.append(Y3_new[c])
    del X_new,Y1_new,Y2_new,Y3_new
    Y1_new = np.asarray(tmp_Y1, dtype=np.float32)
    del tmp_Y1
    Y2_new = np.asarray(tmp_Y2, dtype=np.float32)
    del tmp_Y2
    Y3_new = np.asarray(tmp_Y3, dtype=np.float32)
    del tmp_Y3
    X_new = np.asarray(tmp_X, dtype=np.float32)
    del tmp_X
    assert X_new.shape[0] == Y1_new.shape[0] and Y1_new.shape[0] == Y2_new.shape[0] \
           and Y2_new.shape[0] == Y3_new.shape[0]
    new_dataset = (X_new,Y1_new,Y2_new,Y3_new)
    print 'finish trucating.'
    return new_dataset


#define a function for compute horizontal velocity
def compute_velocity_xz_plane(X):
    velocity_xz_plane=[]
    new_X = []
    for xx in X:
        cur_velocity = np.zeros(shape=(xx.shape[0],2),dtype=np.float32)
        cur_velocity[:-1,:] = xx[1:,0:3:2] - xx[:-1,0:3:2]
        cur_velocity[-1,:] = cur_velocity[-2,:]
        velocity_xz_plane.append(cur_velocity)
        new_X.append(np.delete(xx,[0,2],axis=1))
    assert new_X[0].shape[1] ==70
    assert velocity_xz_plane[3].shape[1] ==2
    velocity_xz_plane = np.asarray(velocity_xz_plane,dtype=np.float32).reshape((len(velocity_xz_plane),velocity_xz_plane[0].shape[0],velocity_xz_plane[0].shape[1]))
    new_X = np.asarray(new_X,dtype=np.float32).reshape((len(new_X),new_X[0].shape[0],new_X[0].shape[1]))
    return new_X, velocity_xz_plane



#extract 1st and 3rd dimensions from each frame for computing speed.
def compute_speed_xz(X):
    '''
    :param X:
    :return: new_X:xz global position coordinates are removed
             speed_xz, speed in the xz plane
    '''
    global_position_xz = []
    new_X = []
    for xx in X:
        cur_position_seq = xx[:,0:3:2].copy()
        global_position_xz.append(cur_position_seq)
        new_X.append(np.delete(xx, [0, 2], axis=1))#delete 0,2 axises for each frame
    assert new_X[0].shape[1]==70

    #compute speed according to global_position_xz
    speed_xz = []
    for xz in global_position_xz:
        len_seq = xz.shape[0]
        cur_speed_seq= np.zeros((len_seq,1),dtype=np.float32)

        for j in range(len_seq):
            if j == 0:
                cur_speed_seq[0] = LA.norm((xz[1] - xz[0]))
            elif j ==len_seq-1:
                cur_speed_seq[j] = LA.norm(xz[j]-xz[j-1])
            else:
                cur_speed_seq[j] = 0.5*LA.norm(xz[j+1]-xz[j-1])

        speed_xz.append(cur_speed_seq)

    #
    # for i, speed in enumerate(speed_xz):
    #     new_X[i] = np.concatenate((new_X[i],speed),axis=1)
    #     assert new_X[i].shape[1] == 71
    new_X = np.asarray(new_X,dtype=np.float32).reshape((len(new_X),new_X[0].shape[0],new_X[0].shape[1]))
    speed_xz = np.asarray(speed_xz,dtype=np.float32).reshape(len(speed_xz),speed_xz[0].shape[0],1)
    return new_X,speed_xz


#define a function for compute the velocity feature and append it to X

def compute_velocity_feature_and_append(X):
    new_X = []
    for seq in X:
        T = seq.shape[0]
        dimension = seq.shape[1]
        new_seq = np.zeros(shape=(T,dimension),dtype=np.float32)
        new_seq[:-1,:] = seq[1:,:]-seq[:-1,:]
        new_seq[-1,:] = new_seq[-2,:]
        cur_x_seq = np.concatenate((seq, new_seq),axis=1)
        assert cur_x_seq.shape[0] == seq.shape[0]
        assert cur_x_seq.shape[1] == 2*seq.shape[1]
        new_X.append(cur_x_seq)

    rval = np.asarray(new_X,dtype=np.float32)
    return rval

def compute_velocity_feature(X,keep_T=False):
    '''
    v_t = y_t-y_(t-1)
    v_0 = 0
    :param X:
    :return:return velocity features
    '''
    new_X = []
    for i, seq in enumerate(X):
        T = seq.shape[0]
        dimension = seq.shape[1]
        if keep_T==True:
            shape0 = T
        else:
            shape0 = T-1
        new_seq = np.zeros(shape=(shape0,dimension),dtype=np.float32)
        if keep_T==True:
            new_seq[:-1,:]=seq[1:,:] - seq[:-1,:]
            new_seq[-1,:] = new_seq[-2,:]
        else:
            new_seq[:,:] = seq[1:,:] - seq[:-1,:]

        new_X.append(new_seq)
    rval = np.asarray(new_X,dtype=np.float32)
    return rval








