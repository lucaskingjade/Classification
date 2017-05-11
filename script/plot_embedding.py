import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
#open weights file

filename = './rnn.h5'

hd5_handle = h5py.File(filename,mode='r')
embedding = hd5_handle['embedding_1']['embedding_1_W'][:]
print " shape of embedding is {}".format(embedding.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axe3d = Axes3D(fig)
axe3d.scatter(xs=embedding[:,0],ys=embedding[:,1],zs=embedding[:,2],color='r')

emotion_list = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']
for i in range(len(embedding)):
    axe3d.text(embedding[i,0],embedding[i,1],embedding[i,2],  '%s' % (emotion_list[i]), size=10, zorder=1,
 color='g')
plt.show()
