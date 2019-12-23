import  tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function

        self.training_scale=scale
        self.weights=dict()

        with tf.name_scope('RawInput'):
            self.x=tf.placeholder(tf.float32,[None,self.n_input])

        with tf.name_scope('NoiseAdder'):
            self.scale=tf.placeholder(tf.float32)
            self.noise_x=self.x+self.scale*tf.random_normal((n_input,))
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='Weight1')
            self.weights['b1'] = tf.Variable(tf.zeros(self.n_hidden, dtype=tf.float32), name='Bias1')
            self.hidden=self.transfer(tf.add(tf.matmul(self.noise_x,self.weights['w1']),self.weights['b1']))
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32), name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros(self.n_input, dtype=tf.float32), name='Bias2')
            self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        with tf.name_scope('Loss'):
            self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2))
        with tf.name_scope('Train'):
            self.optimizer=optimizer.minimize(self.cost)

        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        print('begin to run session...')
    #一个批次上训练模型
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    #返回自编码器隐含层的输出结果，获得抽象后的高阶特征表示
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    #将隐藏层的高阶特征作为输入，将其重建为原始的输入数据
    def generate(self,hidden=None):
        if hidden==None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    #整体运行一遍复原过程，包括提取高阶特征以及重建原始数据，输入原始数据，输出复原后的数据
    def reconstruction(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    #获取隐含层的权重与偏置
    def getWeight(self):
        return self.sess.run(self.weights['w1'])
    def getBias(self):
        return self.sess.run(self.weights['b1'])

    # def _initialize_weight(self):
    #     all_weights=dict()
    #     all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden),name='Weight1')
    #     all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32),name='Bias1')
    #     all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32),name='weight2')
    #     all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32),name='Bias2')
    #     return all_weights
print('产生AdditiveGaussianNoiseAutoencoder类对象实例')
AGN_AC=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,
                                        transfer_function=tf.nn.softplus,
                                        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                        scale=0.01)
print('把计算图写入事件文件，在tensorBoard中察看！！')
writer=tf.summary.FileWriter(logdir='LOG_DenoiseAutoEncoder',graph=tf.get_default_graph())
writer.flush()

mnist=input_data.read_data_sets('mnist_data',one_hot=True)

def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)

n_samples=int(mnist.train.num_examples)
training_epoch=20
batch_size=128
display_step=1

for epoch in range(training_epoch):
    avg_cost=0
    total_batch=int(n_samples/batch_size)  #总的批次

    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=AGN_AC.partial_fit(batch_xs)
        avg_cost+=cost/batch_size
    avg_cost/=total_batch      #每一个epoch上面的损失

    if epoch%display_step==0:
        print('epoch:%04d,cost=%.9f'%(epoch+1,avg_cost))

print('Total cost:',str(AGN_AC.calc_total_cost(X_test)))