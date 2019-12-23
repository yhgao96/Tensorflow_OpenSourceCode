import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('mnist_data/',one_hot=True)

Xtrain,Ytrain=mnist.train.next_batch(5000)
Xtest,Ytest=mnist.test.next_batch(200)
print('Xtrain.shape:',Xtrain.shape,'Xtest.shape:',Xtest.shape)
print('Ytrain.shape:',Ytrain.shape,'Ytest.shape:',Ytest.shape)

xtrain=tf.placeholder('float',[None,784],name='X_train')   #不知道训练样本的个数用None来表示，特证数是28*28=784
xtest=tf.placeholder('float',[784],name='X_test')

distance=tf.reduce_sum(tf.abs(tf.add(xtrain,tf.negative(xtest))),axis=1,name='L1')  #逐行进行计算，缩减为一个行向量

pred=tf.arg_min(distance,0)   #获取最小距离的索引

init=tf.global_variables_initializer()

accuracy=0.

with tf.Session() as sess:
    sess.run(init)
    Ntest=len(Xtest)  #测试样本数量

    for i in range(Ntest):
        nn_index=sess.run(pred,feed_dict={xtrain:Xtrain,xtest:Xtest[i,:]})   #每次只传入一个测试样本

        pred_class_label=np.argmax(Ytrain[nn_index])
        true_class_label=np.argmax(Ytest[i])
        print('Test',i,'Prediction Class Label:',pred_class_label,'True Class Label:',true_class_label)

        if pred_class_label==true_class_label:
            accuracy+=1

    print('Done!')
    accuracy=accuracy/Ntest
    print('Accuracy:',accuracy)


writer=tf.summary.FileWriter(logdir='logs_KNN',graph=tf.get_default_graph())
writer.flush()


