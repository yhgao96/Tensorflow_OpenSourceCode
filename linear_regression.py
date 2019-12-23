import tensorflow as tf
import os
import numpy as np
from matplotlib import pylab as  plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#产生训练数据集
train_X=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                   7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                    2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_train_sample=train_X.shape[0]
print('训练样本数量为:',n_train_sample)

test_X=np.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
test_Y=np.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
n_test_sample=test_X.shape[0]
print('测试样本数量为:',n_test_sample)

plt.scatter(train_X,train_Y)
plt.scatter(test_X,test_Y)
plt.show()

with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X=tf.placeholder(tf.float32,name='X')
        Y_true=tf.placeholder(tf.float32,name='Y_true')
    with tf.name_scope('Inference'):
        w=tf.Variable(tf.zeros([1]),name='weight')
        b=tf.Variable(tf.zeros([1]),name='Bias')
        Y_pred=tf.add(tf.multiply(X,w),b)
    with tf.name_scope('Loss'):
        TrainLoss=tf.reduce_mean(tf.pow(Y_true-Y_pred,2))/2
    with tf.name_scope('Train'):
        Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
        TrainOp=Optimizer.minimize(TrainLoss)
    with tf.name_scope('Evaluate'):
        EvalLoss=tf.reduce_mean(tf.pow(Y_true-Y_pred,2))/2

    initOp=tf.global_variables_initializer()

    writer=tf.summary.FileWriter(logdir='Log_linear_regression',graph=tf.get_default_graph())
    writer.flush()

    print('开启会话，运行计算图，训练模型！')

    with tf.Session() as sess:
        sess.run(initOp)
        for step in range(1000):         #训练1000次
            for tx,ty in zip(train_X,train_Y):
                _,train_loss,train_w,train_b=sess.run([TrainOp,TrainLoss,w,b],feed_dict={X:tx,Y_true:ty})

            if(step+1)%5==0:
                print('step:','%04d'%(step+1),'train_loss=','{:.9f}'.format(train_loss),'w=',train_w,'b=',train_b)
            if (step+1)%10==0:
                test_loss=sess.run(EvalLoss,feed_dict={X:test_X,Y_true:test_Y})
                print('step:','%04d'%(step+1),'test_loss=','{:.9f}'.format(test_loss),
                      'w=',train_w,'b=',train_b)

        print('训练完毕！！')
        w,b=sess.run([w,b])
        print('模型参数:','w=',w,'b=',b)
        train_loss=sess.run(TrainLoss,feed_dict={X:train_X,Y_true:train_Y})
        print('训练集上的损失:',train_loss)
        test_loss=sess.run(EvalLoss,feed_dict={X:test_X,Y_true:test_Y})
        print('测试集上面的损失:',test_loss)

        plt.plot(train_X, train_Y, 'ro', label='Origin Train Points')
        plt.plot(test_X, test_Y, 'b*', label='Origin Test Points')
        plt.plot(train_X, w * train_X + b, label='Fitted Line')
        plt.legend()
        plt.show()
