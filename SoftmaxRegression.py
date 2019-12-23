import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist=input_data.read_data_sets('mnist_data/',one_hot=True)

print('开始构建计算图:')
with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X=tf.placeholder(tf.float32,shape=[None,784],name='X')
        Y_true=tf.placeholder(tf.float32,shape=[None,10],name='Y_true')
    with tf.name_scope('Inference'):
        w=tf.Variable(tf.zeros([784,10]),name='weight')
        b=tf.Variable(tf.zeros([10]),name='Bias')
        logits=tf.add(tf.matmul(X,w),b)
    with tf.name_scope('Softmax'):
        Y_pred=tf.nn.softmax(logits=logits)
    with tf.name_scope('Loss'):
        TrainLoss=tf.reduce_mean(-tf.reduce_sum(Y_true*tf.log(tf.nn.softmax(Y_pred)),axis=1))
    with tf.name_scope('Train'):
        Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
        TrainOp=Optimizer.minimize(TrainLoss)
    with tf.name_scope('Evaluate'):
        correct_prediction=tf.equal(tf.argmax(Y_pred,1),tf.argmax(Y_true,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    Init=tf.global_variables_initializer()

    writer=tf.summary.FileWriter(logdir='LOG_Softmax_Regression',graph=tf.get_default_graph())
    writer.flush()

    print('计算图构建完毕！在TensorBoard中查看！')

    print('开始运行计算图:')

    with tf.Session() as sess:
        sess.run(Init)
        for step in range(3000):
            batch_xs,batch_ys=mnist.train.next_batch(100)
            _,train_loss,train_w,train_b=sess.run([TrainOp,TrainLoss,w,b],feed_dict={X:batch_xs,Y_true:batch_ys})
            print('train step:',step,'Train_Loss:',train_loss)

        accuracy_score=sess.run(accuracy,feed_dict={X:mnist.test.images,Y_true:mnist.test.labels})
        print('模型预测正确率:',accuracy_score)


