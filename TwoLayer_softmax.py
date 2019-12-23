import tensorflow as tf
import os
import math
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#mnist=input_data.read_data_sets('mnist_data/',one_hot=True)
IMAGE_SIZE=28
IMAGE_PIXELS=IMAGE_SIZE*IMAGE_SIZE
#batch_size=50
#hidden1_units=20
#hidden2_units=15
#learning_rate=0.01

NUM_Class=10

def placeholder_inputs(batch_size):
    image_palceholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return image_palceholder,labels_placeholder

def inference(image,hidden1_units,hidden2_units):
    with tf.name_scope('hidden1'):
        w=tf.Variable(tf.random.truncated_normal([IMAGE_PIXELS,hidden1_units],   #mean-2*stddev~mean+2*stddev
                                                 stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),name='Weight')
    # 以weights矩阵的行数规范化标准差，就是让weights矩阵的每一列都服从0均值截断正态分布，这样不会给输入信号加入人为的偏置

        b=tf.Variable(tf.zeros([hidden1_units]),name='bias')
        hidden1=tf.nn.relu(tf.add(tf.matmul(image,w),b))
    with tf.name_scope('hidden2'):
        w=tf.Variable(tf.random.truncated_normal([hidden1_units,hidden2_units],
                                                 stddev=1.0/math.sqrt(float(hidden1_units))),name='weight')
        b=tf.Variable(tf.zeros([hidden2_units]),name='bias')
        hidden2=tf.nn.relu(tf.add(tf.matmul(hidden1,w),b))

    with tf.name_scope('softmax_linear'):
        w=tf.Variable(tf.random.truncated_normal([hidden2_units,NUM_Class],
                                                 stddev=1.0/math.sqrt(float(hidden2_units))),name='weight')
        b=tf.Variable(tf.zeros([NUM_Class]),name='bias')
        logits=tf.add(tf.matmul(hidden2,w),b)
        return logits

def loss(logits,labels):
    labels=tf.to_int64(labels)
    cross_entrypy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='xentrypy')
    #把logist转化为softmax上面的概率分布
    return tf.reduce_mean(cross_entrypy,name='xentrypy_mean')

def training(loss,learning_rate):
    with tf.name_scope('scalar_summaries'):
        tf.summary.scalar('learning_rate',learning_rate)
        tf.summary.scalar('loss',loss)
    Optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    global_step=tf.Variable(0,name='global_step',trainable=False)
    trainOp=Optimizer.minimize(loss,global_step)
    return trainOp

def evaluate(logits,labels):
    correct=tf.nn.in_top_k(predictions=logits,targets=labels,k=1)
    return tf.reduce_sum(input_tensor=tf.cast(correct,tf.int32))

# image_palceholder,labels_placeholder=placeholder_inputs(batch_size)
# logits=inference(image_palceholder,hidden1_units,hidden2_units)
# batch_loss=loss(logits,labels=labels_placeholder)
# train_batch=training(batch_loss,learning_rate=learning_rate)
# correct_counts=evaluate(logits=logits,labels=labels_placeholder)

# Init=tf.global_variables_initializer()
#
# print('计算图已经写入！在Tensorboard中查看！')
# writer=tf.summary.FileWriter(logdir='LOG_TwoLayer_Softmax',graph=tf.get_default_graph())
# writer.flush()

