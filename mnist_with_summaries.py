import argparse
import tensorflow as tf
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NUM_CLASS=10
IMAGE_SIZE=28
IMAGE_PIXELS=IMAGE_SIZE*IMAGE_SIZE

FLAG=None

def weight_variable(shape):
    initial=tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(input_tensor=var)
        tf.summary.scalar('mean',mean)  #返回张量的均值
        with tf.name_scope('stddev'):    #返回张量的标准差
            stddev=tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(input_tensor=var))
        tf.summary.scalar('min',tf.reduce_min(input_tensor=var))
        tf.summary.histogram('histogram',var)

def nn_layer(input_tensor,input_dim,out_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights=weight_variable([input_dim,out_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases=bias_variable([out_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate=tf.add(tf.matmul(input_tensor,weights),biases)
            tf.summary.histogram('preactivate',preactivate)
    with tf.name_scope('activation'):
        activations=act(preactivate)
        tf.summary.histogram('activations',activations)
        return activations
def train():
    with tf.name_scope('input'):
        x=tf.placeholder(tf.float32,[None,IMAGE_PIXELS],name='x-input')
        y_true=tf.placeholder(tf.int64,[None,NUM_CLASS],name='y-input')

    with tf.name_scope('input_shape'):
        image_shape_input=tf.reshape(x,[-1,28,28,1])    #高/宽/通道数
        tf.summary.image('input',image_shape_input,10)

    hidden1=nn_layer(x,IMAGE_PIXELS,FLAG.hidden1,'layer1')

    with tf.name_scope('dropout'):
        keep_prob=tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability',keep_prob)
        dropped=tf.nn.dropout(hidden1,keep_prob)

    logits=nn_layer(dropped,FLAG.hidden1,NUM_CLASS,'layer2',act=tf.identity)

    with tf.name_scope('loss'):
        diff=tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=logits)
        with tf.name_scope('total'):
            cross_entropy=tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy',cross_entropy)

    with tf.name_scope('train'):
        train_step=tf.train.AdamOptimizer(FLAG.learning_rate).minimize(loss=cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction=tf.equal(tf.argmax(input=logits,axis=1),tf.argmax(input=y_true,axis=1))
        with tf.name_scope('accuracy'):
            accuracy=tf.reduce_mean(input_tensor=tf.cast(correct_prediction,dtype=tf.float32))

    mnist=input_data.read_data_sets(FLAG.data_dir,one_hot=True,fake_data=FLAG.fake_data)
    #tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or FLAG.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAG.fake_data)
            k = FLAG.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_true: ys, keep_prob: k}


    with tf.Session() as sess:
        merged=tf.summary.merge_all()
        train_writer=tf.summary.FileWriter(FLAG.log_dir+'/train',sess.graph)
        test_writer=tf.summary.FileWriter(FLAG.log_dir+'/test')

        tf.global_variables_initializer().run()

        for step in range(FLAG.max_step):
            _,summary_str,XentropyLoss=sess.run([train_step,merged,cross_entropy],feed_dict(True))
            train_writer.add_summary(summary_str,global_step=step)
            print('step idx:',step,'XentropyLoss:',XentropyLoss)

            if (step%100==0):
                test_summary_str,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
                test_writer.add_summary(test_summary_str,global_step=step)
                print('Accuracy at step %s: %s'%(step,acc))

def main(_):
    if tf.gfile.Exists(FLAG.log_dir):
        tf.gfile.DeleteRecursively(FLAG.log_dir)
    tf.gfile.MakeDirs(FLAG.log_dir)

    train()


if __name__=='__main__':
    parser=argparse.ArgumentParser()         #解析器
    parser.add_argument(
        '--fake_data',
        type=bool,
        default=False,
        help='If true,uses fake data for unit testing.'
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=1000,
        help='Number of steps to run trainer'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='initial learning_rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.9,
        help='Keep probability for training dropout'
    )
    parser.add_argument(
        '--hidden1',
        type=float,
        default=100,
        help='The number of first layer units'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='mnist_data',
        help='Directory for storing input data'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='LOG_MNIST_WITH_SUMMARIES',
        help='summaries log directory'
    )
    FLAG,unparsed=parser.parse_known_args()     #成功被解析的放在FLAG中，不能被解析的放在unparsed中
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)  #


