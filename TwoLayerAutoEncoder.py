import tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing as prep
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_LOG_MIN_LEVEL']='2'

#Hyperparmameters:
learning_rate=0.01
training_epoch=20
batch_size=256
display_step=1
example_to_show=10

n_hidden1_units=256  #编码器第一隐藏层神经元数量，会影响压缩效果
n_hidden2_units=128  #编码器第一隐藏层神经元数量，会影响压缩效果
n_input_units=784
n_output_units=n_input_units  #解码器输出层神经元数量等于输入数据的units数量

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

def Weightvariable(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out]),dtype=tf.float32,name=name_str)

def BiasesVariable(n_out,name_str):
    return tf.Variable(tf.random_normal([n_out]),dtype=tf.float32,name=name_str)

def Encoder(x_origin,activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        weights=Weightvariable(n_in=n_input_units,n_out=n_hidden1_units,name_str='weight')
        biases=BiasesVariable(n_out=n_hidden1_units,name_str='bias')
        x_code1=activate_func(tf.add(tf.matmul(x_origin,weights),biases))
        variable_summaries(weights)
        variable_summaries(biases)
    with tf.name_scope('Layer2'):
        weights=Weightvariable(n_in=n_hidden1_units,n_out=n_hidden2_units,name_str='weight')
        biases=BiasesVariable(n_out=n_hidden2_units,name_str='bias')
        x_code2=activate_func(tf.add(tf.matmul(x_code1,weights),biases))
        variable_summaries(weights)
        variable_summaries(biases)
    return x_code2

def Decoder(x_code,activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        weights=Weightvariable(n_in=n_hidden2_units,n_out=n_hidden1_units,name_str='weight')
        biases=BiasesVariable(n_out=n_hidden1_units,name_str='bias')
        x_decode1=activate_func(tf.add(tf.matmul(x_code,weights),biases))
        variable_summaries(weights)
        variable_summaries(biases)
    with tf.name_scope('Layer2'):
        weights=Weightvariable(n_in=n_hidden1_units,n_out=n_output_units,name_str='weight')
        biases=BiasesVariable(n_out=n_output_units,name_str='bias')
        x_decode2=activate_func(tf.add(tf.matmul(x_decode1,weights),biases))
        variable_summaries(weights)
        variable_summaries(biases)
    return x_decode2

with tf.Graph().as_default():
    with tf.name_scope('X_origin'):
        X_origin=tf.placeholder(tf.float32,shape=[None,n_input_units])
    with tf.name_scope('Encoder'):
        X_code=Encoder(X_origin,activate_func=tf.nn.sigmoid)
    with tf.name_scope('Decoder'):
        X_decode=Decoder(X_code,activate_func=tf.nn.sigmoid)
    with tf.name_scope('Loss'):
        Loss=tf.reduce_mean(tf.pow(tf.subtract(X_origin,X_decode),2))

    with tf.name_scope('Train'):
        Optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        Train=Optimizer.minimize(Loss)

    with tf.name_scope('LossSummary'):
        tf.summary.scalar('loss',Loss)      #标量用scalar汇总
        tf.summary.scalar('learning_rate',learning_rate)

    with tf.name_scope('Imagesummary'):
        image_origin=tf.reshape(X_origin,[-1,28,28,1])  #1代表单通道
        image_reconstructed=tf.reshape(X_decode,[-1,28,28,1])
        tf.summary.image('image_original',image_origin,10)
        tf.summary.image('image_reconstructed',image_reconstructed,10)


    merged_summary=tf.summary.merge_all()

    init=tf.global_variables_initializer()

    print('计算图构建完毕！可以在TensorBoard中查看！')
    summary_writer=tf.summary.FileWriter(logdir='LOG_TwoLayerAutoEncoder',graph=tf.get_default_graph())
    summary_writer.flush()


    mnist=input_data.read_data_sets('mnist_data',one_hot=True)

    with tf.Session() as sess:
        sess.run(init)
        total_batch=int(mnist.train.num_examples/batch_size)

        for epoch in range(training_epoch):
            for i in range(total_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_size)

                _,loss=sess.run([Train,Loss],feed_dict={X_origin:batch_xs,})

            if epoch%display_step==0:
                print('Epoch:','%04d'%(epoch+1),'loss=','{:.9f}'.format(loss))

                summary_str=sess.run(merged_summary,feed_dict={X_origin:batch_xs})
                summary_writer.add_summary(summary_str,epoch)
                summary_writer.flush()

        summary_writer.close()
        print('模型训练完毕！')


        # reconstruction=sess.run(X_decode,feed_dict={X_origin:mnist.test.images[:example_to_show]})
        # f,a=plt.subplots(2,10,figsize=(10,2))
        # for i in range(example_to_show):
        #     a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        #     a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
        # f.show()
        # plt.draw()
        # plt.waitforbuttonpress()

