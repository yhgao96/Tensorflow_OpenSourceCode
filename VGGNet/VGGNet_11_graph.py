import tensorflow as tf
import os
import numpy as  np
import csv
os.environ['TF_CPP_LOG_MIN_LEVEL']='2'

learning_rate_init=0.01
keep_prob=0.7
training_epoch=1
batch_size=32
display_step=2

num_example_per_epoch_for_train=1000
num_example_per_epoch_for_eval=500
image_size=224
image_channel=3
n_classes=1000

#用假数据进行训练
def get_fake_train_batch(batch_size):
    images=tf.Variable(tf.random_normal(shape=[batch_size,image_size,image_size,image_channel],
                                        mean=0.0,stddev=1.0,dtype=tf.float32))
    labels=tf.Variable(tf.random_uniform(shape=[batch_size],minval=0,maxval=n_classes,dtype=tf.int32))
    return images,labels

def get_fake_test_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel],
                                          mean=0.0, stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0, maxval=n_classes, dtype=tf.int32))
    return images, labels

#2维卷积层activation(conv2d+bias)的封装
def Conv2d_Op(input_op,name,kh,kw,n_out,dh,dw,activation_func=tf.nn.relu,activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in=input_op.get_shape()[-1].value  #image_channel
        kernels=tf.get_variable(scope+'weight',shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernels,strides=(1,dh,dw,1),padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,dtype=tf.float32)
        z=tf.nn.bias_add(conv,biases)
        activation=activation_func(z,name=activation_name)
        return activation

#2维池化层pool进行封装
def Pool2d_Op(input_op,name,kh=2,kw=2,dh=2,dw=2,padding='SAME',pool_func=tf.nn.max_pool):
    with tf.name_scope(name) as scope:
        return pool_func(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding=padding,name=name)

#全连接层activate(wx+b)
def FullyConnected_Op(input_op,name,n_out,activation_func=tf.nn.relu,activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in=input_op.get_shape()[-1].value
        kernels=tf.get_variable(scope+'weight',shape=[n_in,n_out],dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        bias_init_val=tf.constant(0.1,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,dtype=tf.float32)
        z=tf.add(tf.matmul(input_op,kernels),biases)
        activation=activation_func(z,name=activation_name)
        return activation

def AddActivationSummary(x):
    tf.summary.histogram('/activations',x)
    tf.summary.scalar('/sparsity',tf.nn.zero_fraction(x))

def AddLossSummary(losses):
    loss_average=tf.train.ExponentialMovingAverage(0.9,name='avg')
    loss_average_op=loss_average.apply(losses)
    for loss in losses:
        tf.summary.scalar(loss.op.name+'(raw)',loss)
        tf.summary.scalar(loss.op.name+'(avg)',loss_average.average(loss))
    return loss_average_op

#打印出每一层的输出张量的shape
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#Inference
def Inference(image_holder,keep_prob=keep_prob):
    #第一段卷积网络
    conv1_1=Conv2d_Op(image_holder,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1)
    pool1=Pool2d_Op(conv1_1,name='pool1',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    AddActivationSummary(conv1_1)
    print_activations(pool1)

    # 第二段卷积网络
    conv2_1 = Conv2d_Op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = Pool2d_Op(conv2_1, name='pool2', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    AddActivationSummary(conv2_1)
    print_activations(pool2)

    # 第三段卷积网络
    conv3_1 = Conv2d_Op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = Conv2d_Op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = Pool2d_Op(conv3_2, name='pool3', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    AddActivationSummary(conv3_2)
    print_activations(pool3)

    # 第四段卷积网络
    conv4_1 = Conv2d_Op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = Conv2d_Op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4= Pool2d_Op(conv4_2, name='pool4', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    AddActivationSummary(conv4_2)
    print_activations(pool4)

    # 第五段卷积网络
    conv5_1 = Conv2d_Op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = Conv2d_Op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = Pool2d_Op(conv5_2, name='pool5', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    AddActivationSummary(conv5_2)
    print_activations(pool5)

    #将二维特征图转化为一维特征向量
    with tf.name_scope('FeastReshape'):
        feature=tf.reshape(pool5,shape=[batch_size,-1])
        feats_dim=feature.get_shape()[1].value

    #第一个全连接层
    fc1_out=FullyConnected_Op(feature,name='fc1',n_out=4096,activation_func=tf.nn.relu,activation_name='relu')
    AddActivationSummary(fc1_out)
    print_activations(fc1_out)

    #Dropout层
    with tf.name_scope('dropout_1'):
        fc1_dropout=tf.nn.dropout(fc1_out,keep_prob=keep_prob)

    ##第二个全连接层
    fc2_out=FullyConnected_Op(fc1_dropout,name='fc2',n_out=4096,activation_func=tf.nn.relu,activation_name='relu')
    AddActivationSummary(fc2_out)
    print_activations(fc2_out)

    #Dropout层
    with tf.name_scope('dropout_2'):
        fc2_dropout = tf.nn.dropout(fc2_out, keep_prob=keep_prob)

    #第三个全连接层
    logits=FullyConnected_Op(fc2_dropout,name='fc3',n_out=n_classes,activation_func=tf.identity,activation_name='identity')
    AddActivationSummary(logits)
    print_activations(logits)
    return logits


def TrainAndTestModel():
    with tf.Graph().as_default():
        with tf.name_scope('Inputs'):
            images_holder=tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,image_channel],
                                         name='images')
            labels_holder=tf.placeholder(tf.int32,shape=[batch_size],name='labels')

        with tf.name_scope('Inference'):
            keep_prob_holder=tf.placeholder(tf.float32,name='KeepProb')
            logits=Inference(images_holder,keep_prob_holder)

        with tf.name_scope('Loss'):
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder,logits=logits)
            cross_entropy_mean=tf.reduce_mean(cross_entropy)
            total_loss_op=cross_entropy_mean
            average_loss_op=AddLossSummary([total_loss_op])

        with tf.name_scope('Train'):
            learning_rate=tf.placeholder(tf.float32,name='LearningRate')
            global_step=tf.Variable(0,name='global_step',trainable=False,dtype=tf.int64)
            optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate_init)
            train_op=optimizer.minimize(total_loss_op,global_step)

        with tf.name_scope('Evaluate'):
            top_K_op=tf.nn.in_top_k(predictions=logits,targets=labels_holder,k=1)

        with tf.name_scope('GetTrainBatch'):
            images_train,labels_train=get_fake_train_batch(batch_size=batch_size)
            tf.summary.image('images',images_train,max_outputs=8)

        with tf.name_scope('GetTestBatch'):
            images_test,labels_test=get_fake_test_batch(batch_size=batch_size)
            tf.summary.image('images',images_test,max_outputs=8)

        merged_summaries=tf.summary.merge_all()

        init=tf.global_variables_initializer()

        print('计算图写入事件文件，可以在TensorBoard中查看！！')
        summary_writer=tf.summary.FileWriter(logdir='LOG_VGGNet_11',graph=tf.get_default_graph())
        summary_writer.flush()

        result_list=list()

        result_list.append(['learning_rate',learning_rate_init,
                            'training_epoch',training_epoch,
                            'batch_size',batch_size,
                            'display_step',display_step])
        result_list.append(['training_step','train_loss','training_step','train_accuracy'])

        with tf.Session() as sess:
            sess.run(init)

            print('===>>>>>>开始在训练集上训练数据<<<<<<===')
            num_batches_per_epoch = int(num_example_per_epoch_for_train / batch_size)
            print('Per batch Size:', batch_size)
            print("Train sample Count Per Epoch:", num_example_per_epoch_for_train)
            print('Total batch Per Epoch:', num_batches_per_epoch)

            training_step = 0
            for epoch in range(training_epoch):
                for batch_idx in range(num_batches_per_epoch):
                    images_batch, labels_batch = sess.run([images_train, labels_train])
                    _, loss_value, avg_loss = sess.run([train_op, total_loss_op, average_loss_op],
                                                        feed_dict={images_holder: images_batch,
                                                                    labels_holder: labels_batch,
                                                                   keep_prob_holder:keep_prob,
                                                                   learning_rate: learning_rate_init})
                    training_step = sess.run(global_step)

                    if training_step % display_step == 0:
                        predictions = sess.run([top_K_op],
                                                feed_dict={images_holder: images_batch,
                                                            labels_holder: labels_batch,
                                                           keep_prob_holder:keep_prob})
                        batch_accuracy = np.sum(predictions) / batch_size

                        print('Training Epoch:' + str(epoch) + ',Training Step:' + str(training_step) +
                                ',Training Loss=' + '{:.6f}'.format(loss_value) +
                                ',Training Accuracy=' + '{:.5f}'.format(batch_accuracy))

                        result_list.append([training_step, loss_value, training_step, batch_accuracy])

                        summaries_str = sess.run(merged_summaries,
                                                feed_dict={images_holder: images_batch,
                                                            labels_holder: labels_batch,
                                                           keep_prob_holder:keep_prob})

                        summary_writer.add_summary(summary=summaries_str, global_step=training_step)
                        summary_writer.flush()
            print('训练完毕！')

            print('===>>>>>>开始在测试集上测试模型<<<<<<==')
            num_batch_per_epoch = int(num_example_per_epoch_for_eval / batch_size)
            print('Per batch Size:', batch_size)
            print('Test sample Count Per Epoch:', num_example_per_epoch_for_eval)
            print('Total batch Count Per Epoch:', num_batch_per_epoch)

            correct_prediction = 0.0

            for idx in range(num_batches_per_epoch):
                images_batch, labels_batch = sess.run([images_test, labels_test])
                predictions = sess.run([top_K_op], feed_dict={images_holder: images_batch,
                                                            labels_holder: labels_batch})
                correct_prediction += np.sum(predictions)

            accuracy_score = correct_prediction / num_example_per_epoch_for_eval
            print('-------->Test Accuracy: ', accuracy_score)
            result_list.append(['Test Accuracy: ', accuracy_score])

        result_file=open('evaluate.csv','w',newline='')
        csv_writer=csv.writer(result_file,dialect='excel')
        for row in result_list:
            csv_writer.writerow(row)


def main(argv=None):
    train_dir='logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    else:
        tf.gfile.MakeDirs(train_dir)
    TrainAndTestModel()

if __name__=='__main__':
    tf.app.run()