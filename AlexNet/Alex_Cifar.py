import tensorflow as tf
import os
import numpy as np
import csv
import cifar_input_for_alexnet

os.environ['TF_CPP_LOG_MIN_LEVEL']='2'

learning_rate_init=0.01
training_epoch=1
batch_size=100
display_step=5
conv1_kernel_num=64
conv2_kernel_num=192
conv3_kernel_num=384
conv4_kernel_num=256
conv5_kernel_num=256
fc1_units_num=2048
fc2_units_num=2048

dataset_dir_cifar10='../../CIFAR10_dataset/cifar-10-batches-bin'
dataset_dir_cifar100='../CIFAR100_dataset/cifar-100-binary'
num_example_per_epoch_for_train=cifar_input_for_alexnet.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_example_per_epoch_for_eval=cifar_input_for_alexnet.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size=cifar_input_for_alexnet.IMAGE_SIZE
image_channel=cifar_input_for_alexnet.IMAGE_DEPTH

cifar10or20or100=10
if cifar10or20or100==10:
    n_classes=cifar_input_for_alexnet.NUM_CLASSES_CIFAR10
    dataset_dir=dataset_dir_cifar10
if cifar10or20or100==20:
    n_classes=cifar_input_for_alexnet.NUM_CLASSES_CIFAR20
    dataset_dir=dataset_dir_cifar100
if cifar10or20or100==100:
    n_classes=cifar_input_for_alexnet.NUM_CLASSES_CIFAR100
    dataset_dir=dataset_dir_cifar100

def get_distorted_train_batch(data_dir,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images,labels=cifar_input_for_alexnet.distorted_inputs(cifar10or20or100=n_classes,data_dir=data_dir,
                                                           batch_size=batch_size)
    return images,labels

def get_undistorited_eval_batch(data_dir,eval_data=True,batch_size=batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images,labels=cifar_input_for_alexnet.inputs(cifar10or20or100=n_classes,
                                                 eval_data=eval_data,
                                                 data_dir=data_dir,
                                                 batch_size=batch_size)
    return images,labels

def Weightvariable(shape,name_str,stddev=0.1):
    initial=tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def BiasVariable(shape,name_str,init_value=0.0):
    initial=tf.constant(init_value,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def Conv2d(x,w,b,stride=1,padding='SAME',activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y=tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    with tf.name_scope(act_name):
        y=activation(y)
    return y

def Pool2d(x,pool=tf.nn.max_pool,k=3,stride=2,padding='SAME'):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding=padding)

def FullyConnected(x,w,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('wx_plus_b'):
        y=tf.matmul(x,w)
        y=tf.add(y,b)
    with tf.name_scope(act_name):
        y=activate(y)
    return y
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
    print(t.op.name,' ',t.get_shape().as_list())

def Inference(image_holder):
    with tf.name_scope('Conv2d_1'):
        weights=Weightvariable(shape=[5,5,image_channel,conv1_kernel_num],
                              name_str='weights',stddev=1e-1)
        biases=BiasVariable(shape=[conv1_kernel_num],name_str='biases1',init_value=0.0)
        conv1_out=Conv2d(image_holder,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(conv1_out)
        print_activations(conv1_out)
    with tf.name_scope('Pool2d_1'):
        pool1_out=Pool2d(conv1_out,pool=tf.nn.max_pool,k=3,stride=2,padding='VALID')
        print_activations(pool1_out)
    with tf.name_scope('COnv2d_2'):
        weights=Weightvariable(shape=[5,5,conv1_kernel_num,conv2_kernel_num],
                               name_str='weights',stddev=1e-1)
        biases=BiasVariable(shape=[conv2_kernel_num],name_str='biases',init_value=0.0)
        conv2_out=Conv2d(pool1_out,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(conv2_out)
        print_activations(conv2_out)
    with tf.name_scope('Pool2d_2'):
        pool2_out=Pool2d(conv2_out,pool=tf.nn.max_pool,k=3,stride=2,padding='VALID')
        print_activations(pool2_out)
    with tf.name_scope('Conv2d_3'):
        weights=Weightvariable(shape=[3,3,conv2_kernel_num,conv3_kernel_num],
                               name_str='weights',stddev=1e-1)
        biases=BiasVariable(shape=[conv3_kernel_num],name_str='biases',init_value=0.0)
        conv3_out=Conv2d(pool2_out,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(conv3_out)
        print_activations(conv3_out)
    with tf.name_scope('Conv2d_4'):
        weights=Weightvariable(shape=[3,3,conv3_kernel_num,conv4_kernel_num],
                               name_str='weights',stddev=1e-1)
        biases=BiasVariable(shape=[conv4_kernel_num],name_str='biases',init_value=0.0)
        conv4_out=Conv2d(conv3_out,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(conv4_out)
        print_activations(conv4_out)
    with tf.name_scope('Conv2d_5'):
        weights=Weightvariable(shape=[3,3,conv4_kernel_num,conv5_kernel_num],
                               name_str='weights',stddev=1e-1)
        biases=BiasVariable(shape=[conv5_kernel_num],name_str='biases',init_value=0.0)
        conv5_out=Conv2d(conv4_out,weights,biases,stride=1,padding='SAME')
        AddActivationSummary(conv5_out)
        print_activations(conv5_out)
    with tf.name_scope('Pool2d_5'):
        pool5_out=Pool2d(conv5_out,pool=tf.nn.max_pool,k=3,stride=2,padding='VALID')
        print_activations(pool5_out)

    with tf.name_scope('FeastReshape'):
        feature=tf.reshape(pool5_out,[batch_size,-1])
        feast_dim=feature.get_shape()[1].value
    with tf.name_scope('FC1_nonlinear'):
        weights=Weightvariable(shape=[feast_dim,fc1_units_num],
                               name_str='weights',stddev=4e-2)
        biases=BiasVariable(shape=[fc1_units_num],name_str='biases',init_value=0.1)
        fc1_out=FullyConnected(feature,weights,biases,activate=tf.nn.relu,act_name='relu')
        AddActivationSummary(fc1_out)
        print_activations(fc1_out)
    with tf.name_scope('FC2_nonlinear'):
        weights=Weightvariable(shape=[fc1_units_num,fc2_units_num],name_str='weights',stddev=4e-2)
        biases=BiasVariable(shape=[fc2_units_num],name_str='biases',init_value=0.1)
        fc2_out=FullyConnected(fc1_out,weights,biases,activate=tf.nn.relu,act_name='relu')
        AddActivationSummary(fc2_out)
        print_activations(fc2_out)
    with tf.name_scope('FC3_linear'):
        weights=Weightvariable(shape=[fc2_units_num,n_classes],name_str='weights',stddev=1.0/fc2_units_num)
        biases=BiasVariable(shape=[n_classes],name_str='biases',init_value=0.0)
        logits=FullyConnected(fc2_out,weights,biases,activate=tf.identity,act_name='linear')
        AddActivationSummary(logits)
        print_activations(logits)
    return logits
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

with tf.Graph().as_default():
    with tf.name_scope('Input'):
        images_holder=tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,image_channel],name='images')
        label_holder=tf.placeholder(tf.int32,shape=[batch_size],name='labels')
    with tf.name_scope('Inference'):
        logits=Inference(images_holder)
    with tf.name_scope('Loss'):
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_holder,logits=logits)
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        total_loss_op=cross_entropy_mean
        average_loss_op=AddLossSummary([total_loss_op])
    with tf.name_scope('Train'):
        learning_rate=tf.placeholder(tf.float32)
        global_step=tf.Variable(0,name='global_step',trainable=False,dtype=tf.int64)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_init)
        train_op=optimizer.minimize(total_loss_op,global_step=global_step)
    with tf.name_scope('Evaluate'):
        top_K_op=tf.nn.in_top_k(predictions=logits,targets=label_holder,k=2)
    with tf.name_scope('GetTrainBatch'):
        if cifar10or20or100==-1:
            images_train,labels_train=get_fake_train_batch(batch_size=batch_size)
        else:
            images_train,labels_train=get_distorted_train_batch(data_dir=dataset_dir,
                                                            batch_size=batch_size)
        tf.summary.image('images',images_train,max_outputs=8)
    with tf.name_scope('GetTestBatch'):
        if cifar10or20or100==-1:
            images_test,labels_test=get_fake_test_batch(batch_size=batch_size)
        else:
            images_test,labels_test=get_undistorited_eval_batch(data_dir=dataset_dir,
                                                                eval_data=True,
                                                                batch_size=batch_size)
        tf.summary.image('images',images_test,max_outputs=8)

    merged_summaries=tf.summary.merge_all()

    init=tf.global_variables_initializer()

    print('计算图构建完毕，可以在tensorboard中查看！')

    summary_writer = tf.summary.FileWriter(logdir='LOG_Alex_Cifar', graph=tf.get_default_graph())
    summary_writer.flush()

    result_list = list()
    result_list.append(['learning_rate', learning_rate_init,
                        'training_epoch', training_epoch,
                        'batch_size', batch_size,
                        'display_step', display_step,
                        'conv1_kernel_num', conv1_kernel_num,
                        'conv2_kernel_num', conv2_kernel_num,
                        'conv3_kernel_num', conv3_kernel_num,
                        'conv4_kernel_num', conv4_kernel_num,
                        'conv5_kernel_num', conv5_kernel_num,
                        'FC1_units_num', fc1_units_num,
                        'FC2_units_num', fc2_units_num,
                        ])
    result_list.append(['train_step', 'train_loss', 'train_step', 'train_accuracy'])

    with tf.Session() as sess:
        sess.run(init)

        print('===>>>>>>开始在训练集上训练数据<<<<<<===')
        num_batches_per_epoch=int(num_example_per_epoch_for_train/batch_size)
        print('Per batch Size:',batch_size)
        print("Train sample Count Per Epoch:",num_example_per_epoch_for_train)
        print('Total batch Per Epoch:',num_batches_per_epoch)
        tf.train.start_queue_runners()
        training_step=0
        for epoch in range(training_epoch):
            for batch_idx in range(num_batches_per_epoch):
                images_batch,labels_batch=sess.run([images_train,labels_train])
                _,loss_value,avg_loss=sess.run([train_op,total_loss_op,average_loss_op],
                                               feed_dict={images_holder:images_batch,
                                                          label_holder:labels_batch,
                                                          learning_rate:learning_rate_init})
                training_step=sess.run(global_step)

                if training_step%display_step==0:
                    predictions=sess.run([top_K_op],
                                         feed_dict={images_holder:images_batch,
                                                    label_holder:labels_batch})
                    batch_accuracy=np.sum(predictions)/batch_size

                    print('Training Epoch:'+str(epoch)+',Training Step:'+str(training_step)+
                          ',Training Loss='+'{:.6f}'.format(loss_value)+
                          ',Training Accuracy='+'{:.5f}'.format(batch_accuracy))

                    result_list.append([training_step,loss_value,training_step,batch_accuracy])

                    summaries_str=sess.run(merged_summaries,
                                           feed_dict={images_holder:images_batch,
                                                      label_holder:labels_batch})

                    summary_writer.add_summary(summary=summaries_str,global_step=training_step)
                    summary_writer.flush()
        print('训练完毕！')

        print('===>>>>>>开始在测试集上测试模型<<<<<<==')
        num_batch_per_epoch=int(num_example_per_epoch_for_eval/batch_size)
        print('Per batch Size:',batch_size)
        print('Test sample Count Per Epoch:',num_example_per_epoch_for_eval)
        print('Total batch Count Per Epoch:',num_batches_per_epoch)

        correct_prediction=0.0

        for idx in range(num_batches_per_epoch):
            images_batch,labels_batch=sess.run([images_test,labels_test])
            predictions=sess.run([top_K_op],feed_dict={images_holder:images_batch,
                                                       label_holder:labels_batch})
            correct_prediction+=np.sum(predictions)

        accuracy_score=correct_prediction/num_example_per_epoch_for_eval
        print('-------->Test Accuracy: ',accuracy_score)
        result_list.append(['Test Accuracy: ',accuracy_score])

        result_file=open('evaluate_result.csv','w',newline='')
        csv_writer=csv.writer(result_file,dialect='excel')
        for row in result_list:
            csv_writer.writerow(row)
