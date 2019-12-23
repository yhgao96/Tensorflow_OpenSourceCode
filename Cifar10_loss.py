import tensorflow as tf
import os
import csv
import numpy as np
import sys
from six.moves import urllib
import tarfile
import cifar10_input
os.environ['TF_CPP_LOG_MIN_LEVEL']='2'

#Hyperparameters
learning_rate_init=0.01
batch_size=100
training_epoch=10
display_step=50
conv1_kernel_num=32
conv2_kernel_num=32
fc1_units_num=250
fc2_units_num=150
l2loss_ratio=0.00001#0.15 0.1 0.05 0.01 0.001 0.0001 0.00001 0

dataset_dir='../CIFAR10_dataset'
num_examples_per_epoch_for_train=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size=cifar10_input.IMAGE_SIZE
image_channel=3
n_classes=10

def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir # /tmp/cifar10_data
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # 从URL中获得文件名
    filename = DATA_URL.split('/')[-1]
    # 合并文件路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 定义下载过程中打印日志的回调函数
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        # 下载数据集
        filepath, _ =urllib.request.urlretrieve(DATA_URL, filepath,reporthook=_progress)
        print()
        # 获得文件信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        # 解压缩
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def get_distorted_train_batch(data_dir,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir=os.path.join(data_dir,'cifar-10-batches-bin')
    images,labels=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
    return images,labels

def get_undistorted_eval_batch(data_dir,eval_data,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir=os.path.join(data_dir,'cifar-10-batches-bin')
    images,labels=cifar10_input.inputs(eval_data=eval_data,
                                                data_dir=data_dir,
                                                batch_size=batch_size)
    return images,labels

def WeightVariable(shape,name_str,stddev=0.1):
    initial=tf.truncated_normal(shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def BiasVariable(shape,name_str,init_value=0.0):
    initial=tf.constant(init_value,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def Conv2d(x,w,b,stride=1,padding='SAME',activation=tf.nn.relu,act_name='Relu'):
    with tf.name_scope('conv2d_bias'):
        y=tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    with tf.name_scope(act_name):
        y=activation(y)
    return y

def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2,padding='SAME'):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding=padding)
    # ksize是池化窗口的大小，不想在batch和channel上做池化，所以前后两个数字为1

def FullyConnected(x,w,b,activate=tf.nn.relu,act_name='Relu'):
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

def Inference(image_holder):
    activation_func=tf.nn.relu
    activation_name='relu'
    #first conv net
    with tf.name_scope('conv2d_1'):
        weight=WeightVariable(shape=[5,5,image_channel,conv1_kernel_num],
                              name_str='weight',stddev=5e-2)
        biases=BiasVariable(shape=[conv1_kernel_num],name_str='biases',init_value=0.0)
        conv1_out=Conv2d(image_holder,weight,biases,stride=1,padding='SAME',
                         activation=activation_func,act_name=activation_name)
        AddActivationSummary(conv1_out)
    with tf.name_scope('Pool2d_1'):
        pool1_out=Pool2d(conv1_out,pool=tf.nn.max_pool,k=2,stride=2,padding='SAME')
    with tf.name_scope('conv2d_2'):
        weight=WeightVariable(shape=[5,5,conv1_kernel_num,conv2_kernel_num],
                              name_str='weight',stddev=5e-2)
        biases=BiasVariable(shape=[conv2_kernel_num],name_str='biases',init_value=0.0)
        conv2_out=Conv2d(pool1_out,weight,biases,stride=1,padding='SAME',
                         activation=activation_func,act_name=activation_name)
        AddActivationSummary(conv2_out)
    with tf.name_scope('Pool2d_2'):
        pool2_out=Pool2d(conv2_out,pool=tf.nn.max_pool,k=2,stride=2,padding='SAME')
    with tf.name_scope('FeatsReshape'):
        feature=tf.reshape(pool2_out,[batch_size,-1])  #转换为一个列向量
        feats_dim=feature.get_shape()[1].value  #get_shape()[0]返回的是batch_size,如果是1的话，返回的是那一列行向量的索引值
    with tf.name_scope('FC1_nonlinear'):
        weight=WeightVariable(shape=[feats_dim,fc1_units_num],name_str='weight',stddev=4e-2)
        biases=BiasVariable(shape=[fc1_units_num],name_str='biases',init_value=0.1)
        fc1_out=FullyConnected(feature,weight,biases,activate=activation_func,act_name=activation_name)
        AddActivationSummary(fc1_out)
        with tf.name_scope('L2_loss'):
            weight_loss=tf.multiply(tf.nn.l2_loss(weight),l2loss_ratio,name='fc1_weight_loss')
            tf.add_to_collection('losses',weight_loss)
    with tf.name_scope('FC2_nonlinear'):
        weight=WeightVariable(shape=[fc1_units_num,fc2_units_num],name_str='weight',stddev=4e-2)
        biases=BiasVariable(shape=[fc2_units_num],name_str='biases',init_value=0.1)
        fc2_out=FullyConnected(fc1_out,weight,biases,activate=activation_func,act_name=activation_name)
        AddActivationSummary(fc2_out)
        with tf.name_scope('l2_loss'):
            weight_loss=tf.multiply(tf.nn.l2_loss(weight),l2loss_ratio,name='fc2_weight_loss')
            tf.add_to_collection('losses',weight_loss)
    with tf.name_scope('FC3_linear'):
        weight=WeightVariable(shape=[fc2_units_num,n_classes],name_str='weight',stddev=1.0/fc2_units_num)
        biases=BiasVariable(shape=[n_classes],name_str='biases',init_value=0.0)
        logits=FullyConnected(fc2_out,weight,biases,activate=tf.identity,act_name='identity')
        AddActivationSummary(logits)
        return logits

def TrainModel():
    with tf.Graph().as_default():
        with tf.name_scope('Input'):
            images_holder=tf.placeholder(tf.float32,[batch_size,image_size,image_size,image_channel],name='images')
            label_holder=tf.placeholder(tf.int32,[batch_size],name='label')  #label还没有进行one-hot编码
        with tf.name_scope('Inference'):
            logits=Inference(images_holder)
        with tf.name_scope('LOSS'):
            #总体损失（total loss）=交叉熵损失+所有权重的L2损失
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_holder,logits=logits)
                               #sparse会把one_hot编码过程在内部进行
            cross_entropy_loss=tf.reduce_mean(cross_entropy,name='xentrypy_loss')
            tf.add_to_collection('losses',cross_entropy_loss)
            total_loss=tf.add_n(tf.get_collection('losses'),name='total_loss')
            average_losses=AddLossSummary(tf.get_collection('losses')+[total_loss])
        with tf.name_scope('Train'):
            learning_rate=tf.placeholder(tf.float32)
            global_step=tf.Variable(0,name='global_step',trainable=False,dtype=tf.int64)
            #optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate_init,momentum=0.9)
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate_init)
            # optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_init)
            # optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate_init)
            # optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate_init)
            train_op=optimizer.minimize(total_loss,global_step=global_step)
        with tf.name_scope('Evaluate'):
            top_k_op=tf.nn.in_top_k(predictions=logits,targets=label_holder,k=1)
            #定义获取训练样本批次的计算节点
        with tf.name_scope('GetTrainBatch'):
            image_train,label_train=get_distorted_train_batch(data_dir=dataset_dir,batch_size=batch_size)
        with tf.name_scope('GetTestBatch'):
            image_test,label_test=get_undistorted_eval_batch(eval_data=True,data_dir=dataset_dir,batch_size=batch_size)

        merged_summaries=tf.summary.merge_all()

        init=tf.global_variables_initializer()

        print('计算图构建完毕，可以在TensorBoard中察看！！')
        summary_writer=tf.summary.FileWriter(logdir='log_cifar10',graph=tf.get_default_graph())
        summary_writer.flush()
        #将评估结果保存到文件
        result_list=list()
        #写入参数配置
        result_list.append(['learning_rate',learning_rate_init,
                            'training_epoch',training_epoch,
                            'batch_size',batch_size,
                            'display_step',display_step,
                            'conv1_kernel_num',conv1_kernel_num,
                            'conv2_kernel_num',conv2_kernel_num,
                            'fc1_units_num',fc1_units_num,
                            'fc2_units_num',fc2_units_num])
        result_list.append(['train_step','train_loss','train_step','train_accuracy'])


        #Run Graph
        with tf.Session() as sess:
            sess.run(init)
            print('==>>>>>>开始在训练集上训练模型<<<<<<==')
            total_batch=int(num_examples_per_epoch_for_train/batch_size)
            print('Train batch Size:',batch_size)
            print('Train sample Count Per Epoch:',num_examples_per_epoch_for_train)
            print('Total batch Count Per Epoch:',total_batch)
            #启动数据读取队列
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)
            #记录模型被训练的步数
            training_step=0
            #运行指定轮数，每一轮的训练样本总数为：num_example_per_epoch_for_train
            for epoch in range(training_epoch):
                for step in range(total_batch):
                    images_batch,labels_batch=sess.run([image_train,label_train])
                    #运行优化器训练节点
                    _,loss_value,avg_losses=sess.run([train_op,total_loss,average_losses],
                                           feed_dict={images_holder:images_batch,label_holder:labels_batch,learning_rate:learning_rate_init})
                    training_step=sess.run(global_step)
                    if training_step%display_step==0:
                        prediction=sess.run([top_k_op],feed_dict={images_holder:images_batch,label_holder:labels_batch})
                        batch_accuracy=np.sum(prediction)/batch_size
                        result_list.append([training_step,loss_value,training_step,batch_accuracy])
                        print('Training step:',str(training_step)+
                              ',Training Loss='+'{:.6f}'.format(loss_value)+
                             ',Training Accuracy='+'{:.5f}'.format(batch_accuracy))

                        #运行汇总节点
                        summaries_str=sess.run(merged_summaries,feed_dict={images_holder:images_batch,label_holder:labels_batch})
                        summary_writer.add_summary(summary=summaries_str,global_step=training_step)
                        summary_writer.flush()


            print('训练完毕！！')

            print('==>>>>>>开始在测试集上面评估模型<<<<<<==')
            total_batch=int(num_examples_per_epoch_for_eval/batch_size)
            total_example=total_batch*batch_size
            print('Per batch Size:',batch_size)
            print('Test sample Count Per epoch:',total_example)
            print('Total batch Count Per epoch:',total_batch)
            correct_predicted=0
            for test_step in range(total_batch):
                images_batch,labels_batch=sess.run([image_test,label_test])
                prediction=sess.run([top_k_op],feed_dict={images_holder:images_batch,label_holder:labels_batch})
                correct_predicted+=np.sum(prediction)
                # print('test step:', str(test_step) +
                #       ',prediction=' + '{:.5f}'.format(np.sum(prediction)/batch_size))
            accuracy_score=correct_predicted/total_example
            # print('测试完毕，Top_K(K=1)的准确率:' + '{:.5f}'.format(accuracy_score))
            print('------->Accuracy on Test Examples:',accuracy_score)
            result_list.append(['Accuracy on Test Example:',accuracy_score])
        #将评估结果保存到文件
        result_file=open('evaluate_result.csv','w',newline='')
        csv_writer=csv.writer(result_file,dialect='excel')
        for row in result_list:
            csv_writer.writerow(row)
def main(argv=None):
    maybe_download_and_extract(data_dir=dataset_dir)
    train_dir='/log_cifar10'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    TrainModel()

if __name__=='__main__':
    tf.app.run()