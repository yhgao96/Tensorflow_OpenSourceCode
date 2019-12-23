import tensorflow as tf
import os
import csv
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_LOG_MIN_LEVEL']='2'

learning_rate_init=0.01
training_epoch=1
batch_size=100
display_step=10
keep_prob_init=0.8

n_input=784
n_class=10

def WeightVariable(shape,name_str,stddev=0.1):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def BiasVariable(shape,name_str,stddev=0.00001):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def Conv2d(x,w,b,stride=1,padding='SAME'):
    with tf.name_scope('wx_plus_b'):
        y=tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    return y

def Activation(x,activation=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y=activation(x)
    return y

def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding='VALID')  #K代表过滤器的大小，stride代表步长的大小

def FullyConnected(x,w,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_plus_b'):
        y=tf.matmul(x,w)
        y=tf.add(y,b)
    with tf.name_scope(act_name):
        y=activate(y)
    return y

def EvaluateModelOnDataset(sess,images,labels):
    n_samples=images.shape[0]
    per_batch_size=batch_size
    loss=0
    acc=0

    if(n_samples<=per_batch_size):
        batch_count=1
        loss,acc=sess.run([cross_entry_loss,accuracy],
                          feed_dict={x_origin:images,y_true:labels,
                                     learning_rate:learning_rate_init})
    else:
        batch_count=int(n_samples/per_batch_size)
        batch_start=0
        for idx in range(batch_count):
            batch_loss,batch_acc=sess.run([cross_entry_loss,accuracy],
                                          feed_dict={x_origin:images[batch_start:batch_start+per_batch_size,:],
                                                     y_true:labels[batch_start:batch_start+per_batch_size,:],
                                                     learning_rate:learning_rate_init,
                                                     keep_prob:keep_prob_init})
            batch_start+=per_batch_size
            loss+=batch_loss
            acc+=batch_acc

    return loss/batch_count,acc/batch_count


with tf.Graph().as_default():
    with tf.name_scope('Input'):
        x_origin=tf.placeholder(tf.float32,[None,n_input],name='x_origin')
        y_true=tf.placeholder(tf.float32,[None,n_class],name='y_true')

        x_image=tf.reshape(x_origin,[-1,28,28,1])

    with tf.name_scope('Inference'):
        with tf.name_scope('Conv2d'):
            conv1_kernel_num=16  #1,2,4,8,10,12,16,24,32
            weights=WeightVariable(shape=[5,5,1,conv1_kernel_num],name_str='weights')
            biases=BiasVariable(shape=[conv1_kernel_num],name_str='biases')
            conv_out=Conv2d(x_image,weights,biases,stride=1,padding='VALID')

        with tf.name_scope('Activate'):
            activate_out=Activation(conv_out,activation=tf.nn.relu,name='relu')  #改变网络激活函数观察现象

        with tf.name_scope('Pool2d'):
            pool_out=Pool2d(activate_out,pool=tf.nn.max_pool,k=2,stride=2)

        with tf.name_scope('FeastReshape'):
            feature=tf.reshape(pool_out,[-1,12*12*conv1_kernel_num])  #将二维图转化为一维特征向量
        ##
        ##
        with tf.name_scope('FC_Relu'):
            fcl_units_num=100    ###改变神经元数量观察现象
            weights=WeightVariable(shape=[12*12*conv1_kernel_num,fcl_units_num],name_str='weight')
            biases=BiasVariable(shape=[fcl_units_num],name_str='biases')
            fcl_out=FullyConnected(feature,weights,biases,activate=tf.nn.relu,act_name='Relu')
        ##
        ##
        with tf.name_scope('Dropout'):
            keep_prob=tf.placeholder(tf.float32)
            fc_dropout=tf.nn.dropout(fcl_out,keep_prob=keep_prob)

        with tf.name_scope('FC_Linear'):
            weights=WeightVariable(shape=[fcl_units_num,n_class],name_str='weights')
            biases=BiasVariable(shape=[n_class],name_str='biases')
            Y_pred_logist=FullyConnected(fc_dropout,weights,biases,activate=tf.identity,act_name='identify')

    with tf.name_scope('Loss'):
            cross_entry_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=Y_pred_logist))

    with tf.name_scope('Train'):
            learning_rate=tf.placeholder(tf.float32)
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step=tf.Variable(0,name='global_step',trainable=False,dtype=tf.int64)
            trainer=optimizer.minimize(cross_entry_loss,global_step=global_step)

    with tf.name_scope('Evaluate'):
            correct_pred=tf.equal(tf.argmax(Y_pred_logist,1),tf.argmax(y_true,1))
            accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

    init=tf.global_variables_initializer()

    print('计算图已经写入事件文件，请在Tensorboard中查看！')

    summary_writer=tf.summary.FileWriter(logdir='LOG_SimpleConvNet',graph=tf.get_default_graph())
    summary_writer.flush()

    mnist=input_data.read_data_sets('mnist_data',one_hot=True)

    result_list=list()
    result_list.append(['learning_rate',learning_rate_init,
                        'training_epoch',training_epoch,
                        'batch_size',batch_size,
                        'display_step',display_step])
    result_list.append(['train_step','train_loss','validation_loss',
                        'train_step','train_accuracy','validation_accuracy'])

    with tf.Session() as sess:
        sess.run(init)
        total_batches=int(mnist.train.num_examples/batch_size)  #55000/100
        print('Per batch size:',batch_size)
        print('Train sample count:',mnist.train.num_examples)
        print('Total batch:',total_batches)
        training_step=0

        for epoch in range(training_epoch):     #training_epoch=1
            for batch_idx in range(total_batches):
                batch_x,batch_y=mnist.train.next_batch(batch_size)    #batch_x,batch_y是one_hot
                sess.run(trainer,feed_dict={x_origin:batch_x,y_true:batch_y,
                                            learning_rate:learning_rate_init,keep_prob:keep_prob_init})
                #training_step+=1        #每调用一次训练节点，training_step就加1，最终==training_epoch*total_batch
                #每调用一次训练节点，training_step就加1，最终=training_epoch*total_batch
                training_step=sess.run(global_step)

                if training_step%display_step==0:
                    start_idx=max(0,(batch_idx-display_step)*batch_size)
                    end_idx=batch_idx*batch_size
                    training_loss,train_acc=EvaluateModelOnDataset(sess,mnist.train.images[start_idx:end_idx,:],mnist.train.labels[start_idx:end_idx,:])

                    print('Training step:'+str(training_step)+
                          ',Training Loss='+'{:.6f}'.format(training_loss)+
                          ',Training Accuracy='+'{:.5f}'.format(train_acc))

                    validation_loss,validation_acc=EvaluateModelOnDataset(sess,
                                                                          mnist.validation.images,
                                                                          mnist.validation.labels)
                    print('Training step:'+str(training_step)+',Training Loss='+'{:.6f}'.format(validation_loss)+
                          ',Training Accuracy='+'{:.5f}'.format(validation_acc))

                    result_list.append([training_step,training_loss,validation_loss,
                                        training_step,train_acc,validation_acc])
        print('训练完毕！！')

        test_samples_count=mnist.test.num_examples
        test_loss,test_accuracy=EvaluateModelOnDataset(sess,mnist.test.images,mnist.test.labels)
        print('Testing sample count:',test_samples_count)
        print('Test Lost:',test_loss)
        print('Test Accuracy:',test_accuracy)
        result_list.append(['test step','loss',test_loss,'accuracy',test_accuracy])

        result_file=open('evaluate_result.csv','w',newline='')
        csv_writer=csv.writer(result_file,dialect='excel')
        for row in result_list:
            csv_writer.writerow(row)
