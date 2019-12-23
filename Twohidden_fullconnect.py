import os.path
import tensorflow as tf
import os
import sys
from six.moves import xrange
import time
import argparse

from tensorflow.examples.tutorials.mnist import input_data
import TwoLayer_softmax

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#定义模型的超参数
FLAGS=None

def fill_feed_dict(data_set,image_pl,label_pl):
    image_feed,label_feed=data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
    feed_dict={image_pl:image_feed,label_pl:label_feed}
    return feed_dict

def do_eval(sess,eval_correct,image_placeholder,label_placeholder,data_set):
    #运行一个回合的评估过程
    true_count=0
    steps_per_epoch=data_set.num_examples//FLAGS.batch_size  #每个回合的执行步数
    num_examples=steps_per_epoch*FLAGS.batch_size    #样本总量
    #累加每个批次样本中预测正确的样本数量
    for step in xrange(steps_per_epoch):
        feed_dict=fill_feed_dict(data_set,image_placeholder,label_placeholder)
        true_count+=sess.run(eval_correct,feed_dict=feed_dict)
    #所有批次上面的精确度
    precision=float(true_count)/num_examples
    print('Num examples: %d Num correct: %d Precision @ 1:%0.04f' % (num_examples,true_count,precision))





def run_training():
    data_sets=input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    with tf.Graph().as_default():
        image_palceholder, labels_placeholder =TwoLayer_softmax.placeholder_inputs(FLAGS.batch_size)

        logits=TwoLayer_softmax.inference(image_palceholder,FLAGS.hidden1,FLAGS.hidden2)

        loss=TwoLayer_softmax.loss(logits,labels_placeholder)

        trainOp=TwoLayer_softmax.training(loss,FLAGS.learning_rate)

        eval_correct=TwoLayer_softmax.evaluate(logits,labels_placeholder)

        merge_summary = tf.summary.merge_all()

        init=tf.global_variables_initializer()

        saver=tf.train.Saver()

        with tf.Session() as sess:
            summary_writer=tf.summary.FileWriter(FLAGS.log_dir,graph=tf.get_default_graph())

            sess.run(init)

            for step in xrange(FLAGS.max_step):
                start_time=time.time()

                feed_dict=fill_feed_dict(data_sets.train,image_palceholder,labels_placeholder)

                _,loss_value=sess.run([trainOp,loss],feed_dict=feed_dict)

                duration=time.time()-start_time

                if step%100==0:
                    print('Step %d: loss = %.2f(%.3f sec)'%(step,loss_value,duration))
                    summaries_str=sess.run(merge_summary,feed_dict=feed_dict)
                    summary_writer.add_summary(summaries_str,global_step=step)
                    summary_writer.flush()

                if (step+1)%1000==0 or (step+1)==FLAGS.max_step:
                    checkpoint_file=os.path.join(FLAGS.log_dir,'model.ckpt')
                    saver.save(sess,checkpoint_file,global_step=step)

                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_palceholder,
                            labels_placeholder,
                            data_sets.train
                    )

                    print('Validation Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_palceholder,
                            labels_placeholder,
                            data_sets.validation
                    )

                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_palceholder,
                            labels_placeholder,
                            data_sets.test
                    )


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.5,
        help='initial learning rate'
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.Must divide evenly into the dataset size.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='mnist_data/',
        help='Directory to put the input data'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='LOG_Twohidden_fullconnect',
        help='Directory to input the log data'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true,uses fake data for unit testing',
        action='store_true'
    )

    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
