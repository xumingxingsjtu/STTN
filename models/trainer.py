# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    x_stats = inputs.get_stats()
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    #W=W[np.newaxis,np.newaxis,:,:]
    #W=np.tile(W,(batch_size,n_his+n_pred,1,1))
    #print("W:{}".format(W.shape))
    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob=tf.placeholder(tf.float32, name='keep_prob')
    is_training=tf.placeholder(tf.bool, name='is_training')
    # Define model loss
    train_loss, pred = build_model(x,x_stats,n_his, Ks, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)
    weight_loss=0.001*tf.add_n(tf.get_collection('weight_decay'))
    train_loss_w=train_loss+weight_loss
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    lr=tf.maximum(1e-6,lr)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss_w)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss_w)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')
        best=1e5
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                #print("x_batch:{}".format(x_batch.shape))
                #W_concat=np.tile(W,(x_batch.shape[0],n_his+n_pred,1,1))
                #x_st=np.concatenate((x_batch,W_concat),axis=3)    
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 0.8,is_training:True})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss_w,train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob:0.8,is_training:True})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f},{loss_value[1]:.3f}, {loss_value[2]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            min_va_val, min_val = \
                model_inference(sess, pred,inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')
            min_va_mse=[min_va_val[1],min_va_val[4],min_va_val[7]]
            val_mean=np.mean(min_va_mse)
            print('val_mean:{}'.format(val_mean))
            if val_mean<=best:
               best=val_mean
               print('best_validation:{}'.format(best))
               model_save(sess,global_steps,'Model-best-{}'.format(best))            
            #if (i + 1) % args.save == 0:
                #model_save(sess, global_steps, 'STTN-PeMS-M')
            #if (i + 1) % args.save == 0:
                #model_save(sess, global_steps, 'STTN-PeMS-M')
        writer.close()
    print('Training model finished!')
