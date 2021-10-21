# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.modules import *
from os.path import join as pjoin
import tensorflow as tf

def masked_mae(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss *mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)
'''def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Ks - 1)

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred'''
def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv
    
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm_atten(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor
  
def embedding_postprocessor(input_tensor,
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=512,
                            dropout_prob=0.1,training=True):
                            
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  if training==False:
      dropout_prob=0.0
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]
  sp_ini=tf.get_collection('initial_spatial_embeddings')
  output = input_tensor
  if use_position_embeddings:
    assert_op = tf.assert_equal(seq_length,embeddings_length)
    with tf.control_dependencies([assert_op]):
      if position_embedding_name=='spatial_position_embeddings':
          full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[embeddings_length, width],
          initializer=tf.constant_initializer(sp_ini))
      else:
          full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[embeddings_length, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output

def build_model(inputs, x_stats,n_his, Ks, blocks, keep_prob,is_training=True):
    '''
        Build the base model.
        :param inputs: placeholder.
        :param n_his: int, size of historical records for training.
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param blocks: list, channel configs of st_conv blocks.
        :param keep_prob: placeholder.
        '''
    x = inputs[:, 0:n_his, :, :]
    B,T,n,c_in=x.get_shape().as_list()
    #c_out=16
    # ST-Block
    with tf.variable_scope('aggregation'):
        w1 = tf.get_variable(name='w1', shape=[1,1,1, 32], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
        #variable_summaries(ws, 'theta')
        b1 = tf.get_variable(name='b1', initializer=tf.zeros([32]), dtype=tf.float32)
        x=tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
    
    '''with tf.variable_scope('topology_encode'): 
        w1 = tf.get_variable(name='w1', shape=[Ks * c_in, 32], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w1))
        #variable_summaries(ws, 'theta')
        bs = tf.get_variable(name='bs', initializer=tf.zeros([32]), dtype=tf.float32)
        # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
        x = gconv(tf.reshape(x, [-1, n, c_in]), w1, Ks, c_in, 32) + bs'''
        #print('x_gconv:{}'.format(x_gconv))
        #x=tf.concat([tf.reshape(x_gconv,[-1,T,n,c_out]),x],axis=3)
        #print("x:{}".format(x))
    #x=tf.reshape(x,[-1,T,n,32])
    '''B,T,n,c_in=x.get_shape().as_list()
    print(x)
    with tf.variable_scope('spatial_embedding'):
         x=tf.reshape(x,[-1,n,c_in])
         x=embedding_postprocessor(x,
                            use_position_embeddings=True,
                            position_embedding_name="spatial_position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=n,
                            dropout_prob=0.1,training=is_training)
         x=tf.reshape(x,[-1,T,n,c_in])
    with tf.variable_scope('temporal_embedding'):
         #x=tf.reshape(x,[-1,n,c_in])
         x=tf.transpose(x,[0,2,1,3])
         x=tf.reshape(x,[-1,T,c_in])
         x=embedding_postprocessor(x,
                            use_position_embeddings=True,
                            position_embedding_name="temporal_position_embeddings",
                            initializer_range=0.02,
                            embeddings_length=T,
                            dropout_prob=0.1,training=is_training)
         x=tf.reshape(x,[-1,n,T,c_in])
         x=tf.transpose(x,[0,2,1,3])'''
                                     
    for i, channels in enumerate(blocks):
        x = sttn_conv_block(x, Ks, channels, i, keep_prob,is_training)
    
    # Output Layer
    y = output_layer(x, 'output_layer',keep_prob,is_training=is_training)
    y_t=y*x_stats['std']+x_stats['mean']
    copy=inputs[:, n_his - 1:, :, :]*x_stats['std']+x_stats['mean']
    labels=inputs[:, n_his:n_his+1, :, :]*x_stats['std']+x_stats['mean']
    tf.add_to_collection(name='copy_loss',
                         value=tf.reduce_mean(tf.abs(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :])))
    train_loss = masked_mae(y_t,labels)
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred

def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
