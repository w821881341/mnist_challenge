"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x

  def all_perturb(self, x_nat,y,sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(10):
      y_i = np.full_like(y,i)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y_i})

      x -= self.epsilon * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


  def avg_perturb(self, x_nat,y,sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    grad_all = np.full_like(x_nat,0)
    for i in range(10):
      y_i = np.full_like(y, i)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y_i})
      grad_all += grad

    x -= self.epsilon * np.sign(grad_all)

    x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
    x = np.clip(x, 0, 1) # ensure valid pixel range

    return x

  def logsoftmax(self,x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

  def get_normalized_vector(self,d):
    # print(range(1, len(d.get_shape())))
    d /= (1e-12 + tf.reduce_max(tf.abs(d), 1, keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0),1, keep_dims=True))
    return d

  def kl_divergence_with_logit(self,q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * self.logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * self.logsoftmax(p_logit), 1))
    return qlogq - qlogp

  def generate_virtual_adversarial_perturbation(self,x, sess):
    x_adv = sess.run(self.model.gen_virtual_gradient(), feed_dict={self.model.x_input:x })

    return x_adv

if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
