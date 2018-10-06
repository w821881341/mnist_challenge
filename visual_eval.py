# coding='utf-8'
import matplotlib
matplotlib.use('Agg')
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

from tensorflow.examples.tutorials.mnist import input_data



def get_data(x,y):
    data = x
    label = y
    n_samples, n_features = x.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.show()
    plt.savefig(config['pic_dir'])


def main(x,y):

    data, label, n_samples, n_features = get_data(x,y)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))



if __name__ == '__main__':
    import json

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = config['model_dir']
    x_adv = np.load(config['store_adv_path'])
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_nat = mnist.test.images[:10000]
    y = np.load(config['store_y_path'])


    main(x_adv,y)
