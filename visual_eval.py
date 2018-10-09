# coding='utf-8'
import matplotlib
matplotlib.use('Agg')
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

from tensorflow.examples.tutorials.mnist import input_data



def get_data(x):
    data = x
    n_samples, n_features = x.shape
    return data, n_samples, n_features


def plot_embedding(data, label_nat, label_adv, label_mix, label_random, label_random_10,label_random_30, label_true):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(331)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_true[i]),
                 color=plt.cm.Set1(label_true[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("true")



    ax2 = plt.subplot(332)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_nat[i]),
                 color=plt.cm.Set1(0.9) if label_nat[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("natural")



    ax2 = plt.subplot(333)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_adv[i]),
                 color=plt.cm.Set1(0.9) if label_adv[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("adv_trained")


    ax2 = plt.subplot(334)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_mix[i]),
                 color=plt.cm.Set1(0.9) if label_mix[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("mix_trained")


    ax2 = plt.subplot(335)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_random[i]),
                 color=plt.cm.Set1(0.9) if label_random[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("random_trained")


    ax3 = plt.subplot(336)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_random_10[i]),
                 color=plt.cm.Set1(0.9) if label_random_10[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("random_10")

    # plt.show()

    ax3 = plt.subplot(337)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label_random_30[i]),
                 color=plt.cm.Set1(0.9) if label_random_30[i] == label_true[i] else plt.cm.Set1(0),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("random_30")


    plt.savefig(config['pic_dir'])


def main(x,label_nat, label_adv, label_mix, label_random, label_random_10,label_true):

    data, n_samples, n_features = get_data(x)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_embedding(result, label_nat, label_adv, label_mix, label_random, label_random_10, label_true)



if __name__ == '__main__':
    import json

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = config['model_dir']
    x_adv = np.load(config['store_adv_path'])
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_nat = mnist.test.images[:10000]
    label_nat = np.load(config['label_nat'])
    label_adv = np.load(config['label_adv'])
    label_mix = np.load(config['label_mix'])
    label_random = np.load(config['label_random'])
    label_random_10 = np.load(config['label_random_10'])
    label_random_30 = np.load(config['label_random_30'])
    label_true = mnist.test.labels[:10000]


    main(x_adv,label_nat, label_adv, label_mix, label_random, label_random_10,label_random_30, label_true)
