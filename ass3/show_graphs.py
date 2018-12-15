import matplotlib.pyplot as plt
import numpy as np
import sys


def show_graph(train_loss, train_acc, dev_loss, dev_acc, test_acc, name):
    test_acc_str = "test accuracy = " + str(test_acc[0])
    plt.suptitle(name + " | " + test_acc_str)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss")
    plt.plot(dev_loss, label="validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # axes.setyticks(1,100,1)
    plt.plot(train_acc, label="train accuracy")
    plt.plot(dev_acc, label="validation accuracy")
    plt.yticks(np.arange(0,101,10))
    plt.legend()
    plt.savefig(name+".png")


if __name__ == '__main__':
    title = sys.argv[1]
    train_loss_f = sys.argv[2]
    dev_loss_f = sys.argv[3]
    train_acc_f = sys.argv[4]
    dev_acc_f = sys.argv[5]
    test_acc_f = sys.argv[6]

    train_loss  =np.load(train_loss_f)
    dev_loss    =np.load(dev_loss_f)
    train_acc   =np.load(train_acc_f)
    dev_acc     =np.load(dev_acc_f)
    test_acc    =np.load(test_acc_f)
    show_graph(train_loss, train_acc, dev_loss, dev_acc, test_acc, title)