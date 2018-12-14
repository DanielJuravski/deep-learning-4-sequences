import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a_file = 'stat/a.txt'
    b_file = 'stat/b.txt'
    c_file = 'stat/c.txt'
    d_file = 'stat/d.txt'

    a_data = np.loadtxt(a_file)
    b_data = np.loadtxt(b_file)
    c_data = np.loadtxt(c_file)
    d_data = np.loadtxt(d_file)

    plt_file_name = 'stat/all.png'
    plt.plot(a_data, label='a')
    plt.plot(b_data, label='b')
    plt.plot(c_data, label='c')
    plt.plot(d_data, label='d')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Iterations')
    # plt.xticks(np.arange(0, step=5))
    plt.title('All representations')
    plt.legend()
    plt.grid(True)
    plt.savefig(plt_file_name)
    plt.show()
    plt.close()

    pass