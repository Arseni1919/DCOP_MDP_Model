import matplotlib.pyplot as plt

from globals import *


class SquareEnv:
    def __init__(self, side_size=4):
        self.side_size = side_size
        self.fig, self.ax = plt.subplots(2, 1)
        self.field = np.zeros((self.side_size, self.side_size))

    def render(self):
        self.plot_field(self.ax[0])
        plt.pause(0.001)

    def plot_field(self, ax):
        ax.cla()
        # self.field[0, 0] = np.random.random()
        # self.field[3, 3] = np.random.random()
        ax.imshow(self.field)
        ax.invert_yaxis()
        # ax.plot([i for i in range(self.side_size)])
        # ax.set_xlim(0, self.side_size)
        # ax.set_ylim(0, self.side_size)


def main():
    square_env = SquareEnv()

    for step in range(100):
        print(f'\r| step {step} |', end='')
        square_env.render()

    plt.show()


if __name__ == '__main__':
    main()
