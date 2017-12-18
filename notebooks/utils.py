import matplotlib.pyplot as plt
import numpy as np


def plot_function(ax, x, y, title, xlabel, ylabel):
    ax.plot(x, y, 'b-')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()


def plot_hist(ax, ys, title, xlabel):
    for y, ylabel in ys:
        ax.hist(y, bins=1000, histtype='step', normed=True, label=ylabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid()
    ax.legend()


# def plot_results(*args):
#     fig = plt.figure(figsize=(15, 3))
#     n = len(args)
#     for i, result in enumerate(args):
#         values, title, xlabel, ylabel = result
#         ax = fig.add_subplot(1, n, i+1)
#         plot_function(ax, range(len(values)), values, title, xlabel, ylabel)

def plot_expected_loss(parameters, losses):
    plt.plot(parameters, losses, 'b-')
    plt.title('Expected Loss')
    plt.xlabel('$\\theta$')
    plt.ylabel('$\mathcal{L}(\\theta)$')
    plt.grid()
    plt.show()

def plot_results(losses, thetas, grads):
    fig = plt.figure(figsize=(20, 3))

    plt.subplot(131)
    plt.plot(losses)
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('$\\mathcal{L}(\\theta)$')
    plt.grid()

    plt.subplot(132)
    plt.plot(thetas)
    plt.title('Parameter $\\theta$')
    plt.xlabel('Epoch')
    plt.ylabel('$\\theta$')
    plt.grid()

    plt.subplot(133)
    plt.plot(grads)
    plt.title('Gradient $\\nabla_{\\theta} \\mathcal{L}(\\theta)$')
    plt.xlabel('Epoch')
    plt.ylabel('$\\nabla_{\\theta} \\mathcal{L}$')
    plt.grid()

    plt.show()


# 1-step Navigation 

def plot_loss_function(ax, losses, epoch, title=None):
    ax.plot(losses, 'b-')
    ax.set_xlim(0, epoch)
    if title is None:
        ax.set_title('Loss function')
    else:
        ax.set_title('Loss function ({})'.format(title))
    ax.set_xlabel("# iterations")
    ax.set_ylabel("loss")
    ax.grid()

def plot_grid(ax, grid):
    xlim, ylim = grid['size']
    ax.axis([0.0, xlim, 0.0, ylim])
    ax.set_aspect('equal')
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()
    

def plot_policy(ax, grid, action, states):
    
    start = grid['start']
    end = grid['goal']

    # title
    ax.set_title("Policy (1-step)", fontweight="bold", fontsize=16)
    
    # plot grid
    plot_grid(ax, grid)
        
    # plot action
    ax.quiver([start[0]], [start[1]], [action[0]], [action[1]],
              angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=0.005)

    # plot states
    [x, y] = np.split(states, 2, axis=1)
    ax.scatter(x, y, marker=".", c="m", linewidths="1")
    
    # plot start and goal
    ax.plot([start[0]], [start[1]], marker='X', markersize=15, color='limegreen', label='initial')
    ax.plot([end[0]],   [end[1]],   marker='X', markersize=15, color='crimson',   label='goal')
