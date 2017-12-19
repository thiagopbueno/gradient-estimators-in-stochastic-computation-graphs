import matplotlib.pyplot as plt
import numpy as np

red = '#c0392b'
blue = '#2980b9'
green = '#2ecc71'

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
    plt.figure(figsize=(20, 5))
    plt.plot(parameters, losses, color=red)
    plt.title('Expected Loss')
    plt.xlabel('$\\theta$')
    plt.ylabel('$\mathcal{L}(\\theta)$')
    plt.grid()
    plt.show()

def plot_results(losses, thetas=None, grads=None):
    fig = plt.figure(figsize=(20, 3))

    plt.subplot(131)
    plt.plot(losses, color=red)
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('$\\mathcal{L}(\\theta)$')
    plt.grid()

    if thetas is not None:
        plt.subplot(132)
        if not isinstance(thetas, np.ndarray):
            thetas = [thetas]
        for i, params in enumerate(thetas):
            plt.plot(params, color=green)
        plt.title('Parameter $\\theta$')
        plt.xlabel('Epoch')
        plt.ylabel('$\\theta$')
        plt.grid()

    if grads is not None:
        plt.subplot(133)
        plt.plot(grads, color=blue)
        plt.title('Gradient $\\nabla_{\\theta} \\mathcal{L}(\\theta)$')
        plt.xlabel('Epoch')
        plt.ylabel('$\\nabla_{\\theta} \\mathcal{L}$')
        plt.grid()

    plt.show()


# 1-step Navigation 

def build_initial_state(x0, y0, batch_size):
    x_init = np.full([batch_size], x0, np.float32)
    y_init = np.full([batch_size], y0, np.float32)
    initial_state   = np.stack([x_init, y_init], axis=1)
    return initial_state

def build_timesteps(batch_size, max_time):
    timesteps = [np.arange(start=1.0, stop=max_time + 1.0, dtype=np.float32)]
    timesteps = np.repeat(timesteps, batch_size, axis=0)
    timesteps = np.reshape(timesteps, (batch_size, max_time, 1))
    return timesteps


def plot_loss_function(losses, epoch, title=None):
    plt.plot(losses, 'b-')
    plt.xlim(0, epoch)
    if title is None:
        plt.title('Loss function')
    else:
        plt.title('Loss function ({})'.format(title))
    plt.xlabel("# iterations")
    plt.ylabel("loss")
    plt.grid()

def plot_grid(ax, grid):
    xlim, ylim = grid['size']
    ax.axis([0.0, xlim, 0.0, ylim])
    ax.set_aspect('equal')
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()
    

def plot_policy(grid, action, states, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
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
