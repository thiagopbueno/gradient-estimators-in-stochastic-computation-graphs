import matplotlib.pyplot as plt


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


def plot_results(*args):
    fig = plt.figure(figsize=(15, 3))
    n = len(args)
    for i, result in enumerate(args):
        values, title, xlabel, ylabel = result
        ax = fig.add_subplot(1, n, i+1)
        plot_function(ax, range(len(values)), values, title, xlabel, ylabel)
