import matplotlib.pyplot as plt

def plot_results(*args):
    fig = plt.figure(figsize=(15, 3))
    n = len(args)
    for i, result in enumerate(args):
        values, title, xlabel, ylabel = result
        plt.subplot(1, n, i+1)
        plt.plot(values, 'b-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
