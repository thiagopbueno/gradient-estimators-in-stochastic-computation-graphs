import matplotlib.pyplot as plt

def plot_results(parameter, losses, surrogate=None):
    fig = plt.figure(figsize=(15, 3))

    plt.subplot(131)
    plt.plot(parameter, 'b-')
    plt.title('parameter')
    plt.ylabel('$\\theta$')
    plt.grid()

    plt.subplot(132)
    plt.plot(losses, 'r-')
    plt.title('losses')
    plt.grid()

    if surrogate is not None:
        plt.subplot(133)
        plt.plot(surrogate, 'g-')
        plt.title('surrogate loss')
        plt.grid()
