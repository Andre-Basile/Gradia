import matplotlib.pyplot as plt

def draw(x, y = None, title='Graphic', xlabel='X', ylabel='Y'):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    if y is None:plt.plot(x)
    else: plt.plot(x, y)
    plt.show()
