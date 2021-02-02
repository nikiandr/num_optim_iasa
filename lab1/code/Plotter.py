import numpy as np
import matplotlib.pyplot as plt

def make_domains(x_lim, y_lim, target_func):
    domain_x = np.linspace(*x_lim)
    domain_y = np.linspace(*y_lim)
    func = np.zeros((domain_x.size, domain_y.size))
    counter_y = 0
    for x in domain_x:
        counter_x = 0
        for y in domain_y:
            point = np.array([[x], [y]])
            func[counter_x, counter_y] = target_func(point)
            counter_x += 1
        counter_y += 1
    domain_x, domain_y = np.meshgrid(domain_x, domain_y)
    return domain_x, domain_y, func

def PlotContour(x_lim, y_lim, target_func, points = None, fname = None):
    plt.close()
    domain_x, domain_y, func = make_domains(x_lim, y_lim, target_func)
    plt.contourf(domain_x, domain_y, func, levels = 10, cmap = 'viridis')
    if points is not None:
        plt.plot(points[:, 0], points[:, 1], color = 'tab:red', linewidth = 1)
        plt.scatter(points[-1, 0], points[-1, 1], color = 'tab:red', marker = 'x', s = 20)
        plt.scatter(points[0, 0], points[0, 1], color = 'tab:red', marker = 'o', s = 20)
        plt.title(f'x* = argmin F(x) = ({points[-1, 0]:.5f}, {points[-1, 1]:.5f})\nF(x*) = {points[-1, 2]}\nfound in {points.shape[0]} iterations')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    
def PlotSurface(x_lim, y_lim, target_func, points = None, fname = None):
    plt.close()
    domain_x, domain_y, func = make_domains(x_lim, y_lim, target_func)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.plot_surface(domain_x, domain_y, func, cmap = 'viridis', alpha = 0.5)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color = 'tab:red', linewidth = 2, marker = 'o', markersize = 3)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color = 'tab:green', marker = 'd', s = 50)
        plt.title(f'x* = argmin F(x) = ({points[-1, 0]:.5f}, {points[-1, 1]:.5f})\nF(x*) = {points[-1, 2]}\nfound in {points.shape[0]} iterations')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches = 'tight', dpi = 300)