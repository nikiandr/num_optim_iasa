import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

def Ellipsoid(a, b, c, x, y, z):
    return a*x**2 + b*y**2 + c*z**2

def Hyperplane(p, beta, x, y, z):
    return p[0]*x + p[1]*y + p[2]*z - beta

def make_plane(x_lim, y_lim, target_func, p, beta, points=None, fname=None):
    plt.close()
    domain_x = np.linspace(*x_lim)
    domain_y = np.linspace(*y_lim)
    surf = np.zeros((domain_x.size, domain_y.size))
    counter_y = 0
    for x in domain_x:
        counter_x = 0
        for y in domain_y:
            surf[counter_x, counter_y] = (beta - p[0]*x - p[1]*y)/p[2]
            counter_x += 1
        counter_y += 1
    domain_x, domain_y = np.meshgrid(domain_x, domain_y)
    potential = target_func(domain_x, domain_y, surf)
    norm = matplotlib.colors.SymLogNorm(1, vmin=potential.min(), vmax=potential.max(), base=np.e)
    colors = cm.rainbow(norm(potential))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(domain_x, domain_y, surf, facecolors=colors, alpha=0.4)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='tab:red', linewidth=2, marker='o', markersize=3)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='tab:green', marker='d', s=50)
        plt.title(f'x* = argmin F(x) = ({points[-1, 0]:.5f}, {points[-1, 1]:.5f}, {points[-1, 2]:.5f})\nF(x*) = {points[-1, 3]}\nfound in {points.shape[0]-1} iterations')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=300)

def make_ellipsoid(target_func, a, b, c, points=None, fname=None):
    plt.close()
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1/np.sqrt(a) * np.outer(np.cos(u), np.sin(v))
    y = 1/np.sqrt(b) * np.outer(np.sin(u), np.sin(v))
    z = 1/np.sqrt(c) * np.outer(np.ones(np.size(u)), np.cos(v))
    potential = target_func(x, y, z)
    norm = matplotlib.colors.SymLogNorm(1, vmin=potential.min(), vmax=potential.max(), base=np.e)
    colors = cm.rainbow(norm(potential))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=18, azim=-68)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(x, y, z, facecolors=colors, alpha=0.1)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='tab:red', linewidth=2, marker='o', markersize=3)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='tab:green', marker='d', s=50)
        plt.title(f'x* = argmin F(x) = ({points[-1, 0]:.5f}, {points[-1, 1]:.5f}, {points[-1, 2]:.5f})\nF(x*) = {points[-1, 3]}\nfound in {points.shape[0]-1} iterations')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=300)