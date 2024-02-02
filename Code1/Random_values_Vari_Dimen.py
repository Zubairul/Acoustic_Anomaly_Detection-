import numpy as np
from Code1.plotting_functions import get_figure_win           # Importing ploting function
import matplotlib.pyplot as plt
import os

figure_panel_dir = 'Figure panels'
if not os.path.isdir(figure_panel_dir):
    os.makedirs(figure_panel_dir)


def X_Values(Dimension):
    """
    Create Random samples in targeted location with variable dimension for further use.
    
        Arguments:
            Dimension : The dimension of the samples.

    
    """
    n=500               #number of data per class
    normal=np.zeros((n,Dimension))
    normal[:,0:2]=3

    New_normal=np.zeros((n,Dimension))
    New_normal[:,0:2]=3

    #Data creation
    x1=np.random.randn(n,Dimension)-normal
    x2=np.random.randn(n,Dimension)+normal
    X_normal= np.vstack([x1,x2])              #Training data set

    x_new_normal_1=np.random.randn(n,Dimension)-New_normal
    x_new_normal_2=np.random.randn(n,Dimension)+New_normal
    X_new_normal=np.vstack([x_new_normal_1,x_new_normal_2])     #new_normal data set

    b1=np.random.randn(n,Dimension)
    b2=np.random.randn(n,Dimension)
    X_noise=np.vstack([b1,b2])#noise data set
    # figure size declearation

    fig_size_cm = [10, 7]                                  # [width, height]
    plot_rect_cm = [2.75, 2,7,3]                           # [left, bottom, width, height]
    n_subfigs = [1, 1]                                      # [n_rows, n_cols]
    hor_ver_sep_cm = [2, 0]   
    
    if Dimension<3:
        X_new_normal=X_new_normal+[6,6]
        X_normal=X_normal+[6,6]
    

        fig, ax = plt.subplots(1, 1, figsize=[6,4]);  # Ploting figure
        ax.set(title="Data points",xlabel='X1', ylabel='X2')

        plt.scatter(X_normal[:, 0], X_normal[:, 1], c="red", s=40, edgecolors="k")
        ax.xaxis.label.set(fontsize=12)
        ax.yaxis.label.set(fontsize=12)
        #ax[0].plot(X_new_normal[:,0],X_new_normal[:,1],'.',alpha=1,mec='none',c="k")
        #ax[0].plot(X_noise[:,0],X_noise[:,1],'o',alpha=1,mec='none',c="blue")
        fig.savefig(figure_panel_dir + os.sep + 'Sample_Data.png', dpi=300)  # Raster image
        
    if Dimension>2:
        fig, a = get_figure_win(fig_size_cm, [0,0,0,0], n_subfigs, hor_ver_sep_cm)  # Ploting figure 
        ax = plt.axes(projection='3d')
        ax.plot(X_noise[:,0],X_noise[:,1],X_noise[:,2],'.',alpha=1,mec='none',c="blueviolet")
        ax.plot(X_normal[:,0],X_normal[:,1],X_normal[:,2],'.',alpha=1,mec='none',c="red")
        ax.plot(X_new_normal[:,0],X_new_normal[:,1],X_new_normal[:,2],'o',alpha=.2,mec='none',c="blue")
        ax.set(title="Data points",xlabel='X1', ylabel='X2',zlabel="X3")
        ax.set_position([0, 0.1, 0.9, 0.9])
        fig.savefig(figure_panel_dir + os.sep + 'High dimensional data(150d).png', dpi=300)  # Raster image

    return X_normal, X_new_normal,X_noise



