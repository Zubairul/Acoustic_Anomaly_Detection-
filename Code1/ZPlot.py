   # Importing ploting function
import matplotlib.pyplot as plt
import operator
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score    
from os.path import sep
import sys
sys.path.append('../')
from Code1.project_configuration import get_parameter
from Code1.plotting_functions import get_figure_win 
novia_red = get_parameter('color_novia_red')


import os

figure_panel_dir = 'Figure panels'
if not os.path.isdir(figure_panel_dir):
    os.makedirs(figure_panel_dir)
if not os.path.isdir('Tikz'):
    os.makedirs('Tikz')
  

fig_size_cm = [17, 6]                     # [width, height]
plot_rect_cm = [1.6, 1.2, 14, 2.5]  # [left, bottom, width, height]
n_subfigs = [1, 2]                        # [n_rows, n_cols]
hor_ver_sep_cm = [2, 0]  
    

    
# Function for ploting and saving figuer for meshplot
def MeshPlot(Z,Title,X_normal):
    xx, yy = np.meshgrid(np.linspace(-5, 15, 200), np.linspace(-5, 15, 200))
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(1, 1, figsize=[6,4]);
    ax.set(title=Title,xlabel='X1', ylabel='X2')
    ax.xaxis.label.set(fontsize=12)
    ax.yaxis.label.set(fontsize=12)
    


    im_l=ax.contourf(xx, yy, Z, levels=9, cmap=plt.cm.PuBu)
    ax.contour(xx, yy, Z, levels=[1], colors='k')
    plt.scatter(X_normal[:, 0], X_normal[:, 1], c="red", s=40, edgecolors="k")

    plt.colorbar(im_l,)

    plt.show()
    fig.savefig(figure_panel_dir + os.sep + Title+'.png', dpi=300)  # Raster image

    
    
# Function for ploting and saving figuers in a single window
def MeshPlot2(Z_h,Z_l,Title1,Title2,X_normal,lv):
    
    xx, yy = np.meshgrid(np.linspace(-5, 15, 200), np.linspace(-5, 15, 200))
    Z_h = Z_h.reshape(xx.shape)
    Z_l = Z_l.reshape(xx.shape)
    # Get the figure window and the axes
    fig, axs = get_figure_win(fig_size_cm, plot_rect_cm, n_subfigs, hor_ver_sep_cm)   # Ploting in a specific frame

    im_h=axs[0].contourf(xx, yy, Z_h, levels=9, cmap=plt.cm.PuBu)
    axs[0].contour(xx, yy, Z_h, levels=[lv], colors='k')
    axs[0].scatter(X_normal[:, 0], X_normal[:, 1], c="red", s=40, edgecolors="k",alpha=.3)
    axs[0].set(title=Title1,xlabel='X1', ylabel='X2')

    fig.colorbar(im_h, ax=axs[0])

    im_l=axs[1].contourf(xx, yy, Z_l, levels=9, cmap=plt.cm.PuBu)
    axs[1].contour(xx, yy, Z_l, levels=[lv], colors='k')
    axs[1].scatter(X_normal[:, 0], X_normal[:, 1], c="red", s=40, edgecolors="k",alpha=.5)
    axs[1].set(title=Title2,xlabel='X1', ylabel='X2')

    fig.colorbar(im_l, ax=axs[1])

    fig.savefig(figure_panel_dir + os.sep + Title1+'.png', dpi=300)  # Raster image
    
def plot_roc_curve(fpr, tpr,y_test, scores):
    auc = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_AUC_Score = %1.3f' %auc,fontsize=5)
    plt.legend()
    plt.show()
    
def plot_Hyp_Parameter(Name,xlabel, ylabel,Set_x,Set_y,ROC,a,b=None,c=None):
    if b == None:
        n_subfigs = [1, 1] 
        fig, ax = get_figure_win(fig_size_cm, plot_rect_cm, n_subfigs, hor_ver_sep_cm);                        
        ax[0].plot(Set_x,ROC,'-',color='black')
        ax[0].set_xlim(np.amin(Set_x),np.amax(Set_x))
        ax[0].set_ylim(np.amin(Set_y),np.amax(Set_y))
        ax[0].set_ylabel(ylabel)
        ax[0].set_xlabel(xlabel)
        ax[0].set_title(xlabel+"="+" %1.3f" %a+"   ""MAX_ROC"+"="+" %1.3f" %c,fontsize=10)
        plt.grid(color='black', linestyle='-', linewidth=0.3)
        fig.savefig(figure_panel_dir + os.sep + xlabel+Name+'.png', dpi=300)  # Raster image
        plt.show()
        
    else:
        n_subfigs = [1, 1] 
        fig, ax = get_figure_win(fig_size_cm, plot_rect_cm, n_subfigs, hor_ver_sep_cm);       

        im=ax[0].contourf(Set_x,Set_y,ROC)  #PLoted contour plot

        ax[0].set_ylabel(ylabel)
        ax[0].set_xlabel(xlabel)
        ax[0].set_title(xlabel +"="+"%1.3f" %a +"\n"+ ylabel+"="+"%1.1f"%b  , fontsize=10)
        plt.colorbar(im)                          
        fig.savefig(figure_panel_dir + os.sep + xlabel+Name+'.png', dpi=300)  # Raster image
        plt.show()
        

        
def plot_classification_stats(conf_matrix, machine_acc, unique_ids):
    fig, axs = get_figure_win(fig_size_cm, plot_rect_cm, n_subfigs, hor_ver_sep_cm)
    
    
    f = operator.itemgetter(2,3,4,5,6,7,8,9,20,21)
    o = operator.itemgetter(0,1,10,11,12,13,14,15,16,17,18,19)

    im = axs[0].imshow(conf_matrix, cmap=cm.jet_r)
    im.set_clim([0,1])
    axs[0].set_xlabel('Predicted class')
    axs[0].set_ylabel('True class')
    axs[0].set_xticks([0, 1], labels=['Faulty', 'OK'],fontsize=8)  
    axs[0].set_yticks([0, 1], labels=['Faulty', 'OK'],fontsize=8)  
    plt.colorbar(im, ax=axs[0])
    
    axs[1].bar(o(unique_ids), o(machine_acc), 0.5, fc=np.append(novia_red, 0.5), ec=novia_red, lw=2)
    axs[1].bar(f(unique_ids), f(machine_acc), 0.5, fc='b', ec='b', lw=2)
    axs[1].set_xlabel('Machine ID')
    axs[1].set_ylabel('Machinewise \n Accuracy (%)')
    Model_Acc = sum(machine_acc)/machine_acc.size
    axs[1].set_title("Overall Model Accuracy = %1.3f" %Model_Acc,fontsize=5)