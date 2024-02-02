import matplotlib.pyplot as plt

def plot_region_to_fig_units(plot_rect_cm, fig_size_cm):

    plot_rect = [plot_rect_cm[0]/fig_size_cm[0],
                 plot_rect_cm[1]/fig_size_cm[1],
                 plot_rect_cm[2]/fig_size_cm[0],
                 plot_rect_cm[3]/fig_size_cm[1],]
    return plot_rect

def get_figure_win(fig_size_cm, plot_rect_cm, n_subfigs=[1, 1], w_h_sep_cm=[0, 0]):

    # Convert fig size to inches and the rest to relative units within the figure window
    fig_size = [l/2.54 for l in fig_size_cm]
    # The plotting rectangle containing the grid with all axis
    plot_rect = plot_region_to_fig_units(plot_rect_cm, fig_size_cm)
    # Horizontal and vertical separation between axes
    w_h_sep = [w_h_sep_cm[i]/fig_size_cm[i] for i in range(2)]
    # The size of a single axes
    axes_size = [ (plot_rect[2]-(n_subfigs[1]-1)*w_h_sep[0]) / n_subfigs[1],
                  (plot_rect[3]-(n_subfigs[0]-1)*w_h_sep[1]) / n_subfigs[0]]

    # Create the figure window together with all axis on a grid
    fig = plt.figure(figsize=fig_size)
    axs = []
    for row in range(n_subfigs[0]-1, -1, -1):
        for col in range(n_subfigs[1]):
            left_tmp = plot_rect[0] + col*(axes_size[0]+w_h_sep[0])
            bottom_tmp = plot_rect[1] + row*(axes_size[1]+w_h_sep[1])
            ax_rect_tmp = [left_tmp, bottom_tmp, axes_size[0], axes_size[1]]
            axs.append(plt.axes(ax_rect_tmp, facecolor='none'))

    return fig, axs
