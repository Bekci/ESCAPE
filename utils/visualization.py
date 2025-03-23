import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_points(list_of_points, sizes=None):
    colorscales = ['peach', 'Viridis', 'Plasma', 'brwnyl', 'dense', 'gray', 'haline', 'ice', 'matter','solar']
    graphs = []
    for i,points in enumerate(list_of_points):
        cur_size = sizes[i] if sizes is not None else 4
        
        graphs.append(go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(
                size=cur_size,
                colorscale=colorscales[-i],
                opacity=0.8
            )
        ))
    
    fig = go.Figure(data=graphs)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def save_learning_rate_chart(epoch_learning_rates, output_main_folder, model_type, dataset_type):
    chart_save_dir = os.path.join(output_main_folder, model_type, dataset_type)
    os.makedirs(chart_save_dir, exist_ok=True)

    plt.figure()
    plt.title("Learning Rate - Epoch")
    plt.plot(epoch_learning_rates)
    plt.savefig(os.path.join(chart_save_dir, "learning_rate.png"))
    plt.close()

    with open(os.path.join(chart_save_dir, 'learning_rates.npy'),'wb') as npfile:
        np.save(npfile, epoch_learning_rates)

def save_loss_graphs(training_losses, output_main_folder, model_type):

    chart_save_dir = os.path.join(output_main_folder, model_type)
    os.makedirs(chart_save_dir, exist_ok=True)

    plt.figure()
    plt.title("Full training loss graph")
    plt.plot(training_losses[:, -1])
    plt.savefig(os.path.join(chart_save_dir, "full_loss.png"))
    plt.close()

    plt.figure()
    plt.title("Last epochs training loss graph")
    plt.plot(training_losses[-30:, -1])
    plt.savefig(os.path.join(chart_save_dir, "last_losses.png"))
    plt.close()    

    with open(os.path.join(chart_save_dir, 'training_losses.npy'),'wb') as npfile:
        np.save(npfile, training_losses)
