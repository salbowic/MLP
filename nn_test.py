import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from neural_network import *

np.random.seed(1)

def create_gif(image_folder, output_gif_path):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_gif_path, images, duration=0.8)

def test_iters(nn, x, y, min_iters, max_iters, step):
    hl_size = nn.HIDDEN_L_SIZE
    nn_lr = str(nn.LR).replace('.', '_')
    if not os.path.exists(f'images_{hl_size}_lr_{nn_lr}'):
        os.makedirs(f'images_{hl_size}_lr_{nn_lr}')

    mse_values = []

    for iters in range(min_iters, max_iters+1, step):
        nn.train(x, y, step)

        yh = nn.predict(x) # wyniki (y) z sieci

        # Obliczanie Mean Squared Error
        mse = mean_squared_error(y, yh)
        mse_values.append({'Learning rate': nn.LR, 'Hidden layer  size': nn.HIDDEN_L_SIZE, 'Iterations': iters, 'MSE': mse})

        # Wykresy
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.plot(x, y, 'r', label='Target Function')
        plt.plot(x, yh, 'b', label=f'Predicted Function\nMSE: {mse:.6f}')  # Include MSE in the label
        plt.legend()
        plt.title(f'iterations: {iters}, LR: {nn.LR}, HL size: {hl_size}')

        plt.savefig(f'images_{hl_size}_lr_{nn_lr}/NN_after_{iters}_iters.png')

    image_folder = f'images_{hl_size}_lr_{nn_lr}'
    gif_name = f'animation_{hl_size}_lr_{nn_lr}.gif'
    create_gif(image_folder, gif_name)

def test_HL_size(x, y, min_hl, max_hl, step, iters):
    mse_values = []
    for hls in range(min_hl, max_hl + step, step):
        nn = DlNet(x, y, 0.003, hls)
        nn.train(x, y, iters)
        yh = nn.predict(x) # wyniki (y) z sieci

        # Obliczanie Mean Squared Error
        mse = mean_squared_error(y, yh)
        mse_values.append({'Learning rate': nn.LR, 'Hidden layer  size': nn.HIDDEN_L_SIZE, 'Iterations': iters, 'MSE': mse})
    
    mse_df = pd.DataFrame(mse_values)
    mse_df.to_excel(f'mse_results/mse_HL.xlsx', index=False)

def test_LR(x, y, min_lr, max_lr, step, iters):
    mse_values = []
    for lr in range(min_lr, max_lr + step, step):
        i_lr = lr * 0.001
        nn = DlNet(x, y, i_lr , hls = 50)
        nn.train(x, y, iters)
        yh = nn.predict(x) # wyniki (y) z sieci

        # Obliczanie Mean Squared Error
        mse = mean_squared_error(y, yh)
        mse_values.append({'Learning rate': nn.LR, 'Hidden layer  size': nn.HIDDEN_L_SIZE, 'Iterations': iters, 'MSE': mse})

        mse_df = pd.DataFrame(mse_values)
        mse_df.to_excel(f'mse_results/mse_LR.xlsx', index=False)
    
    # mse_df = pd.DataFrame(mse_values)
    # mse_df.to_excel(f'mse_results/mse_LR.xlsx', index=False)
    
    
def main():

    #ToDo tu prosze podac pierwsze cyfry numerow indeksow
    p = [3,8]

    L_BOUND = -5
    U_BOUND = 5

    def q(x):
        return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

    x1 = np.linspace(L_BOUND, U_BOUND, 100)
    y = q(x1)

    x = x1.reshape(-1, 1)  # Reshape x to have two dimensions

    nn = DlNet(x, y, 0.003, hls=50)

    # test_iters(nn, x, y, 200000, 200000, 200000)

    # test_HL_size(x, y, 15, 100, 5, 15000)

    test_LR(x, y, 1, 10, 1, 15000)

if __name__ == "__main__":
    main()