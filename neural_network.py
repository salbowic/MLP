import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(1)

# f logistyczna jako przykład sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)
     
#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )
    
class DlNet:
    def __init__(self, x, y, lr, hls = 50):
        self.x = x
        self.y = y
        self.y_out = 0
        
        self.HIDDEN_L_SIZE = hls
        self.LR = lr

        self.W = np.random.rand(self.HIDDEN_L_SIZE, 2)
        self.b1 = np.zeros((1, self.HIDDEN_L_SIZE))
       
    def forward(self, x):
        # Warstwa ukryta (y0 -> y1)
        self.z1 = np.dot(x, self.W[:, 0].reshape(1, self.HIDDEN_L_SIZE)) + self.b1
        self.a1 = sigmoid(self.z1)
        # Warstwa wyjściowa (y1 -> y2)
        self.y_out = np.dot(self.a1, self.W[:,1])
          
    def predict(self, x):    
        self.forward(x)
        return self.y_out
        
    def backward(self, x, y):
        # Warstwa wyjściowa (y1 <- y2)
        d_loss = d_nloss(self.y_out, y) # gradient f straty względem wyjścia sieci
        # gradient f straty względem wyjścia (przed funkcją aktywacji),
        # w warstwie wyjściowej neurony są liniowe, więc d_z2 = d_loss
        d_z2 = d_loss
        d_W2 = np.dot(self.a1.T, d_z2) # gradient f straty względem wag

        # Warstwa ukryta (y0 <- y1)
        d_a1 = np.dot(d_z2, self.W[:,1].reshape(1, self.HIDDEN_L_SIZE)) # gradient f straty względem f aktywacji
        d_z1 = d_a1 * d_sigmoid(self.z1) # gradient f straty względem wartości przed f aktywacji
        d_W1 = np.dot(x.T, d_z1) # gradient f straty względem wag
        d_b1 = np.sum(d_z1, axis=0) # gradient f straty względem obciążeń

        # Aktualizacja wag i obciążeń
        self.W[:,0] -= self.LR * d_W1
        self.b1 -= self.LR * d_b1
        self.W[:,1] -= self.LR * d_W2   
        
    def train(self, x_set, y_set, iters):    
        for i in range(0, iters):
            for x, y in zip(x_set, y_set):
                self.forward(x)
                self.backward(x, y)
             