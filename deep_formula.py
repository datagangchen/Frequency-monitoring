
from robustness import *
from struct_formula import *
from load_data import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF
from numpy.linalg import inv

def generate_train(data_name, Num):
    signal, time2, name, label = load_data(data_name)
    label = np.squeeze(label)
    X =[]
    y = []
    for _ in range(Num):
        vector, formula = Init_state(name, signal, time2, 3)
        tree = formula.get_tree()
        rewards = reward(tree, name, signal,time2 )
        rob = [a*b for a, b in zip(rewards, label)]
        X.append(vector)
        y.append(min(rob))
    header = "X_train, y_train \n"
    data = np.column_stack((X, y))
    dat_name = 'nn_'+data_name[0:-3]+'dat'
    np.savetxt(dat_name, data, header=header)
    return X , y

def generate_vector(data_name, Num):
    signal, time2, name, label = load_data(data_name)
    X =[]
    for _ in range(Num):
        vector, formula = Init_state(name,signal, time2,3)
        X.append(vector)
    header = "X_sample\n"
    dat_name = 'Xs_'+data_name[0:-3]+'dat'
    np.savetxt(dat_name, X, header=header)
    return X


def robust_function(vector,data_name, pool):
    signal, time1, name, label = load_data(data_name)
    state, formulas = Init_state(name, signal, time1,3)
    tree = formulas.vector_tree(vector)
    if pool:
        rewards = poolreward(tree,name,signal,time1)
    else:
        rewards= reward(tree,name,signal,time1)

    robust = [a*b for a, b in zip(rewards, label)]
    return min(robust)

def find_a_candidate(X_train, Xs,ys, kappa):
    var = calculate_variance(X_train,Xs)
    ucb = [m+kappa*v for m, v in zip(ys,var)]
    max_idx = np.argmax(ucb)
    return Xs[max_idx].tolist()

def calculate_variance(X,Xs):
    kernel = RBF()
    kernel.__init__()
    sigma = 0.01
    ks = kernel.diag(Xs)
    Kt = kernel.__call__(X,X)+sigma**2
    Kt = inv(Kt)
    K = kernel.__call__(Xs,X)
    Ktt = np.matmul(K, Kt)
    Ktt = np.matmul(Ktt,K.transpose())
    Ktt = Ktt.diagonal()
    var = ks - Ktt.transpose()
    return var


def deep_regression(X,y,Xs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # define and fit the final model
    model = Sequential()
    model.add(Dense(200, kernel_regularizer=regularizers.l2(0.0001),  input_dim=len(X[0]), activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(600, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.0001), activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X_train, y_train, epochs=100, batch_size = 201,validation_data =(X_test,y_test), verbose=0)
    # make a prediction
    # loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128) 
    prediction = model.predict(Xs, batch_size=128)  
    return prediction





def run_program():
    data_name ='train_inner.mat'
    T = 100
    #X, y = generate_train(data_name,1000)
    dat_name = 'nn_'+data_name[0:-3]+'dat'
    X, y  = load_py_data(dat_name)
    y  = np.squeeze(y)
    #Xs   = generate_vector(data_name,5000)
    dat_name = 'Xs_'+data_name[0:-3]+'dat'
    Xs  = load_tree_vector(dat_name)
    #X = X.tolist()
    res =[max(y)]

    for idx  in range(T):
        ys = deep_regression(X,y,Xs)
        x_new = find_a_candidate(X,Xs,ys,2)
        X = X.tolist()
        X.append(x_new)
        X = np.asarray(X)
        y_new = robust_function(x_new,data_name, False)
        y = np.concatenate((y,y_new))
        res.append(max(y))

        print('Iteration', idx)

    plt.figure()
    plt.plot(res)
    plt.show()




 



if __name__ == '__main__':
	run_program()





