from scipy.io import loadmat
import numpy as np
from numpy.linalg import multi_dot, inv
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from absl import app, flags


flags.DEFINE_integer("delta", 10, "time interval between sampling")

def main(argv):
    delta = flags.FLAGS.delta
    T = 0.1
    mat = loadmat("dataset1.mat")
    num_states = len(mat["r"])
    num_samples = num_states//delta
    t = np.array(mat["t"][delta::delta])
    # H <- [A_inv, C]T
    A_inv = inv(np.tril(np.ones((num_samples, num_samples))))
    C = np.eye(num_samples)
    H = csc_matrix(np.concatenate((A_inv,C),axis = 0))
    H_t = csc_matrix(np.transpose(H))

    # z <- [v, y]T
    r = np.array(mat["r"][delta::delta])
    l = np.array(mat["l"])
    y = l-r
    u = mat["v"]
    v = np.zeros((num_samples,1))
    for i in range(delta, num_states, delta):
        v[int(i/delta-1)] = T*np.sum(u[i-delta:i])
    z = np.vstack((v,y))

    # W <-[[Q,0],[0, R]]
    r_var = mat["r_var"][0,0]
    R_inv = inv(np.diag([r_var]*num_samples))
    v_var = mat["v_var"][0,0]
    Q_inv = inv(np.diag([v_var]*num_samples))
    W_inv = csc_matrix(np.block([[Q_inv, np.zeros((num_samples, num_samples))],
                          [np.zeros((num_samples, num_samples)), R_inv]]))
    
    # sigma
    P_hat = inv(multi_dot([H_t,W_inv,H]))
    sigma = np.sqrt(np.diag(P_hat))

    # error
    x_estimate = multi_dot([P_hat,H_t,W_inv,z])
    x_true = np.array(mat["x_true"][delta::delta])
    error = x_true-x_estimate

    # plot 
    plt.figure(figsize=(10, 8)) 
    plt.subplot(2, 1, 1)
    plt.plot(t,error,"-")
    plt.plot(t,sigma*3,"--")
    plt.plot(t,-sigma*3,"--")
    plt.xlabel("t [s]")
    plt.ylabel("error [m]")
    plt.title(f"delta = {delta}")

    plt.subplot(2, 1, 2)
    plt.hist(error, bins = "auto", density = True)
    plt.xlabel("error[m]")
    plt.ylabel("frequency")
    plt.show 
    plt.savefig(f"delta {delta}.jpg")

if __name__ =="__main__":
    app.run(main)

