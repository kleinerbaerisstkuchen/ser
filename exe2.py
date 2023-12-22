from scipy.io import loadmat
import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, tril, vstack, diags, hstack
from scipy.sparse.linalg import inv
from absl import app, flags
import scipy.sparse.linalg as splinalg

flags.DEFINE_integer("delta", 1, "time interval between sampling")

def main(argv):
    delta = flags.FLAGS.delta
    T = 0.1
    mat = loadmat("dataset1.mat")
    num_states = len(mat["r"])
    num_samples = num_states//delta
    t = np.array(mat["t"][delta::delta])
    # H <- [A_inv, C]T

    A_inv = inv(tril(np.ones((num_samples, num_samples))).tocsr())
    # A_inv = inv(np.tril(np.ones((num_samples, num_samples))))
    C = csr_matrix(np.eye(num_samples))
    H = vstack([A_inv, C])
    H_T = H.T

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
    R_inv = diags([1/r_var] * num_samples, format='csr')
    # R_inv = inv(np.diag([r_var]*num_samples))
    v_var = mat["v_var"][0,0]
    Q_inv = diags([1/v_var] * num_samples, format='csr')
    # Q_inv = inv(np.diag([v_var]*num_samples))
    zero_matrix = csr_matrix((num_samples, num_samples))
    W_inv = vstack([hstack([Q_inv, zero_matrix]), hstack([zero_matrix, R_inv])])

    # sigma
    P_hat_inv = H_T.dot(W_inv.dot(H))
    
    P_hat = inv(P_hat_inv)
    sigma = np.sqrt(P_hat.diagonal())

    # error
    x_estimate = P_hat* H.T * W_inv* z
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
    plt.savefig(f"4444delta {delta}.jpg")

if __name__ =="__main__":
    app.run(main)
