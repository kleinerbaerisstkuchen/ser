from scipy.io import loadmat
import numpy as np
from numpy.linalg import multi_dot,inv
import sympy as sp
import yaml
from absl import app, flags
import matplotlib.pyplot as plt
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "exe3/EKF_config.yaml", "Path to the config file")
r_max = 5
data = loadmat('/home/hnaxiong/ser/exe3/dataset2.mat')
t = data['t']
x_true = data['x_true'] 
y_true = data['y_true']
th_true = data['th_true']
true_valid = data['true_valid']

v = data['v']
om = data['om']

v_var = data['v_var'][0,0]
om_var = data['om_var'][0,0]
W = np.array([[v_var],[om_var]])
Q = np.diag((v_var, om_var))

l = data['l']
r = data['r']
r[r > r_max] = 0
b = data['b']
r_var = data['r_var'][0,0]
b_var = data['b_var'][0,0]
n = np.array([[r_var],[b_var]])
R = np.diag((r_var , b_var))

d = data['d'][0,0]

def compute_Jacobian_F():
    x, y, theta, v, omega, T, w_v, w_om = sp.symbols('x y theta v omega T w_v w_om')
    x_pred = x + T * (v + w_v) * sp.cos(theta) 
    y_pred = y + T * (v + w_v) * sp.sin(theta) 
    theta_pred = theta + T * (omega + w_om)
    state = [x, y, theta]
    state_pred = [x_pred, y_pred, theta_pred]
    process_noise = [w_v, w_om]
    Jacobian_F = sp.Matrix(state_pred).jacobian(state)
    Func_F = sp.lambdify((T, v, theta, w_v), Jacobian_F, modules = 'numpy')
    Jacobian_w = sp.Matrix(state_pred).jacobian(process_noise)
    Func_w = sp.lambdify((T, theta), Jacobian_w, modules = 'numpy')
    motion_model_Matrix = sp.Matrix(state_pred)
    Func_pred = sp.lambdify((x, y, theta, v, omega, T, w_v, w_om), motion_model_Matrix, modules='numpy')
    return Func_F, Func_w, Func_pred


def motion_model(state, input,T,Func_F, Func_w, Func_pred_state):
    x = state[0,0]
    y = state[1,0]
    theta = state[2,0]
    v = input[0,0]
    om = input[1,0]
    state_pred = Func_pred_state(x, y, theta, v, om, T, 0, 0)
    Jacobian_F = Func_F(T,v,theta,0)
    Jacobian_w = Func_w(T, theta)
    Q_prime = Jacobian_w.dot(Q).dot(Jacobian_w.T)

    return state_pred, Jacobian_F, Q_prime

def compute_Jacobian_G():
    x, y, theta, x_l, y_l, d = sp.symbols('x y theta x_l y_l d')
    state = [x, y, theta]
    r = sp.sqrt((x_l - x - d * sp.cos(theta))**2 + (y_l - y - d * sp.sin(theta))**2)
    phi = sp.atan2((y_l - y - d * sp.sin(theta)),(x_l - x - d * sp.cos(theta))) - theta
    pred_measurement = [r, phi]
    observation_model_Matrix = sp.Matrix(pred_measurement)
    Func_pred = sp.lambdify((x, y, theta, x_l, y_l, d), observation_model_Matrix, modules = 'numpy')
    Jacobian_G = sp.Matrix(pred_measurement).jacobian(state)
    Func_G = sp.lambdify((x, y, theta, x_l, y_l, d), Jacobian_G, modules = 'numpy')
    return Func_G, Func_pred

def observation_model(state, landmarks, valid_lm_indices,Func_G, Func_pred_measurement):
    x = state[0,0]
    y = state[1,0]
    theta = state[2,0]
    landmarks = landmarks[valid_lm_indices]
    pred_measurements = None
    Jacobian_Gs = None
    R_prime = np.diag(np.tile([r_var, b_var],len(valid_lm_indices[0])))
    for i, landmark in enumerate(landmarks):

        x_l, y_l = landmark
        pred_measurement = Func_pred_measurement(x, y, theta, x_l, y_l, d)
        Jacobian_G = Func_G(x, y, theta, x_l, y_l, d)

        if i == 0:
            pred_measurements = pred_measurement
            Jacobian_Gs = Jacobian_G
        else:
            pred_measurements = np.vstack((pred_measurements, pred_measurement))
            Jacobian_Gs = np.vstack((Jacobian_Gs, Jacobian_G))
    return pred_measurements, Jacobian_Gs, R_prime


def main(argv):
    #get config
    with open(FLAGS.config, 'r') as stream:
        config = yaml.safe_load(stream)
        r_max = config.get('r_max')
        poor_init = config.get('poor_init')
        CRLB = config.get('CRLB')

    Func_F, Func_w, Func_pred_x = compute_Jacobian_F()
    Func_G, Func_pred_y = compute_Jacobian_G()
    data = loadmat('/home/hnaxiong/ser/exe3/dataset2.mat')
    t = data['t']
    x_true = data['x_true'] 
    y_true = data['y_true']
    th_true = data['th_true']
    true_valid = data['true_valid']

    v = data['v']
    om = data['om']

    v_var = data['v_var'][0,0]
    om_var = data['om_var'][0,0]
    W = np.array([[v_var],[om_var]])
    Q = np.diag((v_var, om_var))

    l = data['l']
    r = data['r']
    r[r > r_max] = 0
    b = data['b']
    r_var = data['r_var'][0,0]
    b_var = data['b_var'][0,0]
    n = np.array([[r_var],[b_var]])
    R = np.diag((r_var , b_var))

    d = data['d'][0,0]


    #initial state and covariance
    if(poor_init):
        x = np.array([1],[1],[0.1])
    else: 
        x = np.array([[x_true[0,0]], [y_true[0,0]],[th_true[0,0]]])
    P = np.diag((1, 1, 0.1))
    state_list = [x]
    cov_list = [P]
    t_last = 0
    t_list = [t_last]
    variance_xyth_list = [(1,1,0.1)]

    # for k in tqdm(range(100)):
    for k in tqdm(range(1,len(t))):
        if true_valid[k] == 1:
            # prediction
            T = t[k][0]-t_last
            input = np.array([[v[k,0]], [om[k,0]]])
            x_check, Jacobian_F, Q_prime = motion_model(x, input, T,Func_F, Func_w, Func_pred_x)
            P_check = multi_dot([Jacobian_F, P, Jacobian_F.T]) + Q_prime

            #correction
            valid_lm_indices = np.where(r[k]!= 0)
            #if we got no measurment from landmark
            if valid_lm_indices[0].size == 0:
                x = x_check
                P = P_check
                variance_xyth = np.diagonal(P_check)
            else:
                y_preds, Jacobian_Gs, R_prime = observation_model(x_check, l, valid_lm_indices, Func_G, Func_pred_y)
                r_true_valid = r[k][valid_lm_indices]
                b_true_valid = b[k][valid_lm_indices]
                y_true_valid = np.empty(r_true_valid.size *2, dtype=b_true_valid.dtype)
                y_true_valid[0::2] = r_true_valid
                y_true_valid[1::2] = b_true_valid
                y_true_valid = y_true_valid.reshape(-1, 1)


                #Kalman gain
                # test = inv(Jacobian_Gs.dot(P_check).dot(Jacobian_Gs.T))
                K = multi_dot([P_check, Jacobian_Gs.T,inv(multi_dot([Jacobian_Gs, P_check, Jacobian_Gs.T]) + R_prime)])

                P_hat = (1 - K.dot(Jacobian_Gs)).dot(P_check)
                innovation = y_true_valid - y_preds
                x_hat = x_check + K.dot(innovation)
                variance_xyth = np.diagonal(P_hat)

                # update
                x = x_hat
                P = P_hat

            t_last = t[k][0]
            
            state_list.append(x)
            cov_list.append(P)
            variance_xyth_list.append(variance_xyth)
            t_list.append(t_last)
    
    # compute error and extract variance
    x_true = x_true[true_valid == 1]
    y_true = y_true[true_valid == 1]
    th_true = th_true[true_valid == 1]

    x_error = x_true - [state[0,0] for state in state_list]
    y_error = y_true - [state[1,0] for state in state_list]
    th_error = th_true - [state[2,0] for state in state_list]

    x_variance = [var[0] for var in variance_xyth_list]
    y_variance = [var[1] for var in variance_xyth_list]
    th_variance = [var[2] for var in variance_xyth_list] 

    x_3sigma = [np.sqrt(x)*3 for x in x_variance]
    y_3sigma = [np.sqrt(x)*3 for x in y_variance]
    th_3sigma = [np.sqrt(x)*3 for x in th_variance]


    # plot
    plt.figure(figsize=(10, 8)) 
    plt.subplot(3, 1, 1)
    plt.plot(t_list, x_error,"-")
    plt.plot(t_list, x_3sigma,"--")
    plt.plot(t_list, -x_3sigma,"--")
    plt.xlabel("t [s]")
    plt.ylabel("error [m]")

    plt.subplot(3, 1, 2)
    plt.plot(t_list, y_error,"-")
    plt.plot(t_list, y_3sigma,"--")
    plt.plot(t_list, -y_3sigma,"--")
    plt.xlabel("t [s]")
    plt.ylabel("error [m]")

    plt.subplot(3, 1, 3)
    plt.plot(t_list, th_error,"-")
    plt.plot(t_list, th_3sigma,"--")
    plt.plot(t_list, -th_3sigma,"--")
    plt.xlabel("t [s]")
    plt.ylabel("error [m]")

    plt.savefig(f"/home/hnaxiong/ser/exe3/r_max {r_max}.jpg")

    # annimation
    from matplotlib.animation import FuncAnimation


if __name__ == '__main__': 
    app.run(main)
