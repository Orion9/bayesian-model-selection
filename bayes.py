import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


class MultiGauss:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


def normalize_data(data, mean, var):
    std = np.math.sqrt(var)
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * std)

    if data.shape == mean.shape:
        return np.array(np.prod(norm_const * np.exp(-0.5 * ((data - mean) ** 2) / var)))
    else:
        res = np.array([])
        for x in data:
            new_x = np.array(np.prod(norm_const * np.exp(-0.5 * ((x - mean) ** 2) / var)))
            np.append(res, new_x)
        return res


def calculate_phi(data, order, type):
    if type == 'poly':
        if np.isscalar(data):
            return np.array([data ** j for j in range(0, order)]).reshape((1, order))

        phi_as_list = []
        for x in data:
            phi_as_list.append([x ** j for j in range(0, order)])
        return np.array(phi_as_list)

    else:
        if np.isscalar(data):
            return np.concatenate(( [np.math.cos(np.pi * j * data) for j in range(0, int(order / 2))],  [np.math.sin(np.pi * j * data) for j in range(0, int(order / 2))])).reshape((1, order))

        res = np.array([])
        for x in data:
            np.append(np.concatenate( [np.math.cos(np.pi * j * x) for j in range(0, int(order / 2))],  [np.math.sin(np.pi * j * x) for j in range(0, int(order / 2))]))
        return res


def map_fit_plot(ax, x_values, order_list, wmaps, step):
    for j, (order, w) in enumerate(zip(order_list, wmaps)):
        temp = np.reshape(w, (len(w), 1))
        x_max = np.max(x_values)
        x_min = np.min(x_values)

        z = np.round(((x_max - x_min) / step) - 0.5)
        Nx = z.astype(int)
        x_axis = np.min(x_values) + step * np.array(range(Nx))
        outs = np.empty(x_axis.shape)
        for i, x in enumerate(x_axis):
            hh = np.reshape(np.array(calculate_phi(x, order, 'tri' if order == 8 or order == 12 else 'poly')), temp.shape)
            outs[i] = np.dot(np.transpose(temp), hh)

        ax.plot(x_axis, outs, label=str('{0}{1}'.format('tri' if order == 8 or order == 12 else 'poly', order)))
        # ax.legend(title='MAP fit', loc = 2)


def prob_plots(ax, prob_list, order_list):
    x = range(len(prob_list[0]))
    for i, prob in enumerate(prob_list):
        ax.plot(x, prob, label='{0}{1}'.format('tri' if order_list[i] == 8 or order_list[i] == 12 else 'poly', order_list[i]))
    ax.legend(title='$p(H_k)$', loc=2)


if __name__ == '__main__':
    df = pd.read_csv('./data_stocks.csv')
    data = df.values

    company_names = list(df)[1:]
    dates = np.int32(data[:, 0])
    data = data[:, 1:].T

    company_id = 5
    is_tri = False

    Y = data[company_id]
    Y = preprocessing.scale(Y)[:500]
    X = np.arange(Y.shape[0], dtype=np.float64)
    X = preprocessing.scale(X)[:500]

    data = np.vstack((X, Y)).T

    param_noise = data[:, 1].std()
    observ_noise = 0.1

    # order = 3
    x_list = np.array([])
    y_list = np.array([])

    order_list = [1, 3, 5, 8, 12]

    phi_list = [np.ndarray((0, order_list[i])) for i in range(len(order_list))]
    model_probs = [[] for i in range(len(order_list))]

    parameter_list = []
    for i, order in enumerate(order_list):
        prior_mean = np.zeros((order, 1))
        prior_var = np.eye(order) * param_noise

        initial_parameter = MultiGauss(mean=prior_mean, variance=prior_var)
        parameter_list.append(initial_parameter)

    for new_x, new_y in data:
        x_list = np.append(x_list, new_x)
        y_list = np.append(y_list, new_y)
        evidence = np.zeros((len(order_list), 1))

        for i, order in enumerate(order_list):
            # Parameter estimation
            parameter = parameter_list[i]

            if order == 8 or order == 12:
                type = 'tri'
            else:
                type = 'poly'

            phi = calculate_phi(new_x, order, type)
            inv_var = np.linalg.inv(parameter.variance)

            new_inv_var = inv_var + np.dot(phi.T, phi) / observ_noise ** 2
            temp = np.dot(inv_var, parameter.mean) + phi.T * new_y / observ_noise ** 2
            parameter.variance = np.linalg.inv(new_inv_var)

            new_mean = np.dot(parameter.variance, temp)
            parameter.mean = new_mean

            # Model selection
            phi_list[i] = np.vstack((phi_list[i], calculate_phi(new_x, order, type)))
            phi = phi_list[i]
            parameter = parameter_list[i]

            sigma = parameter.variance
            num_data_points = phi.shape[0]
            num_parameters = phi.shape[1]

            A = (np.dot(phi.T, phi) / observ_noise ** 2 + np.eye(num_parameters, num_parameters) / param_noise)

            model_mean = np.dot(phi, parameter.mean)

            temp = y_list.reshape(num_data_points, -1)
            num1 = normalize_data(temp, model_mean, observ_noise ** 2)
            num2 = normalize_data(parameter.mean, np.zeros((order, 1)), param_noise)
            denom = np.math.sqrt(np.linalg.det(A / (2 * np.pi)))
            evidence[i] = (num1 * num2) / denom

        for i, prob in enumerate(model_probs):
            sum_evidence = np.sum(evidence)
            prob.append(evidence[i] / sum_evidence)

    wmap = [param.mean for param in parameter_list]

    # Plots
    fig, (ax1, ax2) = plt.subplots(2)
    map_fit_plot(ax1, x_list, order_list, wmap, 0.005)
    ax1.set_ylim([np.min(y_list), np.max(y_list)])
    ax1.plot(x_list, y_list, 'k+', ms=4, alpha=0.5)
    ax1.set_title("Data and MAP fits")
    prob_plots(ax2, model_probs, order_list)
    ax2.set_title("Incremental model probability")
    fig.subplots_adjust(hspace=0.5)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.show()
