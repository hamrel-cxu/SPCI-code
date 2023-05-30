from scipy.sparse import random
import pandas as pd
import numpy as np
import warnings
import torch
import pickle
import utils_EnbPI
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
warnings.filterwarnings("ignore")


class real_data_loader():
    def __init__(self):
        pass

    def get_data(self, data_name, solar_args=None, wind_args=None):
        if data_name == 'solar':
            # Get solar data WITH time t as covariate
            univariate, filter_zero, non_stat_solar = solar_args
            Y_full, X_full_old, X_full_nonstat = self.get_non_stationary_solar(
                univariate=univariate, filter_zero=filter_zero)
            if non_stat_solar:
                X_full = X_full_nonstat
            else:
                X_full = X_full_old
        if data_name == 'electric':
            X_full, Y_full = self.electric_dataset()
        if data_name == 'wind':
            wind_loc = wind_args[0]
            X_full, Y_full = self.get_wind_real(location=wind_loc)
        return X_full, Y_full

    def get_wind_real(self, location=0):
        # Stationary real
        rootpath = 'Data/data_k30'
        wind = np.load(os.path.join(rootpath, 'sample_wind.npy'))
        # len-T vector, denoting wind speed
        speeds = wind[:, location, 1]
        data_y = speeds
        print(f'Shape of full data at location {location}')
        print(data_y.shape)
        data_x = rolling(data_y, window=10)
        N = len(data_x)
        return data_x, data_y[-N:]

    def get_non_stationary_solar(self, univariate=True, max_N=2000, filter_zero=False):
        # Stationary real
        data = utils_EnbPI.read_data(3, 'Data/Solar_Atl_data.csv', 10000)
        data_y = data['DHI'].to_numpy()  # Convert to numpy
        if univariate:
            # Univariate feature
            data_x_old = rolling(data_y, window=20)
        else:
            # Multivariate feature
            data_x_old = data.loc[:, data.columns
                                  != 'DHI'].to_numpy()  # Convert to numpy
        # Add one-hot-encoded DAY features using // (or hour features using %)
        hours = int(data_y.shape[0]/365)
        N = data_x_old.shape[0]
        day_feature = False
        if day_feature:
            # Day one-hot 0,...,364
            one_hot_feature = (np.arange(N) // hours).reshape(-1, 1)
        else:
            # Hourly one-hot 0,...,23
            one_hot_feature = (np.arange(N) % hours).reshape(-1, 1)
        one_hot_feature = OneHotEncoder().fit_transform(one_hot_feature).toarray()
        data_x_new = np.c_[one_hot_feature, data_x_old]
        data_y, data_x_old, data_x_new = data_y[-max_N:
                                                ], data_x_old[-max_N:], data_x_new[-max_N:]
        if filter_zero:
            nonzero_idx = data_y > 0.2
            data_y, data_x_old, data_x_new = data_y[nonzero_idx], data_x_old[nonzero_idx], data_x_new[nonzero_idx]
        return data_y, data_x_old, data_x_new

    def electric_dataset(self):
        # ELEC2 data set
        # downloaded from https://www.kaggle.com/yashsharan/the-elec2-dataset
        data = pd.read_csv(f'Data/electricity-normalized.csv')
        col_names = data.columns
        data = data.to_numpy()

        # remove the first stretch of time where 'transfer' does not vary
        data = data[17760:]

        # set up variables for the task (predicting 'transfer')
        covariate_col = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand']
        response_col = 'transfer'
        # keep data points for 9:00am - 12:00pm
        keep_rows = np.where((data[:, 2] > data[17, 2])
                             & (data[:, 2] < data[24, 2]))[0]

        X = data[keep_rows][:, np.where(
            [t in covariate_col for t in col_names])[0]]
        Y = data[keep_rows][:, np.where(
            col_names == response_col)[0]].flatten()
        X = X.astype('float64')
        Y = Y.astype('float64')

        return X, Y


class simulate_data_loader():
    def __init__(self):
        pass

    def get_simul_data(self, simul_type):
        if simul_type == 1:
            Data_dict = self.simulation_state_space(
                num_pts=2000, alpha=0.9, beta=0.9)
        if simul_type == 2:
            Data_dict = self.simulation_non_stationary()
            Data_dict['X'] = torch.from_numpy(Data_dict['X']).float()
            Data_dict['Y'] = torch.from_numpy(Data_dict['Y']).float()
        if simul_type == 3:
            # NOTE: somehow for this case, currently RF quantile regression does not yield shorter interval. We may tune past window to get different results (like decrease it to 250) if need
            Data_dict = self.simultaion_heteroskedastic()
        return Data_dict

    def simulation_state_space(self, num_pts, alpha, beta):
        '''
            Y_t = alpha*Y_{t-1}+\eps_t
            \eps_t = beta*\eps_{t-1}+v_t
            v_t ~ N(0,1)
            So X_t = Y_{t-1}, f(X_t) = alpha*X_t
            If t = 0:
                X_t = 0, Y_t=\eps_t = v_t
        '''
        v0 = torch.randn(1)
        Y, X, fX, eps = [v0], [torch.zeros(1)], [torch.zeros(1)], [v0]
        scale = torch.sqrt(torch.ones(1)*0.1)
        for _ in range(num_pts-1):
            vt = torch.randn(1)*scale
            X.append(Y[-1])
            fX.append(alpha*Y[-1])
            eps.append(beta*eps[-1]+vt)
            Y.append(fX[-1]+eps[-1])
        Y, X, fX, eps = torch.hstack(Y), torch.vstack(
            X), torch.vstack(fX), torch.hstack(eps)
        return {'Y': Y.float(), 'X': X.float(), 'f(X)': fX, 'Eps': eps}

    def simulation_non_stationary(self):
        with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
            Data_dc_old = pickle.load(fp)
        fXold = Data_dc_old['f(X)']
        gX = non_stationarity(len(fXold))
        fXnew = gX*fXold
        for _ in ['quick_plot']:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(fXold, label='old f(X)')
            ax.plot(fXnew, label='new f(X)')
            ax.legend()
        Data_dc_new = {}
        for key in Data_dc_old.keys():
            if key == 'Y':
                continue
            if key == 'X':
                Data_dc_new[key] = np.c_[
                    np.arange(Data_dc_old[key].shape[0]) % 12, Data_dc_old[key]]
            elif key == 'f(X)':
                Data_dc_new[key] = fXnew
            else:
                Data_dc_new[key] = Data_dc_old[key]
        Data_dc_new['Y'] = Data_dc_new['f(X)']+Data_dc_new['Eps']
        Data_dc_old['Y'] = Data_dc_new['Y']
        Data_dc_old['f(X)'] = Data_dc_new['f(X)']
        # return Data_dc_old, Data_dc_new
        return Data_dc_new

    def simultaion_heteroskedastic(self):
        ''' Note, the difference from earlier case 3 in paper is that
            1) I reduce d from 100 to 20,
            2) I let X to be different, so sigmaX differs
                The sigmaX is a linear model so this effect in X is immediate
            I keep the same AR(1) eps & everything else.'''
        def True_mod_nonlinear_pre(feature):
            '''
            Input:
            Output:
            Description:
                f(feature): R^d -> R
            '''
            # Attempt 3 Nonlinear model:
            # f(X)=sqrt(1+(beta^TX)+(beta^TX)^2+(beta^TX)^3), where 1 is added in case beta^TX is zero
            d = len(feature)
            np.random.seed(0)
            # e.g. 20% of the entries are NON-missing
            beta1 = random(1, d, density=0.2).A
            betaX = np.abs(beta1.dot(feature))
            return (betaX + betaX**2 + betaX**3)**(1/4)
        Tot, d = 1000, 20
        Fmap = True_mod_nonlinear_pre
        # Multiply each random feature by exponential component, which is repeated every Tot/365 elements
        mult = np.exp(0.01*np.mod(np.arange(Tot), 100))
        X = np.random.rand(Tot, d)*mult.reshape(-1, 1)
        fX = np.array([Fmap(x) for x in X]).flatten()
        beta_Sigma = np.ones(d)
        sigmaX = np.maximum(X.dot(beta_Sigma).T, 0)
        with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
            Data_dc = pickle.load(fp)
        eps = Data_dc['Eps']
        Y = fX + sigmaX*eps[:Tot]
        np.random.seed(1103)
        idx = np.random.choice(Tot, Tot, replace=False)
        Y, X, fX, sigmaX, eps = Y[idx], X[idx], fX[idx], sigmaX[idx], eps[idx]
        return {'Y': torch.from_numpy(Y).float(), 'X': torch.from_numpy(X).float(), 'f(X)': fX, 'sigma(X)': sigmaX, 'Eps': eps}


''' Data Helpers '''


def non_stationarity(N):
    '''
        Compute g(t)=t'*sin(2*pi*t'/12), which is multiplicative on top of f(X), where
        t' = t mod 12 (for seaonality)
    '''
    cycle = 12
    trange = np.arange(N)
    tprime = trange % cycle
    term2 = np.sin(2*np.pi*tprime/cycle)
    return tprime*term2


def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

###
