# from statsmodels.regression.quantile_regression import QuantReg
# from sklearn.linear_model import QuantileRegressor
import numpy as np
import math
import pandas as pd


#### From utils_EnbPI ####
def adjust_alpha_t(alpha_t, alpha, errs, gamma=0.005, method='simple'):
    if method == 'simple':
        # Eq. (2) of Adaptive CI
        return alpha_t+gamma*(alpha-errs[-1])
    else:
        # Eq. (3) of Adaptive CI with particular w_s as given
        t = len(errs)
        errs = np.array(errs)
        w_s_ls = np.array([0.95**(t-i) for i in range(t)]
                          )  # Furtherest to Most recent
        return alpha_t+gamma*(alpha-w_s_ls.dot(errs))


def ave_cov_width(df, Y):
    coverage_res = ((np.array(df['lower']) <= Y) & (
        np.array(df['upper']) >= Y)).mean()
    print(f'Average Coverage is {coverage_res}')
    width_res = (df['upper'] - df['lower']).mean()
    print(f'Average Width is {width_res}')
    return [coverage_res, width_res]

#### Miscellaneous ####


window_size = 300


def rolling_avg(x, window=window_size):
    return np.convolve(x, np.ones(window)/window)[(window-1):-window]


def dict_to_latex(dict, train_ls):
    DF = pd.DataFrame.from_dict(np.vstack(dict.values()))
    keys = list(dict.keys())
    index = np.array([[f'{key} coverage', f'{key} width']
                     for key in keys]).flatten()
    DF.index = index
    DF.columns = train_ls
    print(DF)
    print(DF.round(2).to_latex())


def make_NP_df(X, Y):
    Xnames = [f'X{a}' for a in np.arange(X.shape[1]).astype(str)]
    full_names = ['ds']+Xnames+['y']
    date_tmp = pd.date_range(
        start='1/1/2018', periods=len(Y)).astype(str)  # Artificial
    df_tmp = pd.DataFrame(np.c_[date_tmp, X, Y], columns=full_names)
    return df_tmp, Xnames


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def binning(past_resid, alpha):
    '''
    Input:
        past residuals: evident
        alpha: signifance level
    Output:
        beta_hat_bin as argmin of the difference
    Description:
        Compute the beta^hat_bin from past_resid, by breaking [0,alpha] into bins (like 20). It is enough for small alpha
        number of bins are determined rather automatic, relative the size of whole domain
    '''
    bins = 5  # For computation, can just reduce it to like 10 or 5 in real data
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    width = np.zeros(bins)
    for i in range(bins):
        width[i] = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_ls[i]))
    i_star = np.argmin(width)
    return beta_ls[i_star]


def binning_use_RF_quantile_regr(quantile_regr, Xtrain, Ytrain, feature, beta_ls, sample_weight=None):
    # API ref: https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html
    feature = feature.reshape(1, -1)
    low_high_pred = quantile_regr.fit(Xtrain, Ytrain,sample_weight).predict(feature)
    num_mid = int(len(low_high_pred)/2)
    low_pred, high_pred = low_high_pred[:num_mid], low_high_pred[num_mid:]
    width = (high_pred-low_pred).flatten()
    i_star = np.argmin(width)
    wid_left, wid_right = low_pred[i_star], high_pred[i_star]
    return i_star, beta_ls[i_star], wid_left, wid_right


def merge_table_mean_std(table_result, colnames=None):
    M, N = table_result.shape[0], int(table_result.shape[1]/2)
    table = np.zeros((M, N), dtype=object)
    idx = table_result.index
    for i in range(M):
        for j in range(N):
            table[i,
                  j] = f'{table_result.iloc[i,2*j]} ({table_result.iloc[i,2*j+1]})'
    colnames = np.array([[f'{name} coverage', f'{name} width']
                        for name in colnames]).flatten()
    return pd.DataFrame(table, index=idx, columns=colnames)


# def binning_use_linear_quantile_regr(residX, residY, alpha):
#     # bins = 5
#     # beta_ls = np.linspace(start=1e-5, stop=alpha-1e-5, num=bins)
#     bins = 1
#     beta_ls = [alpha/2]  # No search, as this is too slow.
#     width = np.zeros(bins)
#     width_left = np.zeros(bins)
#     width_right = np.zeros(bins)
#     for i in range(bins):
#         feature = residX[-1]
#         '''
#             Sklearn class
#             See scipy: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html#optimize-linprog-interior-point
#             for a list of option. "solver_options" are given as "options" therein
#
#             NOTE, we CANNOT afford many iterations, as this is VERY COSTLY (about 4 sec per point for this loop below even for 10 iterations...)
#             Even just 1 iter, stll like 2 sec
#
#             See sklearn for which solver to use:
#             https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor
#
#             BUT solver = 'highs' is claimed to be fast but actually does not work
#         '''
#         solver = 'interior-point'
#         sol_options = {'maxiter': 10}
#         reg_low = QuantileRegressor(
#             quantile=beta_ls[i], solver=solver, solver_options=sol_options)
#         reg_high = QuantileRegressor(
#             quantile=1 - alpha + beta_ls[i], solver=solver, solver_options=sol_options)
#         reg_low.fit(residX[:-1], residY)
#         reg_high.fit(residX[:-1], residY)
#         width_left[i] = reg_low.predict(feature.reshape(1, -1))
#         width_right[i] = reg_high.predict(feature.reshape(1, -1))
#         # ############################
#         # # Statsmodel class
#         # '''
#         #     https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html?highlight=quantreg
#         #     Actually, still not fast....
#         #     Hence, removed this "Optimizer width", but width can then be wider than necessary
#         # '''
#         # mod = QuantReg(residY, residX[:-1], max_iter=1)
#         # reg_low = mod.fit(q=beta_ls[i])
#         # reg_high = mod.fit(q=1-alpha+beta_ls[i])
#         # width_left[i] = mod.predict(reg_low.params, feature)
#         # width_right[i] = mod.predict(reg_high.params, feature)
#         width[i] = width_right[i] - width_left[i]
#     i_star = np.argmin(width)
#     return width_left[i_star], width_right[i_star]


#######
