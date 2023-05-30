import sys
import importlib as ipb
import pandas as pd
import numpy as np
import math
import time as time
import utils_SPCI as utils
import warnings
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pdb
import data
import torch.nn as nn
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view
# from neuralprophet import NeuralProphet
from skranger.ensemble import RangerForestRegressor
warnings.filterwarnings("ignore")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#### Main Class ####


def detach_torch(input):
    return input.cpu().detach().numpy()


class SPCI_and_EnbPI():
    '''
        Create prediction intervals assuming Y_t = f(X_t) + \sigma(X_t)\eps_t
        Currently, assume the regression function is by default MLP implemented with PyTorch, as it needs to estimate BOTH f(X_t) and \sigma(X_t), where the latter is impossible to estimate using scikit-learn modules

        Most things carry out, except that we need to have different estimators for f and \sigma.

        fit_func = None: use MLP above
    '''

    def __init__(self, X_train, X_predict, Y_train, Y_predict, fit_func=None):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        n, n1 = len(self.X_train), len(self.X_predict)
        self.Ensemble_train_interval_centers = np.ones(n)*np.inf
        self.Ensemble_train_interval_sigma = np.ones(n)*np.inf
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = np.ones(n1)*np.inf
        self.Ensemble_pred_interval_sigma = np.ones(n1)*np.inf
        self.Ensemble_online_resid = np.ones(n+n1)*np.inf  # LOO scores
        self.beta_hat_bins = []
        ### New ones under development ###
        self.use_NeuralProphet = False
        #### Other hyperparameters for training (mostly simulation) ####
        # point predictor \hat f
        self.use_WLS = True # Whether to use WLS for fitting (compare with Nex-CP)
        self.WLS_c = 0.99
        # QRF training & how it treats the samples
        self.weigh_residuals = False # Whether we weigh current residuals more.
        self.c = 0.995 # If self.weight_residuals, weights[s] = self.c ** s, s\geq 0
        self.n_estimators = 10 # Num trees for QRF
        self.max_d = 2 # Max depth for fitting QRF
        self.criterion = 'mse' # 'mse' or 'mae'
        # search of \beta^* \in [0,\alpha]
        self.bins = 5 # break [0,\alpha] into bins
        # how many LOO training residuals to use for training current QRF 
        self.T1 = None # None = use all
    def one_boot_prediction(self, Xboot, Yboot, Xfull):
        if self.use_NeuralProphet:
            '''
                Added NeuralPropeht in
                Note, hyperparameters not tuned yet
            '''
            Xboot, Yboot = detach_torch(Xboot), detach_torch(Yboot)
            nlags = 1
            model = NeuralProphet(
                n_forecasts=1,
                n_lags=nlags,
            )
            df_tmp, _ = utils.make_NP_df(Xboot, Yboot)
            model = model.add_lagged_regressor(names=self.Xnames)
            _ = model.fit(df_tmp, freq="D")  # Also outputs the metrics
            boot_pred = model.predict(self.df_full)['yhat1'].to_numpy()
            boot_pred[:nlags] = self.Y_train[:nlags]
            boot_pred = boot_pred.astype(np.float)
            boot_fX_pred = torch.from_numpy(boot_pred.flatten()).to(device)
            boot_sigma_pred = 0
        else:
            if self.regressor.__class__.__name__ == 'NoneType':
                start1 = time.time()
                model_f = MLP(self.d).to(device)
                optimizer_f = torch.optim.Adam(
                    model_f.parameters(), lr=1e-3)
                if self.fit_sigmaX:
                    model_sigma = MLP(self.d, sigma=True).to(device)
                    optimizer_sigma = torch.optim.Adam(
                        model_sigma.parameters(), lr=2e-3)
                for epoch in range(300):
                    fXhat = model_f(Xboot)
                    sigmaXhat = torch.ones(len(fXhat)).to(device)
                    if self.fit_sigmaX:
                        sigmaXhat = model_sigma(Xboot)
                    loss = ((Yboot - fXhat)
                            / sigmaXhat).pow(2).mean() / 2
                    optimizer_f.zero_grad()
                    if self.fit_sigmaX:
                        optimizer_sigma.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    if self.fit_sigmaX:
                        optimizer_sigma.step()
                with torch.no_grad():
                    boot_fX_pred = model_f(
                        Xfull).flatten().cpu().detach().numpy()
                    boot_sigma_pred = 0
                    if self.fit_sigmaX:
                        boot_sigma_pred = model_sigma(
                            Xfull).flatten().cpu().detach().numpy()
                print(
                    f'Took {time.time()-start1} secs to finish the {self.b}th boostrap model')
            else:
                Xboot, Yboot = detach_torch(Xboot), detach_torch(Yboot)
                Xfull = detach_torch(Xfull)
                # NOTE, NO sigma estimation because these methods by deFAULT are fitting Y, but we have no observation of errors
                model = self.regressor
                if self.use_WLS and isinstance(model,LinearRegression):
                    # To compare with Nex-CP when using WLS
                    # Taken from Nex-CP code
                    n = len(Xboot)
                    tags=self.WLS_c**(np.arange(n,0,-1))
                    model.fit(Xboot, Yboot, sample_weight=tags)
                else:
                    model.fit(Xboot, Yboot)
                boot_fX_pred = torch.from_numpy(
                    model.predict(Xfull).flatten()).to(device)
                boot_sigma_pred = 0
            return boot_fX_pred, boot_sigma_pred

    def fit_bootstrap_models_online_multistep(self, B, fit_sigmaX=True, stride=1):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
          fit_sigmaX: If False, just avoid predicting \sigma(X_t) by defaulting it to 1

          stride: int. If > 1, then we perform multi-step prediction, where we have to fit stride*B boostrap predictors.
            Idea: train on (X_i,Y_i), i=1,...,n-stride
            Then predict on X_1,X_{1+s},...,X_{1+k*s} where 1+k*s <= n+n1
            Note, when doing LOO prediction thus only care above the points above
        '''
        n, self.d = self.X_train.shape
        self.fit_sigmaX = fit_sigmaX
        n1 = len(self.X_predict)
        N = n-stride+1  # Total training data each one-step predictor sees
        # We make prediction every s step ahead, so these are feature the model sees
        train_pred_idx = np.arange(0, n, stride)
        # We make prediction every s step ahead, so these are feature the model sees
        test_pred_idx = np.arange(n, n+n1, stride)
        self.train_idx = train_pred_idx
        self.test_idx = test_pred_idx
        # Only contains features that are observed every stride steps
        Xfull = torch.vstack(
            [self.X_train[train_pred_idx], self.X_predict[test_pred_idx-n]])
        nsub, n1sub = len(train_pred_idx), len(test_pred_idx)
        for s in range(stride):
            ''' 1. Create containers for predictions '''
            # hold indices of training data for each f^b
            boot_samples_idx = utils.generate_bootstrap_samples(N, N, B)
            # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
            in_boot_sample = np.zeros((B, N), dtype=bool)
            # hold predictions from each f^b for fX and sigma&b for sigma
            boot_predictionsFX = np.zeros((B, nsub+n1sub))
            boot_predictionsSigmaX = np.ones((B, nsub+n1sub))
            # We actually would just use n1sub rows, as we only observe this number of features
            out_sample_predictFX = np.zeros((n, n1sub))
            out_sample_predictSigmaX = np.ones((n, n1sub))

            ''' 2. Start bootstrap prediction '''
            start = time.time()
            if self.use_NeuralProphet:
                self.df_full, self.Xnames = utils.make_NP_df(
                    Xfull, np.zeros(n + n1))
            for b in range(B):
                self.b = b
                Xboot, Yboot = self.X_train[boot_samples_idx[b],
                                            :], self.Y_train[s:s+N][boot_samples_idx[b], ]
                in_boot_sample[b, boot_samples_idx[b]] = True
                boot_fX_pred, boot_sigma_pred = self.one_boot_prediction(
                    Xboot, Yboot, Xfull)
                boot_predictionsFX[b] = boot_fX_pred
                if self.fit_sigmaX:
                    boot_predictionsSigmaX[b] = boot_sigma_pred
            print(
                f'{s+1}/{stride} multi-step: finish Fitting {B} Bootstrap models, took {time.time()-start} secs.')

            ''' 3. Obtain LOO residuals (train and test) and prediction for test data '''
            start = time.time()
            # Consider LOO, but here ONLY for the indices being predicted
            for j, i in enumerate(train_pred_idx):
                # j: counter and i: actual index X_{0+j*stride}
                if i < N:
                    b_keep = np.argwhere(
                        ~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        # All bootstrap estimators are trained on this model
                        b_keep = 0  # More rigorously, it should be None, but in practice, the difference is minor
                else:
                    # This feature is not used in training, but used in prediction
                    b_keep = range(B)
                pred_iFX = boot_predictionsFX[b_keep, j].mean()
                pred_iSigmaX = boot_predictionsSigmaX[b_keep, j].mean()
                pred_testFX = boot_predictionsFX[b_keep, nsub:].mean(0)
                pred_testSigmaX = boot_predictionsSigmaX[b_keep, nsub:].mean(0)
                # Populate the training prediction
                # We add s because of multi-step procedure, so f(X_t) is for Y_t+s
                true_idx = min(i+s, n-1)
                self.Ensemble_train_interval_centers[true_idx] = pred_iFX
                self.Ensemble_train_interval_sigma[true_idx] = pred_iSigmaX
                resid_LOO = (detach_torch(
                    self.Y_train[true_idx]) - pred_iFX) / pred_iSigmaX
                out_sample_predictFX[i] = pred_testFX
                out_sample_predictSigmaX[i] = pred_testSigmaX
                self.Ensemble_online_resid[true_idx] = resid_LOO.item()
            sorted_out_sample_predictFX = out_sample_predictFX[train_pred_idx].mean(
                0)  # length ceil(n1/stride)
            sorted_out_sample_predictSigmaX = out_sample_predictSigmaX[train_pred_idx].mean(
                0)  # length ceil(n1/stride)
            pred_idx = np.minimum(test_pred_idx-n+s, n1-1)
            self.Ensemble_pred_interval_centers[pred_idx] = sorted_out_sample_predictFX
            self.Ensemble_pred_interval_sigma[pred_idx] = sorted_out_sample_predictSigmaX
            pred_full_idx = np.minimum(test_pred_idx+s, n+n1-1)
            resid_out_sample = (
                detach_torch(self.Y_predict[pred_idx]) - sorted_out_sample_predictFX) / sorted_out_sample_predictSigmaX
            self.Ensemble_online_resid[pred_full_idx] = resid_out_sample
        # Sanity check
        num_inf = (self.Ensemble_online_resid == np.inf).sum()
        if num_inf > 0:
            print(
                f'Something can be wrong, as {num_inf}/{n+n1} residuals are not all computed')
            print(np.where(self.Ensemble_online_resid == np.inf))

    def compute_PIs_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_SPCI=False, quantile_regr='RF'):
        '''
            stride: control how many steps we predict ahead
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_SPCI: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        '''
        self.alpha = alpha
        n1 = len(self.X_train)
        self.past_window = past_window # For SPCI, this is the "lag" for predicting quantile
        if smallT:
            # Namely, for special use of EnbPI, only use at most past_window number of LOO residuals.
            n1 = min(self.past_window, len(self.X_train))
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma
        start = time.time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        if use_SPCI:
            s = stride
            stride = 1
        # NOTE, NOT ALL rows are actually "observable" in multi-step context, as this is rolling
        resid_strided = utils.strided_app(
            self.Ensemble_online_resid[len(self.X_train) - n1:-1], n1, stride)
        print(f'Shape of slided residual lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        # # NEW, alpha becomes alpha_t. Uncomment things below if we decide to use this upgraded EnbPI
        # alpha_t = alpha
        # errs = []
        # gamma = 0.005
        # method = 'simple'  # 'simple' or 'complex'
        # self.alphas = []
        # NOTE: 'max_features='log2', max_depth=2' make the model "simpler", which improves performance in practice
        self.QRF_ls = []
        self.i_star_ls = []
        for i in range(num_unique_resid):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    # Update QRF
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                # Use the fitted regressor.
                # NOTE, residX is NOT the same as before, as it depends on
                # "past_resid", which has most entries replaced.
                rfqr= self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all)/2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid+i_star]
                width_left[i] = curr_SigmaX * wid_left
                width_right[i] = curr_SigmaX * wid_right
                num_print = int(num_unique_resid / 20)
                if num_print == 0:
                    print(
                            f'Width at test {i} is {width_right[i]-width_left[i]}')
                else:
                    if i % num_print == 0:
                        print(
                            f'Width at test {i} is {width_right[i]-width_left[i]}')
            else:
                past_resid = resid_strided[i, :]
                # Naive empirical quantile, where we use the SAME residuals for multi-step prediction
                # The number of bins will be determined INSIDE binning
                beta_hat_bin = utils.binning(past_resid, alpha)
                # beta_hat_bin = utils.binning(past_resid, alpha_t)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = curr_SigmaX * np.percentile(
                    past_resid, math.ceil(100 * beta_hat_bin))
                width_right[i] = curr_SigmaX * np.percentile(
                    past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
        print(
            f'Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time.time()-start} secs.')
        Ntest = len(out_sample_predict)
        # This is because |width|=T1/stride.
        width_left = np.repeat(width_left, stride)[:Ntest]
        # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)[:Ntest]
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict + width_left,
                                          out_sample_predict + width_right], columns=['lower', 'upper'])
        self.PIs_Ensemble = PIs_Ensemble
    '''
        Get Multi-step QRF
    '''

    def multi_step_QRF(self, past_resid, i, s, n2):
        '''
            Train multi-step QRF with the most recent residuals
            i: prediction index
            s: num of multi-step, same as stride
            n2: past window w
        '''
        # 1. Get "past_resid" into an auto-regressive fashion
        # This should be more carefully examined, b/c it depends on how long \hat{\eps}_t depends on the past
        # From practice, making it small make intervals wider
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        for k in range(s):
            residY = past_resid[n2+k:num-(s-k-1)]
            self.train_QRF(residX, residY)
            if i == 0:
                # Initial training, append QRF to QRF_ls
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                # Retraining, update QRF to QRF_ls
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        return resid_pred

    def train_QRF(self, residX, residY):
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        self.common_params = dict(n_estimators = self.n_estimators,
                                  max_depth = self.max_d,
                                  criterion = self.criterion,
                                  n_jobs = -1)
        if residX[:-1].shape[0] > 10000:
            # see API ref. https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html?highlight=RandomForestQuantileRegressor#sklearn_quantile.RandomForestQuantileRegressor
            # NOTE, should NOT warm start, as it makes result poor
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        # 3. Find best \hat{\beta} via evaluating many quantiles
        # rfqr.fit(residX[:-1], residY)
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY)) # Sanity check to make sure no errors in training
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, residX[-(self.T1+1):-1], residY[-self.T1:], residX[-1], beta_ls, sample_weight)
        else:
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, residX[:-1], residY, residX[-1], beta_ls, sample_weight)
    '''
        All together
    '''

    def get_results(self, alpha, data_name, itrial, true_Y_predict=[], method='Ensemble'):
        '''
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
        '''
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        train_size = len(self.X_train)
        if method == 'Ensemble':
            PI = self.PIs_Ensemble
        Ytest = self.Y_predict.cpu().detach().numpy()
        coverage = ((np.array(PI['lower']) <= Ytest) & (
            np.array(PI['upper']) >= Ytest)).mean()
        if len(true_Y_predict) > 0:
            coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                np.array(PI['upper']) >= true_Y_predict)).mean()
        print(f'Average Coverage is {coverage}')
        width = (PI['upper'] - PI['lower']).mean()
        print(f'Average Width is {width}')
        results.loc[len(results)] = [itrial, data_name,
                                     'torch_MLP', method, train_size, coverage, width]
        return results


class MLP(nn.Module):
    def __init__(self, d, sigma=False):
        super(MLP, self).__init__()
        H = 64
        layers = [nn.Linear(d, H), nn.ReLU(), nn.Linear(
            H, H), nn.ReLU(), nn.Linear(H, 1)]
        self.sigma = sigma
        if self.sigma:
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        perturb = 1e-3 if self.sigma else 0
        return self.layers(x) + perturb


#### Competing Methods ####


class QOOB_or_adaptive_CI():
    '''
        Implementation of the QOOB method (Gupta et al., 2021) or the adaptive CI (Gibbs et al., 2022)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
    ##############################
    # First on QOOB

    def fit_bootstrap_agg_get_lower_upper(self, B, beta_quantiles):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, compute scors r_i(X_i,Y_i), and finally get the intervals [l_i(X_n+j),u_i(X_n+j)] for each LOO predictor and the jth prediction in test sample
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = utils.generate_bootstrap_samples(n, n, B)
        # hold lower and upper quantile predictions from each f^b
        boot_predictions_lower = np.zeros((B, (n + n1)), dtype=float)
        boot_predictions_upper = np.zeros((B, (n + n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict_lower = np.zeros((n, n1))
        out_sample_predict_upper = np.zeros((n, n1))
        start = time.time()
        for b in range(B):
            # Fit quantile random forests
            model = self.regressor
            model = model.fit(self.X_train[boot_samples_idx[b], :],
                              self.Y_train[boot_samples_idx[b], ])
            pred_boot = model.predict_quantiles(
                np.r_[self.X_train, self.X_predict], quantiles=beta_quantiles)
            boot_predictions_lower[b] = pred_boot[:, 0]
            boot_predictions_upper[b] = pred_boot[:, 1]
            in_boot_sample[b, boot_samples_idx[b]] = True
        print(
            f'Finish Fitting B Bootstrap models, took {time.time()-start} secs.')
        start = time.time()
        self.QOOB_rXY = []  # the non-conformity scores
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                # NOTE: Append these training centers too see their magnitude
                # The reason is sometimes they are TOO close to actual Y.
                quantile_lower = boot_predictions_lower[b_keep, i].mean()
                quantile_upper = boot_predictions_upper[b_keep, i].mean()
                out_sample_predict_lower[i] = boot_predictions_lower[b_keep, n:].mean(
                    0)
                out_sample_predict_upper[i] = boot_predictions_upper[b_keep, n:].mean(
                    0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                print(f'No bootstrap estimator for {i}th LOO estimator')
                quantile_lower = np.percentile(
                    self.Y_train, beta_quantiles[0] * 100)
                quantile_upper = np.percentile(
                    self.Y_train, beta_quantiles[1] * 100)
                out_sample_predict_lower[i] = np.repeat(quantile_lower, n1)
                out_sample_predict_upper[i] = np.repeat(quantile_upper, n1)
            self.QOOB_rXY.append(self.get_rXY(
                self.Y_train[i], quantile_lower, quantile_upper))
        # print('Finish Computing QOOB training' +
        #       r'$\{r_i(X_i,Y_i)\}_{i=1}^N$'+f', took {time.time()-start} secs.')
        # Finally, subtract/add the QOOB_rXY from the predictions
        self.QOOB_rXY = np.array(self.QOOB_rXY)
        out_sample_predict_lower = (
            out_sample_predict_lower.transpose() - self.QOOB_rXY).transpose()
        out_sample_predict_upper = (
            out_sample_predict_upper.transpose() + self.QOOB_rXY).transpose()
        F_minus_i_out_sample = np.r_[
            out_sample_predict_lower, out_sample_predict_upper]
        return F_minus_i_out_sample  # Matrix of shape 2n-by-n1

    def compute_QOOB_intervals(self, data_name, itrial, B, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        beta_quantiles = [alpha * 2, 1 - alpha * 2]
        # beta_quantiles = [alpha/2, 1-alpha/2]  # Even make thresholds smaller, still not good
        F_minus_i_out_sample = self.fit_bootstrap_agg_get_lower_upper(
            B, beta_quantiles)
        n1 = F_minus_i_out_sample.shape[1]
        PIs = []
        for i in range(n1):
            curr_lower_upper = F_minus_i_out_sample[:, i]
            # print(f'Test point {i}')
            PIs.append(self.get_lower_upper_n_plus_i(curr_lower_upper, alpha))
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        self.PIs = PIs
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'QOOB', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
    # QOOB helpers

    def get_rXY(self, Ytrain_i, quantile_lower, quantile_upper):
        # Get r_i(X_i,Y_i) as in Eq. (2) of QOOB
        if Ytrain_i < quantile_lower:
            return quantile_lower - Ytrain_i
        elif Ytrain_i > quantile_upper:
            return Ytrain_i - quantile_upper  # There was a small error here
        else:
            return 0

    # AdaptCI helpers
    def get_Ei(self, Ytrain_i, quantile_lower, quantile_upper):
        return np.max([quantile_lower - Ytrain_i, Ytrain_i - quantile_upper])

    def get_lower_upper_n_plus_i(self, curr_lower_upper, alpha):
        # This implements Algorithm 1 of QOOB
        # See https://github.com/AIgen/QOOB/blob/master/MATLAB/methods/QOOB_interval.m for matlab implementation
        n2 = len(curr_lower_upper)
        n = int(n2 / 2)
        S_ls = np.r_[np.repeat(1, n), np.repeat(0, n)]
        idx_sort = np.argsort(curr_lower_upper)  # smallest to larget
        S_ls = S_ls[idx_sort]
        curr_lower_upper = curr_lower_upper[idx_sort]
        count = 0
        lower_i = np.inf
        upper_i = -np.inf
        threshold = alpha * (n + 1) - 1
        for i in range(n2):
            if S_ls[i] == 1:
                count += 1
                if count > threshold and count - 1 <= threshold and lower_i == np.inf:
                    lower_i = curr_lower_upper[i]
                    # print(f'QOOB lower_end {lower_i}')
            else:
                if count > threshold and count - 1 <= threshold and upper_i == -np.inf:
                    upper_i = curr_lower_upper[i]
                    # print(f'QOOB upper_end {upper_i}')
                count -= 1
        return [lower_i, upper_i]

    ##############################
    # Next on AdaptiveCI

    def compute_AdaptiveCI_intervals(self, data_name, itrial, l, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        n = len(self.X_train)
        proper_train = np.arange(l)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # NOTE: below works when the model can takes in MULTIPLE quantiles together (e.g., the RangerForest)
        model = self.regressor
        model = model.fit(X_train, Y_train)
        quantile_pred = model.predict_quantiles(
            np.r_[X_calibrate, self.X_predict], quantiles=[alpha / 2, 1 - alpha / 2])
        # NOTE: below works for sklearn linear quantile: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor
        # # In particular, it is much slower than the quantile RF with similar results
        # model_l, model_u = self.regressor
        # qpred_l, qpred_u = model_l.fit(X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict]), model_u.fit(
        #     X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict])
        # quantile_pred = np.c_[qpred_l, qpred_u]
        self.quantile_pred = quantile_pred
        Dcal_pred = quantile_pred[:n - l]
        Test_pred = quantile_pred[n - l:]
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100 * (1 - alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred - width, upper_pred + width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = utils.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100 * (1 - alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred - width, upper_pred + width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = utils.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results


def NEX_CP(X, Y, x, alpha, weights=[], tags=[], seed=1103):
    '''
    # Barber et al. 2022: Nex-CP
    # weights are used for computing quantiles for the prediction interval
    # tags are used as weights in weighted least squares regression
    '''
    n = len(Y)

    if(len(tags) == 0):
        tags = np.ones(n + 1)

    if(len(weights) == 0):
        weights = np.ones(n + 1)
    if(len(weights) == n):
        weights = np.r_[weights, 1]
    weights = weights / np.sum(weights)
    np.random.seed(seed)
    # randomly permute one weight for the regression
    random_ind = int(np.where(np.random.multinomial(1, weights, 1))[1])
    tags[np.c_[random_ind, n]] = tags[np.c_[n, random_ind]]

    XtX = (X.T * tags[:-1]).dot(X) + np.outer(x, x) * tags[-1]
    a = Y - X.dot(np.linalg.solve(XtX, (X.T * tags[:-1]).dot(Y)))
    b = -X.dot(np.linalg.solve(XtX, x)) * tags[-1]
    a1 = -x.T.dot(np.linalg.solve(XtX, (X.T * tags[:-1]).dot(Y)))
    b1 = 1 - x.T.dot(np.linalg.solve(XtX, x)) * tags[-1]
    # if we run weighted least squares on (X[1,],Y[1]),...(X[n,],Y[n]),(x,y)
    # then a + b*y = residuals of data points 1,..,n
    # and a1 + b1*y = residual of data point n+1

    y_knots = np.sort(
        np.unique(np.r_[((a - a1) / (b1 - b))[b1 - b != 0], ((-a - a1) / (b1 + b))[b1 + b != 0]]))
    y_inds_keep = np.where(((np.abs(np.outer(a1 + b1 * y_knots, np.ones(n)))
                             > np.abs(np.outer(np.ones(len(y_knots)), a) + np.outer(y_knots, b))) *
                            weights[:-1]).sum(1) <= 1 - alpha)[0]
    y_PI = np.array([y_knots[y_inds_keep.min()], y_knots[y_inds_keep.max()]])
    if(weights[:-1].sum() <= 1 - alpha):
        y_PI = np.array([-np.inf, np.inf])
    return y_PI

#### Testing functions based on methods above #####


wind_loc = 0  # Can change this to try wind prediction on different locations


def test_EnbPI_or_SPCI(main_condition, results_EnbPI_SPCI, itrial=0):
    '''
    Arguments:

        main_condition: Contain these three below:
            bool. simulation:  True use simulated data. False use solar
                simul_type: int. 1 = simple state-space. 2 = non-statioanry. 3 = heteroskedastic
                The latter 2 follows from case 3 in paper
            bool. use_SPCI: True use `quantile_regr`. False use empirical quatile
            str. quantile_regr:  Which quantile regression to fit residuals (e.g., "RF", "LR")

    Other (not arguments)

        fit_func: None or sklearn module with methods `.fit` & `.predict`. If None, use MLP above

        fit_sigmaX: bool. True if to fit heteroskedastic errors. ONLY activated if fit_func is NONE (i.e. MLP), because errors are unobserved so `.fit()` does not work

        smallT: bool. True if empirical quantile uses not ALL T residual in the past to get quantile (should be tuned as sometimes longer memory causes poor coverage)
            past_window: int. If smallT True, EnbPI uses `past_window` most residuals to get width. FOR quantile_regr of residuals, it determines the dimension of the "feature" that predict new quantile of residuals autoregressively

    Results:
        dict: contains dictionary of coverage and width under different training fraction (fix alpha) under various argument combinations
    '''
    simulation, use_SPCI, quantile_regr, use_NeuralProphet = main_condition
    non_stat_solar, save_dict_rolling = results_EnbPI_SPCI.other_conditions
    train_ls, alpha = results_EnbPI_SPCI.train_ls, results_EnbPI_SPCI.alpha
    univariate, filter_zero = results_EnbPI_SPCI.data_conditions
    result_cov, result_width = [], []
    for train_frac in train_ls:
        print('########################################')
        print(f'Train frac at {train_frac}')
        ''' Get Data '''
        if simulation:
            simul_type = results_EnbPI_SPCI.simul_type  # 1, 2, 3
            fit_sigmaX = True if simul_type == 3 else False  # If we fit variance given X_t
            simul_name_dict = {1: 'simulation_state_space',
                               2: 'simulate_nonstationary', 3: 'simulate_heteroskedastic'}
            data_name = simul_name_dict[simul_type]
            simul_loader = data.simulate_data_loader()
            Data_dict = simul_loader.get_simul_data(simul_type)
            X_full, Y_full = Data_dict['X'].to(
                device), Data_dict['Y'].to(device)
            B = 20
            past_window = 500
            fit_func = None
            # if simul_type == 3:
            #     fit_func = None  # It is MLP above
            # else:
            #     fit_func = RandomForestRegressor(n_estimators=10, criterion='mse',
            #                                      bootstrap=False, n_jobs=-1, random_state=1103+itrial)
        else:
            data_name = results_EnbPI_SPCI.data_name
            dloader = data.real_data_loader()
            solar_args = [univariate, filter_zero, non_stat_solar]
            wind_args = [wind_loc]
            X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
            RF_seed = 1103+itrial
            if data_name == 'solar':
                fit_func = RandomForestRegressor(n_estimators=10, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 200 if use_SPCI else 300
            if data_name == 'electric':
                fit_func = RandomForestRegressor(n_estimators=10, max_depth=1, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 300
            if data_name == 'wind':
                fit_func = RandomForestRegressor(n_estimators=10, max_depth=1, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 300
            Y_full, X_full = torch.from_numpy(Y_full).float().to(
                device), torch.from_numpy(X_full).float().to(device)
            fit_sigmaX = False
            B = 25
        N = int(X_full.shape[0] * train_frac)
        X_train, X_predict, Y_train, Y_predict = X_full[:
                                                        N], X_full[N:], Y_full[:N], Y_full[N:]

        ''' Train '''
        EnbPI = SPCI_and_EnbPI(
            X_train, X_predict, Y_train, Y_predict, fit_func=fit_func)
        EnbPI.use_NeuralProphet = use_NeuralProphet
        stride = results_EnbPI_SPCI.stride
        EnbPI.fit_bootstrap_models_online_multistep(
            B, fit_sigmaX=fit_sigmaX, stride=stride)
        # Under cond quantile, we are ALREADY using the last window for prediction so smallT is FALSE, instead, we use ALL residuals in the past (in a sliding window fashion) for training the quantile regressor
        smallT = not use_SPCI
        EnbPI.compute_PIs_Ensemble_online(
            alpha, smallT=smallT, past_window=past_window, use_SPCI=use_SPCI,
            quantile_regr=quantile_regr, stride=stride)
        results = EnbPI.get_results(alpha, data_name, itrial)

        ''' Save results '''
        result_cov.append(results['coverage'].item())
        result_width.append(results['width'].item())
        PI = EnbPI.PIs_Ensemble
        if use_SPCI:
            if use_NeuralProphet:
                results_EnbPI_SPCI.PIs_SPCINeuralProphet = PI
            else:
                results_EnbPI_SPCI.PIs_SPCI = PI
        else:
            results_EnbPI_SPCI.PIs_EnbPI = PI
        Ytest = EnbPI.Y_predict.cpu().detach().numpy()
        results_EnbPI_SPCI.dict_rolling[f'Itrial{itrial}'] = PI
        name = 'SPCI' if use_SPCI else 'EnbPI'
        if use_NeuralProphet:
            name = 'SPCI-NeuralProphet'
        if save_dict_rolling:
            with open(f'{name}_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_EnbPI_SPCI.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if simulation:
            # # Examine recovery of F and Sigma
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].plot(Data_dict['f(X)'])
            Y_t_hat = EnbPI.Ensemble_pred_interval_centers
            ax[0, 1].plot(Y_t_hat)
            ax[1, 0].plot(Data_dict['Eps'])
            ax[1, 1].plot(EnbPI.Ensemble_online_resid)
            titles = [r'True $f(X)$', r'Est $f(X)$',
                      r'True $\epsilon$', r'Est $\epsilon$']
            fig.tight_layout()
            for i, ax_i in enumerate(ax.flatten()):
                ax_i.set_title(titles[i])
            fig.tight_layout()
            plt.show()
            plt.close()
    results_EnbPI_SPCI.dict_full[name] = np.vstack(
        [result_cov, result_width])
    results_EnbPI_SPCI.Ytest = Ytest
    results_EnbPI_SPCI.train_size = N
    results_EnbPI_SPCI.data_name = data_name
    utils.dict_to_latex(results_EnbPI_SPCI.dict_full, train_ls)
    return results_EnbPI_SPCI


def test_adaptive_CI(results_Adapt_CI, itrial=0):
    train_ls, alpha = results_Adapt_CI.train_ls, results_Adapt_CI.alpha
    non_stat_solar, save_dict_rolling = results_Adapt_CI.other_conditions
    univariate, filter_zero = results_Adapt_CI.data_conditions
    # NOTE: the variance of this method seems high, and I often need to tune a LOT to avoid yielding very very high coverage.
    data_name = results_Adapt_CI.data_name
    cov_ls, width_ls = [], []
    for train_frac in train_ls:
        # As it is split conformal, the result can be random, so we repeat over seed
        seeds = [524, 1103, 1111, 1214, 1228]
        seeds = [seed+itrial+1231 for seed in seeds]
        cov_tmp_ls, width_tmp_ls = [], []
        print('########################################')
        print(f'Train frac at {train_frac} over {len(seeds)} seeds')
        PI_ls = []
        for seed in seeds:
            data_name = results_Adapt_CI.data_name
            dloader = data.real_data_loader()
            solar_args = [univariate, filter_zero, non_stat_solar]
            wind_args = [wind_loc]
            X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
            N = int(X_full.shape[0] * train_frac)
            X_train, X_predict, Y_train, Y_predict = X_full[:
                                                            N], X_full[N:], Y_full[:N], Y_full[N:]
            if non_stat_solar:
                # More complex yields wider intervals and more conservative coverage
                fit_func = RangerForestRegressor(
                    n_estimators=5, quantiles=True, seed=seed)
            else:
                fit_func = RangerForestRegressor(
                    n_estimators=10, quantiles=True, seed=seed)
            PI_test_adaptive = QOOB_or_adaptive_CI(
                fit_func, X_train, X_predict, Y_train, Y_predict)
            PI_test_adaptive.compute_AdaptiveCI_intervals(
                data_name, 0, l=int(0.75 * X_train.shape[0]),
                alpha=alpha)
            PIs_AdaptiveCI = PI_test_adaptive.PIs
            PI_ls.append(PIs_AdaptiveCI)
            Ytest = PI_test_adaptive.Y_predict
            coverage = ((np.array(PIs_AdaptiveCI['lower']) <= Ytest)
                        & (np.array(PIs_AdaptiveCI['upper']) >= Ytest))
            width = (
                (np.array(PIs_AdaptiveCI['upper']) - np.array(PIs_AdaptiveCI['lower'])))
            cov_tmp_ls.append(coverage)
            width_tmp_ls.append(width)
        lowers = np.mean([a['lower'] for a in PI_ls], axis=0)
        uppers = np.mean([a['upper'] for a in PI_ls], axis=0)
        PIs_AdaptiveCI = pd.DataFrame(
            np.c_[lowers, uppers], columns=['lower', 'upper'])
        coverage = np.vstack(cov_tmp_ls).mean(axis=0)
        width = np.vstack(width_tmp_ls).mean(axis=0)
        results_Adapt_CI.PIs_AdaptiveCI = PIs_AdaptiveCI
        results_Adapt_CI.dict_rolling[f'Itrial{itrial}'] = PIs_AdaptiveCI
        if save_dict_rolling:
            with open(f'AdaptiveCI_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_Adapt_CI.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
        cov_ls.append(np.mean(coverage))
        width_ls.append(np.mean(width))
    results_Adapt_CI.dict_full['AdaptiveCI'] = np.vstack(
        [cov_ls, width_ls])
    utils.dict_to_latex(results_Adapt_CI.dict_full, train_ls)
    return results_Adapt_CI


def test_NEX_CP(results_NEX_CP, itrial=0):
    train_ls, alpha = results_NEX_CP.train_ls, results_NEX_CP.alpha
    non_stat_solar, save_dict_rolling = results_NEX_CP.other_conditions
    univariate, filter_zero = results_NEX_CP.data_conditions
    cov, width = [], []
    data_name = results_NEX_CP.data_name
    dloader = data.real_data_loader()
    solar_args = [univariate, filter_zero, non_stat_solar]
    wind_args = [wind_loc]
    X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
    N = len(X_full)
    for train_frac in train_ls:
        train_size = int(train_frac * N)
        PI_nexCP_WLS = np.zeros((N, 2))
        for n in np.arange(train_size, N):
            # weights and tags (parameters for new methods)
            rho = 0.99
            rho_LS = 0.99
            weights = rho**(np.arange(n, 0, -1))
            tags = rho_LS**(np.arange(n, -1, -1))
            PI_nexCP_WLS[n, :] = NEX_CP(X_full[:n, :], Y_full[:n], X_full[n, :], alpha,
                                        weights=weights, tags=tags, seed=1103+itrial)
            inc = int((N - train_size) / 20)
            if (n - train_size) % inc == 0:
                print(
                    f'NEX-CP WLS width at {n-train_size} is: {PI_nexCP_WLS[n,1] - PI_nexCP_WLS[n,0]}')
        cov_nexCP_WLS = (PI_nexCP_WLS[train_size:, 0] <= Y_full[train_size:N]) *\
            (PI_nexCP_WLS[train_size:, 1] >= Y_full[train_size:N])
        PI_width_nexCP_WLS = PI_nexCP_WLS[train_size:,
                                          1] - PI_nexCP_WLS[train_size:, 0]
        PI_nexCP_WLS = PI_nexCP_WLS[train_size:]
        PI_nexCP_WLS = pd.DataFrame(PI_nexCP_WLS, columns=['lower', 'upper'])
        cov.append(np.mean(cov_nexCP_WLS))
        width.append(np.mean(PI_width_nexCP_WLS))
        print(
            f'At {train_frac} tot data \n cov: {cov[-1]} & width: {width[-1]}')
        # Rolling coverage and width
        # cov_moving = utils.rolling_avg(cov_nexCP_WLS)
        # width_moving = utils.rolling_avg(PI_width_nexCP_WLS)
        results_NEX_CP.PI_nexCP_WLS = PI_nexCP_WLS
        results_NEX_CP.dict_rolling[f'Itrial{itrial}'] = PI_nexCP_WLS
        if save_dict_rolling:
            with open(f'NEXCP_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_NEX_CP.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
    results_NEX_CP.dict_full['NEXCP'] = np.vstack([cov, width])
    utils.dict_to_latex(results_NEX_CP.dict_full, train_ls)
    return results_NEX_CP

################################################################
################################################################

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
