"""
time_series_anomaly.py

Two classes:
- time_series_anomaly_univariate
- time_series_anomaly_multivariate

Plus: lightweight loaders for time-series inputs from path/url/excel,
and synthetic/public fallback generators when no user inputs are provided.

Plots follow legend convention:
- blue line label: "time series data"
- red circular markers label: "prophet"
Axes are de-cluttered by formatting ticks (hourly/day), rotating labels, and tight layout.
"""

from typing import Optional, Sequence, Literal, Dict, Tuple
import io, os, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

# --------------------------- Loaders & utils ---------------------------

def load_timeseries(path: Optional[str]=None, url: Optional[str]=None, excel: Optional[str]=None,
                    time_col: Optional[str]=None, value_col: Optional[str]=None) -> pd.Series:
    """
    Load a univariate time series from local path / URL / Excel.
    Tries CSV, then Excel for URL. Returns a pandas.Series with DateTimeIndex.
    If value_col is None, picks the first numeric column.
    """
    if path and os.path.exists(path):
        df = pd.read_excel(path) if path.lower().endswith(('.xlsx','.xls')) else pd.read_csv(path)
    elif url:
        r = requests.get(url, timeout=30); r.raise_for_status()
        bio = io.BytesIO(r.content)
        try:
            df = pd.read_excel(bio)
        except Exception:
            bio.seek(0); df = pd.read_csv(bio)
    elif excel and os.path.exists(excel):
        df = pd.read_excel(excel)
    else:
        raise FileNotFoundError("No valid TS input. Provide path, url, or excel.")

    # Time index
    if time_col and time_col in df.columns:
        dt = pd.to_datetime(df[time_col])
        df = df.drop(columns=[time_col]).set_index(dt).sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        # try first column as datetime
        try:
            dt = pd.to_datetime(df.iloc[:,0])
            df = df.drop(df.columns[0], axis=1).set_index(dt).sort_index()
        except Exception as e:
            raise ValueError("Could not find/parse a datetime column.") from e

    # pick value column
    if value_col and value_col in df.columns:
        s = df[value_col].astype(float)
    else:
        num = df.select_dtypes(include='number')
        if num.shape[1]==0:
            raise ValueError("No numeric column found for TS values.")
        s = num.iloc[:,0].astype(float)
    s.name = s.name or "y"
    return s

def synth_timeseries(n:int=1000, freq:str="H", anomaly_frac:float=0.03, seed:int=42) -> Tuple[pd.Series, pd.Series]:
    """
    Robust synthetic time series with hourly seasonality and random spikes.
    Returns (series y, labels) where labels=1 at injected anomalies.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    t = np.arange(n)
    y = 5 + 0.002*t + 2*np.sin(2*np.pi*t/24) + rng.normal(0,0.4,size=n)
    k = max(1, int(anomaly_frac*n))
    an_idx = rng.choice(n, k, replace=False)
    y[an_idx] += rng.normal(6,2,size=k)
    s = pd.Series(y, index=idx, name="y")
    labels = pd.Series(0, index=idx)
    labels.iloc[an_idx]=1
    return s, labels

def public_timeseries_url(name:str="airline") -> str:
    """
    Provide a known public TS CSV URL.
    Options: 'airline' (AirPassengers), 'sunspots'.
    """
    if name=="airline":
        return "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    if name=="sunspots":
        return "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    raise ValueError("Unknown public TS name.")

def _safe_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        s = y.astype(float); s.name = s.name or "y"; return s
    if isinstance(y, (list, np.ndarray)):
        return pd.Series(y, name="y", dtype=float)
    raise TypeError("Input must be a pandas.Series, list, or numpy array.")

def _std(x: np.ndarray) -> float: return float(np.std(x) + 1e-12)
def _mad(x: np.ndarray) -> float:
    med = np.median(x); return float(np.median(np.abs(x - med)) + 1e-12)

def _window_features(y: pd.Series, lags: Sequence[int]=(1,2,3,24)) -> pd.DataFrame:
    df = pd.DataFrame({"y": y})
    for L in lags: df[f"lag_{L}"] = y.shift(L)
    return df.dropna()

def _plot_ts(y: pd.Series, scores: pd.Series, title:str):
    # top 3% anomalies by score
    k = max(1, int(0.03*len(scores)))
    thr = scores.sort_values(ascending=False).iloc[k-1]
    idx = scores[scores >= thr].index
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(y.index, y.values, label="time series data")
    ax.scatter(idx, y.loc[idx].values, marker="o", c="red", label="prophet")
    ax.set_title(title)
    ax.set_xlabel("time"); ax.set_ylabel("value")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    plt.xticks(rotation=20)
    plt.tight_layout()
    return idx

# ----------------- UNIVARIATE: A–I (key methods runnable on CPU) -----------------

class time_series_anomaly_univariate:
    """Univariate TS detectors (A–I). Returns pd.Series of anomaly scores."""

    def simple_stats(self, y: pd.Series,
                     method: Literal["zscore","modified_z","iqr","percentile","rolling_z","rolling_mad","rolling_iqr"]="rolling_z",
                     window:int=48, percentile:float=0.975, seasonal_key: Literal["hour","dow"]="hour") -> pd.Series:
        y = _safe_series(y); x = y.values
        if method == "zscore":
            mu, sd = np.mean(x), _std(x); s = np.abs((x - mu)/sd)
        elif method == "modified_z":
            med, mad = np.median(x), _mad(x); s = 0.6745 * np.abs(x - med) / mad
        elif method == "iqr":
            q1, q3 = np.percentile(x, [25,75]); iqr = q3-q1+1e-12
            lower, upper = q1-1.5*iqr, q3+1.5*iqr
            s = np.clip(lower-x,0,None)+np.clip(x-upper,0,None)
        elif method == "percentile":
            q = np.quantile(x, percentile); s = np.clip(x-q,0,None)
        elif method == "rolling_z":
            m = y.rolling(window, min_periods=window//2).mean()
            sd = y.rolling(window, min_periods=window//2).std().replace(0, np.nan)
            s = np.abs((y - m)/(sd+1e-12)).fillna(0).values
        elif method == "rolling_mad":
            med = y.rolling(window, min_periods=window//2).median()
            mad = (y-med).abs().rolling(window, min_periods=window//2).median() + 1e-12
            s = (0.6745*(y-med).abs()/mad).fillna(0).values
        elif method == "rolling_iqr":
            q1 = y.rolling(window, min_periods=window//2).quantile(0.25)
            q3 = y.rolling(window, min_periods=window//2).quantile(0.75)
            iqr = q3-q1
            lower, upper = q1-1.5*iqr, q3+1.5*iqr
            s = ((lower - y).clip(lower=0) + (y - upper).clip(lower=0)).fillna(0).values
        else:
            raise ValueError("Unknown method.")
        return pd.Series(np.asarray(s,float), index=y.index, name=f"A_simple_{method}")

    def decomposition(self, y: pd.Series, method: Literal["stl","ets","arima","kalman"]="stl", period:int=24) -> pd.Series:
        y = _safe_series(y)
        if method == "stl":
            stl = STL(y, period=period, robust=True); res = stl.fit()
            resid = res.resid.values; z = (resid - np.mean(resid))/(_std(resid))
            return pd.Series(np.abs(z), index=y.index, name="B_stl_resid_z")
        if method == "ets":
            fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=period, initialization_method="estimated").fit()
            resid = y.values - fit.fittedvalues.values
            z = (resid - np.mean(resid))/(_std(resid))
            return pd.Series(np.abs(z), index=y.index, name="B_ets_resid_z")
        if method == "arima":
            fit = ARIMA(y, order=(1,0,1)).fit()
            resid = fit.resid.values; z = (resid - np.mean(resid))/(_std(resid))
            return pd.Series(np.abs(z), index=y.index, name="B_arima_resid_z")
        if method == "kalman":
            mod = SARIMAX(y, order=(1,0,1), enforce_stationarity=False, enforce_invertibility=False)
            fit = mod.fit(disp=False)
            innov = fit.filter_results.standardized_forecasts_error[0]
            return pd.Series(np.abs(innov), index=y.index, name="B_kalman_innov_z")
        raise ValueError("Unknown method.")

    def classical_ml(self, y: pd.Series, method: Literal["ocsvm","iforest","lof","knn","kmeans"]="iforest",
                     lags: Sequence[int]=(1,2,3,24)) -> pd.Series:
        y = _safe_series(y)
        Z = _window_features(y, lags)
        X = StandardScaler().fit_transform(Z.values)
        if method == "ocsvm":
            m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X); s = -m.decision_function(X)
        elif method == "iforest":
            m = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(X); s = -m.score_samples(X)
        elif method == "lof":
            lof = LocalOutlierFactor(n_neighbors=20, novelty=False); lof.fit_predict(X); s = -lof.negative_outlier_factor_
        elif method == "knn":
            nn = NearestNeighbors(n_neighbors=5).fit(X); dists,_ = nn.kneighbors(X); s = dists.mean(axis=1)
        elif method == "kmeans":
            km = KMeans(n_clusters=8, n_init=10, random_state=42); lab = km.fit_predict(X); ctr = km.cluster_centers_
            dif = X - ctr[lab]; s = np.sqrt((dif**2).sum(axis=1))
        else: raise ValueError("Unknown method.")
        return pd.Series(s, index=Z.index, name=f"F_{method}")

    # helper plot
    def plot(self, y: pd.Series, scores: pd.Series, title:str="TS anomalies"):
        return _plot_ts(y, scores, title)


# ------------- MULTIVARIATE (key runnable parts) -------------

class time_series_anomaly_multivariate:
    """Multivariate TS detectors (A–H core baselines)."""

    def forecasting_residuals(self, X: pd.DataFrame, lags:int=1) -> pd.Series:
        model = VAR(X.dropna()); fit = model.fit(maxlags=lags, ic=None, trend="c")
        E = fit.resid; cov = EmpiricalCovariance().fit(E.values)
        md = cov.mahalanobis(E.values)
        return pd.Series(md, index=E.index, name="A_var_mahalanobis")

    def subspace_scores(self, X: pd.DataFrame, n_components:int=None) -> pd.Series:
        Xn = X.dropna(); n_components = n_components or max(1, min(Xn.shape[1],5))
        Z = StandardScaler().fit_transform(Xn.values)
        pca = PCA(n_components=n_components, random_state=42)
        T = pca.fit_transform(Z); Z_hat = pca.inverse_transform(T); E = Z - Z_hat
        covT = EmpiricalCovariance().fit(T); T2 = covT.mahalanobis(T); SPE = (E**2).sum(axis=1)
        T2z = (T2 - T2.mean())/(T2.std()+1e-12); SPEz = (SPE - SPE.mean())/(SPE.std()+1e-12)
        return pd.Series(T2z+SPEz, index=Xn.index, name="B_pca_T2_SPE")

    def distance_density(self, X: pd.DataFrame, method: Literal["iforest","lof","knn","kmeans","ocsvm","kde"]="iforest",
                         n_neighbors:int=20, k:int=8) -> pd.Series:
        Xn = X.dropna(); Z = StandardScaler().fit_transform(Xn.values)
        if method == "iforest":
            m = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(Z); s = -m.score_samples(Z)
        elif method == "lof":
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False); lof.fit_predict(Z); s = -lof.negative_outlier_factor_
        elif method == "knn":
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(Z); dists,_ = nn.kneighbors(Z); s = dists.mean(axis=1)
        elif method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=42); lab = km.fit_predict(Z); ctr = km.cluster_centers_
            dif = Z - ctr[lab]; s = np.sqrt((dif**2).sum(axis=1))
        elif method == "ocsvm":
            svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Z); s = -svm.decision_function(Z)
        elif method == "kde":
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(Z); s = -kde.score_samples(Z)
        else: raise ValueError("Unknown method.")
        return pd.Series(s, index=Xn.index, name=f"D_{method}")

# --------------------------- Public helper ---------------------------
def plot_ts_with_legends(y: pd.Series, anomalies_index: pd.Index, title:str):
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(y.index, y.values, label="time series data")
    ax.scatter(anomalies_index, y.loc[anomalies_index].values, marker="o", c="red", label="prophet")
    ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("value")
    ax.legend(); ax.xaxis.set_major_locator(MaxNLocator(nbins=7)); plt.xticks(rotation=20); plt.tight_layout()
    return ax
