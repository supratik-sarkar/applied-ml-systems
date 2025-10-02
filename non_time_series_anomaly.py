"""
non_time_series_anomaly.py

Two classes:
- non_time_series_anomaly_univariate
- non_time_series_anomaly_multivariate

Plus: loaders for tabular inputs from path/url/excel and synthetic/public fallbacks.
Plots follow legend convention:
- blue line label: "non time series data"
- red circular markers label: "prophet"
"""

from typing import Optional, Sequence, Literal, Tuple, List
import io, os, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# --------------------------- Loaders & utils ---------------------------

def load_tabular(path: Optional[str]=None, url: Optional[str]=None, excel: Optional[str]=None) -> pd.DataFrame:
    """Load tabular data (numeric columns retained) from path/url/excel."""
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
        raise FileNotFoundError("No valid tabular input. Provide path, url, or excel.")
    num = df.select_dtypes(include="number")
    if num.shape[1]==0: raise ValueError("No numeric columns in provided tabular data.")
    return num

def synth_tabular(n:int=1000, d:int=6, anomaly_frac:float=0.03, seed:int=7) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic multivariate tabular data with injected outliers. Returns (X, labels)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,size=(n,d))
    k = max(1, int(anomaly_frac*n))
    idx = rng.choice(n, k, replace=False)
    X[idx] += rng.normal(6,2,size=(k,d))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    y = pd.Series(0, index=df.index); y.iloc[idx]=1
    return df, y

def public_tabular_url(name:str="creditcard") -> str:
    """
    Provide a known public tabular dataset URL (small ones to keep Colab CPU happy).
    Options: 'iris' (UCI), 'wine'.
    """
    if name=="iris":
        return "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    if name=="wine":
        return "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/wine.csv"
    raise ValueError("Unknown public tabular name.")

def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.astype(float); s.name = s.name or "x"; return s
    if isinstance(x, (list, np.ndarray)):
        return pd.Series(x, name="x", dtype=float)
    raise TypeError("Input must be a pandas.Series, list, or numpy array.")

def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").copy()
    if num.shape[1]==0: raise ValueError("No numeric columns found.")
    return num

def _plot_scores(scores: pd.Series, title:str="Tabular anomaly scores"):
    scores = scores.astype(float)
    k = max(1, int(0.03*len(scores)))
    thr = scores.sort_values(ascending=False).iloc[k-1]
    idx = scores[scores >= thr].index
    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.plot(x, scores.values, label="non time series data")
    ax.scatter(np.where(scores.index.isin(idx))[0], scores.loc[idx].values, marker="o", c="red", label="prophet")
    ax.set_title(title); ax.set_xlabel("index"); ax.set_ylabel("score"); ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.tight_layout()
    return idx

# ----------------- UNIVARIATE -----------------

class non_time_series_anomaly_univariate:
    """Univariate (1D) tabular anomaly detectors."""

    def simple_stats(self, x: pd.Series,
                     method: Literal["zscore","modified_z","iqr","percentile","rolling_z","rolling_mad","rolling_iqr"]="modified_z",
                     window:int=50, percentile:float=0.975) -> pd.Series:
        x = _ensure_series(x); v = x.values
        if method=="zscore":
            mu, sd = np.mean(v), float(np.std(v)+1e-12); s = np.abs((v - mu)/sd)
        elif method=="modified_z":
            med = np.median(v); mad = float(np.median(np.abs(v-med))+1e-12); s = 0.6745 * np.abs(v-med) / mad
        elif method=="iqr":
            q1,q3 = np.percentile(v,[25,75]); iqr=q3-q1+1e-12; lower,upper = q1-1.5*iqr, q3+1.5*iqr
            s = np.clip(lower-v,0,None)+np.clip(v-upper,0,None)
        elif method=="percentile":
            q = np.quantile(v, percentile); s = np.clip(v-q,0,None)
        elif method=="rolling_z":
            m = x.rolling(window, min_periods=window//2).mean()
            sd = x.rolling(window, min_periods=window//2).std().replace(0, np.nan)
            s = np.abs((x - m)/(sd+1e-12)).fillna(0).values
        elif method=="rolling_mad":
            med = x.rolling(window, min_periods=window//2).median()
            mad = (x-med).abs().rolling(window, min_periods=window//2).median() + 1e-12
            s = (0.6745*(x-med).abs()/mad).fillna(0).values
        elif method=="rolling_iqr":
            q1 = x.rolling(window, min_periods=window//2).quantile(0.25)
            q3 = x.rolling(window, min_periods=window//2).quantile(0.75)
            iqr = q3-q1; lower,upper = q1-1.5*iqr, q3+1.5*iqr
            s = ((lower-x).clip(lower=0) + (x-upper).clip(lower=0)).fillna(0).values
        else: raise ValueError("Unknown method.")
        return pd.Series(np.asarray(s,float), index=x.index, name=f"UNI_A_{method}")

    def ml_1d(self, x: pd.Series, method: Literal["ocsvm","iforest","lof","knn","kmeans"]="iforest") -> pd.Series:
        x = _ensure_series(x); v = x.values.reshape(-1,1); z = StandardScaler().fit_transform(v)
        if method=="ocsvm":
            m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(z); s = -m.decision_function(z)
        elif method=="iforest":
            m = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(z); s = -m.score_samples(z)
        elif method=="lof":
            lof = LocalOutlierFactor(n_neighbors=20, novelty=False); lof.fit_predict(z); s = -lof.negative_outlier_factor_
        elif method=="knn":
            nn = NearestNeighbors(n_neighbors=5).fit(z); dists,_ = nn.kneighbors(z); s = dists.mean(axis=1)
        elif method=="kmeans":
            km = KMeans(n_clusters=8, n_init=10, random_state=42); lab = km.fit_predict(z)
            ctr = km.cluster_centers_; s = np.sqrt(((z - ctr[lab])**2).sum(axis=1))
        else: raise ValueError("Unknown method.")
        return pd.Series(s, index=x.index, name=f"UNI_C_{method}")

    def plot_univariate(self, x: pd.Series, scores: pd.Series, title:str="Univariate anomalies (tabular)"):
        x = _ensure_series(x); idx = _plot_scores(scores, title); return idx

# ----------------- MULTIVARIATE -----------------

class non_time_series_anomaly_multivariate:
    """Multivariate tabular anomaly detectors."""

    def distance_density(self, X: pd.DataFrame, method: Literal["iforest","lof","knn","kmeans","ocsvm","kde","gmm"]="iforest",
                         n_neighbors:int=20, k:int=8) -> pd.Series:
        num = _ensure_numeric_df(X); Z = StandardScaler().fit_transform(num.values)
        if method=="iforest":
            m = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(Z); s = -m.score_samples(Z)
        elif method=="lof":
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False); lof.fit_predict(Z); s = -lof.negative_outlier_factor_
        elif method=="knn":
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(Z); dists,_ = nn.kneighbors(Z); s = dists.mean(axis=1)
        elif method=="kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=42); lab = km.fit_predict(Z); ctr=km.cluster_centers_
            s = np.sqrt(((Z - ctr[lab])**2).sum(axis=1))
        elif method=="ocsvm":
            svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Z); s = -svm.decision_function(Z)
        elif method=="kde":
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(Z); s = -kde.score_samples(Z)
        elif method=="gmm":
            gmm = GaussianMixture(n_components=min(8, max(1, Z.shape[1]//2)), covariance_type="full", random_state=42).fit(Z); s = -gmm.score_samples(Z)
        else: raise ValueError("Unknown method.")
        return pd.Series(s, index=num.index, name=f"MV_A_{method}")

    def subspace(self, X: pd.DataFrame, n_components:int=None) -> pd.Series:
        num = _ensure_numeric_df(X); Z = StandardScaler().fit_transform(num.values)
        n_components = n_components or max(1, min(Z.shape[1],5))
        pca = PCA(n_components=n_components, random_state=42); T = pca.fit_transform(Z)
        Z_hat = pca.inverse_transform(T); E = Z - Z_hat
        covT = EmpiricalCovariance().fit(T); T2 = covT.mahalanobis(T); SPE = (E**2).sum(axis=1)
        T2z = (T2 - T2.mean())/(T2.std()+1e-12); SPEz = (SPE - SPE.mean())/(SPE.std()+1e-12)
        return pd.Series(T2z+SPEz, index=num.index, name="MV_B_pca_T2_SPE")

    def plot_scores(self, scores: pd.Series, title:str="Multivariate anomalies (tabular)"):
        return _plot_scores(scores, title)
