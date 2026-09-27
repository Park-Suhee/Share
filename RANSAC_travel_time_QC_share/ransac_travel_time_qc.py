#!/usr/bin/env python3
"""Generic RANSAC + MAD quality control for P/S travel-time data."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,RANSACRegressor

INPUT_CSV=Path("example_travel_times.csv") ### need to edit!
OUTPUT_DIR=Path("output")
OUTPUT_CSV=OUTPUT_DIR/"ransac_qc_results.csv"
OUTPUT_PNG=OUTPUT_DIR/"ransac_qc_plot.png"

DISTANCE_COL="hypocentral_distance_km"
P_COL="P_travel_time"
S_COL="S_travel_time"

### Parameters
P_RANSAC_THRESHOLD=1.0
S_RANSAC_THRESHOLD=1.5
RANSAC_MIN_SAMPLES=0.5
RANSAC_MAX_TRIALS=1000
CHECK_SIGMA=3.0
STRONG_SIGMA=5.0
RANDOM_STATE=42

def make_ransac(threshold):
    kwargs=dict(min_samples=RANSAC_MIN_SAMPLES,residual_threshold=threshold,max_trials=RANSAC_MAX_TRIALS,random_state=RANDOM_STATE)
    try: return RANSACRegressor(estimator=LinearRegression(),**kwargs)
    except TypeError: return RANSACRegressor(base_estimator=LinearRegression(),**kwargs)

def robust_stats(residuals):
    x=np.asarray(residuals,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.nan,np.nan,np.nan
    median=np.median(x); mad=np.median(np.abs(x-median)); sigma=1.4826*mad
    return median,mad,sigma

def classify(residual,median,sigma):
    if not all(np.isfinite(v) for v in (residual,median,sigma)) or sigma<=0: return np.nan,"undefined"
    z=abs(residual-median)/sigma
    return (z,"normal") if z<=CHECK_SIGMA else (z,"check") if z<=STRONG_SIGMA else (z,"strong_outlier")

def run_phase(df,phase_col,prefix,threshold):
    valid=df[[DISTANCE_COL,phase_col]].replace([np.inf,-np.inf],np.nan).dropna().index
    if len(valid)<2: raise ValueError(f"Not enough valid observations for {prefix}.")
    X=df.loc[valid,[DISTANCE_COL]].to_numpy(); y=df.loc[valid,phase_col].to_numpy()
    model=make_ransac(threshold); model.fit(X,y)
    predicted=model.predict(X); residual=y-predicted; inlier=model.inlier_mask_
    df.loc[valid,f"{prefix}_predicted"]=predicted
    df.loc[valid,f"{prefix}_residual"]=residual
    df.loc[valid,f"{prefix}_RANSAC_inlier"]=inlier
    median,mad,sigma=robust_stats(residual[inlier])
    for idx,r in zip(valid,residual):
        z,qc=classify(r,median,sigma)
        df.loc[idx,f"{prefix}_robust_z"]=z
        df.loc[idx,f"{prefix}_QC"]=qc
    est=getattr(model,"estimator_",None)
    if est is None: est=model.base_estimator_
    return model,median,mad,sigma,float(est.coef_[0]),float(est.intercept_)

def plot_results(df,p_model,s_model):
    p=df[[DISTANCE_COL,P_COL]].dropna()
    s=df[[DISTANCE_COL,S_COL]].dropna()

    fig,ax=plt.subplots(figsize=(8,6))

    ### Normal P and S observations
    p_normal=df["P_QC"]=="normal"
    s_normal=df["S_QC"]=="normal"

    ax.scatter(df.loc[p_normal,DISTANCE_COL],df.loc[p_normal,P_COL],
               s=16,facecolors="none",edgecolors="red",label="P normal")
    ax.scatter(df.loc[s_normal,DISTANCE_COL],df.loc[s_normal,S_COL],
               s=16,facecolors="none",edgecolors="blue",label="S normal")

    ### RANSAC reference trends
    xmin=min(p[DISTANCE_COL].min(),s[DISTANCE_COL].min())
    xmax=max(p[DISTANCE_COL].max(),s[DISTANCE_COL].max())
    x=np.linspace(xmin,xmax,300).reshape(-1,1)

    ax.plot(x[:,0],p_model.predict(x),"--",color="red",label="P RANSAC")
    ax.plot(x[:,0],s_model.predict(x),"--",color="blue",label="S RANSAC")

    ### QC flags: color indicates phase, marker indicates QC level
    qc_styles={
        ("P","check"):("red","+"),
        ("S","check"):("blue","+"),
        ("P","strong_outlier"):("red","x"),
        ("S","strong_outlier"):("blue","x")
    }

    for (phase,qc),(color,marker) in qc_styles.items():
        mask=df[f"{phase}_QC"]==qc
        ax.scatter(df.loc[mask,DISTANCE_COL],
                   df.loc[mask,f"{phase}_travel_time"],
                   s=40,marker=marker,color=color,
                   label=f"{phase} {qc}")

    ax.set_xlabel("Hypocentral distance [km]")
    ax.set_ylabel("Travel time [s]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG,dpi=200)
    plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(INPUT_CSV)

    missing={DISTANCE_COL,P_COL,S_COL}-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")

    pm,pmed,pmad,psig,pa,pb=run_phase(df,P_COL,"P",P_RANSAC_THRESHOLD)
    sm,smed,smad,ssig,sa,sb=run_phase(df,S_COL,"S",S_RANSAC_THRESHOLD)

    df.to_csv(OUTPUT_CSV,index=False)
    plot_results(df,pm,sm)

    print(f"P: T={pa:.6f}R+{pb:.6f}, median={pmed:.6f}, MAD={pmad:.6f}, robust_sigma={psig:.6f}")
    print(f"S: T={sa:.6f}R+{sb:.6f}, median={smed:.6f}, MAD={smad:.6f}, robust_sigma={ssig:.6f}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")

if __name__=="__main__":
    main()