#!/usr/bin/env python3
"""Compare current M1 against Praj M2 as clarified by email.
Praj noise estimators are exploratory and do not establish his original QC logic.
"""
import functions as common
from _core import input_core as base
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib.pyplot as plt
from obspy import read,UTCDateTime


def waveform_index(sac_dir):
    index={}
    for path in sorted(sac_dir.iterdir()):
        if not path.is_file():continue
        try:stream=read(str(path))
        except Exception:continue
        for tr in stream:
            net,sta,cha=common.trace_id(tr)
            if not sta or not cha or cha[-1] not in 'NEZ':continue
            key=(net,sta,cha[:2],cha[-1])
            if key not in index or tr.stats.npts>index[key].stats.npts:index[key]=tr
    return index


def get_traces(index,obs):
    key=(obs['network'],obs['station'],obs['channel_family'][:2])
    traces={c:index.get((*key,c)) for c in 'NEZ'}
    if any(tr is None for tr in traces.values()):raise ValueError('missing matching N/E/Z traces')
    return {c:tr.copy() for c,tr in traces.items()}


M2_PREFILT=(0.5,0.6,5.0,35.0)
M2_FILTER=(2.0,15.0)

def prepare(tr,inventory,lo,hi,prefilt):
    return common.prepare_trace(tr,inventory,lo,hi,prefilt,base.RESPONSE_WATER_LEVEL,True)


def align(traces):
    aligned=common.align_three(traces['N'],traces['E'],traces['Z']) if set(traces)=={'N','E','Z'} else common.align_three(traces['R'],traces['T'],traces['Z'])
    if any(tr is None for tr in aligned):raise ValueError('no common waveform duration or insufficient samples')
    keys=('N','E','Z') if set(traces)=={'N','E','Z'} else ('R','T','Z')
    if len({round(tr.stats.sampling_rate,8) for tr in aligned})!=1:raise ValueError('different component sampling rates')
    return dict(zip(keys,aligned))


def get_rtz(raw,obs,event,inventory,lo,hi,prefilt):
    xyz=align({c:prepare(tr,inventory,lo,hi,prefilt) for c,tr in raw.items()})
    r,t,baz=common.rotate_ne_to_rt(xyz['N'],xyz['E'],event['latitude'],event['longitude'],obs['station_latitude'],obs['station_longitude'])
    if r is None or t is None:raise ValueError('RT rotation failed')
    rtz=align({'R':r,'T':t,'Z':xyz['Z']})
    return rtz['R'],rtz['T'],rtz['Z'],baz


def current_measure(r,t,z,p,s):
    p_amp=base.find_p_peak(r,z,p)
    s_amp=base._s_three_component_peak(r,t,z,s)
    end=p-base.NOISE_GAP_BEFORE_P
    start=end-base.NOISE_WINDOW
    p_noise=common.vector_rms(r,z,start,end)
    s_parts=[common.rms(tr,start,end) for tr in (r,t,z)]
    s_noise=None if any(x is None for x in s_parts) else max(s_parts)
    return p_amp,s_amp,p_noise,s_noise


def praj_measure(r,t,z,p,s):
    # M2 amplitudes follow Praj's final clarification rather than his older notebook.
    pv=common.window_data(z,p-0.5,p+0.5)
    rv=common.window_data(r,s-1.0,s+1.0)
    tv=common.window_data(t,s-1.0,s+1.0)
    if any(v is None for v in (pv,rv,tv)):return None,None,None,None
    m=min(len(rv),len(tv))
    p_amp=float(np.ptp(pv))
    s_amp=float(np.max(np.hypot(rv[:m],tv[:m])))
    # The notebook's noise windows are retained, with corrected vector/peak-to-peak operators.
    # These estimators are provisional: Praj did not confirm the updated noise/QC recipe.
    pn=common.window_data(z,p-1.5,p-1.0)
    rn=common.window_data(r,s-2.0,s-1.0)
    tn=common.window_data(t,s-2.0,s-1.0)
    if any(v is None for v in (pn,rn,tn)):return p_amp,s_amp,None,None
    n=min(len(rn),len(tn))
    p_noise=float(np.ptp(pn))
    s_noise=float(np.max(np.hypot(rn[:n],tn[:n])))
    return p_amp,s_amp,p_noise,s_noise


def evaluate(result):
    if any(x is None or not np.isfinite(x) or x<=0 for x in result):return None
    ap,ass,np_,ns=result
    return dict(amp_p=ap,amp_s=ass,noise_p=np_,noise_s=ns,snr_p=ap/np_,snr_s=ass/ns,sp=ass/ap,log10_sp=float(np.log10(ass/ap)),passed=bool(ap/np_>=base.RATMIN and ass/ns>=base.RATMIN))


def process(obs,event,index,inventory):
    raw=get_traces(index,obs)
    p=UTCDateTime(obs['p_time']);s=UTCDateTime(obs['s_time'])
    if s<=p:raise ValueError('S pick not after P pick')
    r,t,z,baz=get_rtz(raw,obs,event,inventory,base.FILTER_FREQMIN,base.FILTER_FREQMAX,base.RESPONSE_PREFILT)
    m1=evaluate(current_measure(r,t,z,p,s))
    r,t,z,_=get_rtz(raw,obs,event,inventory,*M2_FILTER,M2_PREFILT)
    m2=evaluate(praj_measure(r,t,z,p,s))
    row={'event_id':event['event_id'],'network':obs['network'],'station':obs['station'],'back_azimuth':baz,'m2_qc_provisional':True}
    for label,result in [('m1',m1),('m2',m2)]:
        row[f'{label}_status']='valid' if result else 'invalid_amplitude_or_noise'
        if result:row.update({f'{label}_{k}':v for k,v in result.items()})
    if m1 and m2:
        row['delta_log10_m2_minus_m1']=m2['log10_sp']-m1['log10_sp']
        row['ratio_m2_over_m1']=m2['sp']/m1['sp']
        row['pass_both']=m1['passed'] and m2['passed']
    return row


def plot(rows,path):
    paired=[x for x in rows if x.get('m1_sp') and x.get('m2_sp')]
    if not paired:return
    a=np.array([x['m1_log10_sp'] for x in paired]);b=np.array([x['m2_log10_sp'] for x in paired]);d=b-a
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    axes[0].scatter(a,b,s=28)
    low=min(a.min(),b.min());high=max(a.max(),b.max())
    axes[0].plot([low,high],[low,high],'k--',linewidth=1)
    axes[0].set(xlabel='M1 log10(S/P)',ylabel='M2 log10(S/P)',title=f'Paired measurements (n={len(a)})')
    axes[1].hist(d,bins=min(20,max(3,int(np.sqrt(len(d))*2))),edgecolor='black')
    axes[1].axvline(0,color='black',linestyle='--')
    axes[1].set(xlabel='M2 - M1 log10(S/P)',ylabel='Station count',title='Difference distribution')
    fig.savefig(path,dpi=250)
    plt.close(fig)


# Choose a single event or all event directories.
RUN_MODE="all" # "single" or "all"
EVENT_ID="20220409.174546"
DATA_DIR=base.DATA_DIR
OUTPUT_DIR=Path("sp_method_comparison")
SAVE_EVENT_FIGURES=False # Avoid hundreds of images in all mode.


def write_csv(rows,path):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event_ids(data_dir,mode,event_id):
    if mode=="single":
        if not (data_dir/event_id).is_dir():raise FileNotFoundError(data_dir/event_id)
        return [event_id]
    if mode=="all":
        return [p.name for p in sorted(data_dir.iterdir()) if p.is_dir() and (p/"event_info").is_file()]
    raise ValueError(f"Invalid RUN_MODE: {mode}")


def process_event(event_id,data_dir,metadata,inventory):
    event_dir=data_dir/event_id
    event=base.read_event_info(event_dir/"event_info")
    event["event_id"]=event_id
    picks=base.read_pickinfo(event_dir/"pickinfo.txt")
    stn1=base.read_stn1(event_dir/"stn1.dat")
    observations,_=base.build_observations(picks,metadata,event,stn1)
    index=waveform_index(event_dir/"sac")
    rows=[]
    for obs in observations:
        if obs.get("s_time") is None:continue
        try:row=process(obs,event,index,inventory)
        except Exception as exc:
            row={"event_id":event_id,"network":obs["network"],"station":obs["station"],"error":str(exc)}
        rows.append(row)
    return rows


def summarize(rows):
    by_event={}
    for row in rows:by_event.setdefault(row["event_id"],[]).append(row)
    summaries=[]
    for event_id,items in sorted(by_event.items()):
        paired=[r for r in items if "delta_log10_m2_minus_m1" in r]
        both=[r for r in paired if r.get("pass_both")]
        d=np.array([r["delta_log10_m2_minus_m1"] for r in paired],dtype=float)
        db=np.array([r["delta_log10_m2_minus_m1"] for r in both],dtype=float)
        summaries.append({"event_id":event_id,"n_attempted":len(items),"n_paired":len(paired),"n_both_pass":len(both),"median_delta_all":float(np.median(d)) if d.size else "","median_delta_both_pass":float(np.median(db)) if db.size else ""})
    return summaries


def plot_overview(rows,path):
    paired=[r for r in rows if "delta_log10_m2_minus_m1" in r]
    if not paired:return
    a=np.array([r["m1_log10_sp"] for r in paired]);b=np.array([r["m2_log10_sp"] for r in paired]);d=b-a
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax=axes[0,0]
    hb=ax.hexbin(a,b,gridsize=50,mincnt=1,bins="log",cmap="viridis") if len(a)>500 else None
    if hb is None:ax.scatter(a,b,s=12,alpha=0.5)
    else:fig.colorbar(hb,ax=ax,label="log10(number of measurements)")
    low=float(min(a.min(),b.min()));high=float(max(a.max(),b.max()))
    ax.plot([low,high],[low,high],"k--",linewidth=1)
    ax.set(xlabel="M1 log10(S/P)",ylabel="M2 log10(S/P)",title=f"Paired measurements (n={len(a)})")
    axes[0,1].hist(d,bins=min(60,max(10,int(np.sqrt(len(d))))),edgecolor="black")
    axes[0,1].axvline(0,color="black",linestyle="--")
    axes[0,1].set(xlabel="M2 - M1 log10(S/P)",ylabel="Number of measurements",title="All paired differences")
    summaries=summarize(rows)
    values=[x["median_delta_all"] for x in summaries if x["median_delta_all"]!=""]
    axes[1,0].hist(values,bins=min(40,max(8,int(np.sqrt(len(values))))) if values else 8,edgecolor="black")
    axes[1,0].axvline(0,color="black",linestyle="--")
    axes[1,0].set(xlabel="Event median M2 - M1 log10(S/P)",ylabel="Number of events",title=f"Event-wise medians (n={len(values)})")
    counts=[x["n_paired"] for x in summaries]
    axes[1,1].hist(counts,bins=min(40,max(8,int(np.sqrt(len(counts))))) if counts else 8,edgecolor="black")
    axes[1,1].set(xlabel="Paired measurements per event",ylabel="Number of events",title="Event coverage")
    fig.savefig(path,dpi=250)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("single","all"),default=RUN_MODE)
    parser.add_argument("--event",default=EVENT_ID)
    parser.add_argument("--data-dir",type=Path,default=DATA_DIR)
    parser.add_argument("--output",type=Path,default=OUTPUT_DIR)
    args=parser.parse_args()
    ids=event_ids(args.data_dir,args.mode,args.event)
    if not ids:raise ValueError(f"No event directories with event_info in {args.data_dir}")
    metadata=base.read_korean_station_metadata(base.STATION_METADATA_FILE)
    inventory=common.load_response_inventory(base.RESPONSE_XML_FILE)
    all_rows=[];failures=[]
    for i,event_id in enumerate(ids,1):
        print(f"[{i}/{len(ids)}] {event_id}",flush=True)
        try:
            rows=process_event(event_id,args.data_dir,metadata,inventory)
            if not rows:raise ValueError("No observations with S picks")
            all_rows.extend(rows)
            if args.mode=="single" or SAVE_EVENT_FIGURES:
                out=args.output/event_id
                write_csv(rows,out/"SP_comparison_all.csv")
                plot(rows,out/"SP_comparison.png")
        except Exception as exc:
            failures.append({"event_id":event_id,"error":str(exc)})
            print(f"SKIP {event_id}: {exc}",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(all_rows,args.output/("all_events_SP_comparison.csv" if args.mode=="all" else "single_event_SP_comparison.csv"))
    write_csv(summarize(all_rows),args.output/"event_summary.csv")
    write_csv(failures,args.output/"event_errors.csv")
    if all_rows:plot_overview(all_rows,args.output/("all_events_SP_overview.png" if args.mode=="all" else "single_event_SP_overview.png"))
    print(f"Completed {len(ids)-len(failures)}/{len(ids)} events; {len(all_rows)} measurements; {len(failures)} event failures. Output: {args.output}")


if __name__=="__main__":main()
