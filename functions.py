#!/usr/bin/env python3
"""Shared waveform processing and geometry helpers for Yeongdeok analysis."""
from pathlib import Path
import numpy as np


def normalize_channel(channel):
    channel=str(channel).strip().upper()
    if len(channel)>3 and channel[:3][-1] in "NEZ" and channel[3:].isdigit():
        return channel[:3]
    return channel


def normalize_trace_channel(tr):
    tr=tr.copy()
    tr.stats.channel=normalize_channel(tr.stats.channel)
    return tr


def trace_id(tr):
    return (str(getattr(tr.stats,"network","")).upper(),str(getattr(tr.stats,"station","")).upper(),normalize_channel(getattr(tr.stats,"channel","")))


def load_response_inventory(path):
    from obspy import read_inventory
    path=Path(path)
    if not path.exists():raise FileNotFoundError(f"StationXML not found: {path}")
    inventory=read_inventory(str(path),format="STATIONXML")
    print(f"StationXML loaded: {path}")
    print(f"Networks in StationXML: {', '.join(net.code for net in inventory)}")
    return inventory


def prepare_trace(tr,inventory,freqmin,freqmax,response_prefilt,water_level,remove_response=True,adapt_nyquist_10=False):
    tr=normalize_trace_channel(tr)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05,type="cosine")
    nyquist=0.5*float(tr.stats.sampling_rate)
    if adapt_nyquist_10 and np.isclose(nyquist,10.0,rtol=0,atol=0.01):
        freqmax=9.0
        response_prefilt=(0.5,0.8,9.2,9.8)
    if freqmax>=nyquist:raise ValueError(f"{tr.id}: bandpass upper frequency >= Nyquist ({nyquist:g} Hz)")
    if remove_response:
        if inventory is None:raise ValueError("Response removal requested without inventory")
        if response_prefilt[3]>=nyquist:raise ValueError(f"{tr.id}: response pre-filter upper corner >= Nyquist ({nyquist:g} Hz)")
        tr.remove_response(inventory=inventory,output="DISP",pre_filt=response_prefilt,water_level=water_level,zero_mean=False,taper=False)
    tr.filter("bandpass",freqmin=freqmin,freqmax=freqmax,corners=4,zerophase=True)
    return tr


def window_data(tr,start,end):
    cut=tr.copy().trim(starttime=start,endtime=end,pad=False)
    if cut.stats.npts<2:return None
    return np.asarray(cut.data,dtype=float)


def rms(tr,start,end):
    x=window_data(tr,start,end)
    if x is None or x.size==0:return None
    return float(np.sqrt(np.mean(x*x)))


def vector_rms(tr1,tr2,start,end):
    x1=window_data(tr1,start,end)
    x2=window_data(tr2,start,end)
    if x1 is None or x2 is None:return None
    m=min(len(x1),len(x2))
    if m<2:return None
    return float(np.sqrt(np.mean(x1[:m]**2+x2[:m]**2)))


def rotate_ne_to_rt(n,e,event_lat,event_lon,station_lat,station_lon):
    from obspy.geodetics import gps2dist_azimuth
    from obspy.signal.rotate import rotate_ne_rt as obspy_rotate
    _,_,baz=gps2dist_azimuth(event_lat,event_lon,station_lat,station_lon)
    start=max(n.stats.starttime,e.stats.starttime)
    end=min(n.stats.endtime,e.stats.endtime)
    if end<=start:return None,None,None
    nc=n.copy().trim(start,end,pad=False)
    ec=e.copy().trim(start,end,pad=False)
    m=min(nc.stats.npts,ec.stats.npts)
    if m<2:return None,None,None
    nc.data=np.asarray(nc.data[:m],dtype=float)
    ec.data=np.asarray(ec.data[:m],dtype=float)
    rdata,tdata=obspy_rotate(nc.data,ec.data,baz)
    r=nc.copy();t=ec.copy()
    r.data=np.asarray(rdata,dtype=float)
    t.data=np.asarray(tdata,dtype=float)
    return r,t,float(baz)


def align_three(a,b,c):
    start=max(a.stats.starttime,b.stats.starttime,c.stats.starttime)
    end=min(a.stats.endtime,b.stats.endtime,c.stats.endtime)
    if end<=start:return None,None,None
    traces=[tr.copy().trim(start,end,pad=False) for tr in (a,b,c)]
    m=min(tr.stats.npts for tr in traces)
    if m<2:return None,None,None
    for tr in traces:tr.data=np.asarray(tr.data[:m],dtype=float)
    return tuple(traces)


def run_input(event_id,data_dir=None,output_root=None):
    from _core import input_core as mod
    mod.EVENT_ID=str(event_id)
    if data_dir is not None:mod.DATA_DIR=Path(data_dir)
    if output_root is not None:mod.OUTPUT_ROOT=Path(output_root)
    mod.main()

def run_qc(event_id,data_dir=None,output_dir=None,stations=None,max_stations=None):
    from _core import qc_core as mod
    mod.EVENT_ID=str(event_id)
    if data_dir is not None:mod.DATA_DIR=Path(data_dir)
    if output_dir is not None:mod.OUTPUT_DIR=Path(output_dir)
    mod.STATIONS_TO_PLOT=stations
    mod.MAX_STATIONS=max_stations
    mod.main()

def run_figure(event_id,output_root=None,topography=None,faults=None):
    from _core import figure_core as mod
    import sys
    root=Path(output_root) if output_root is not None else Path('skhash_test')
    d=root/str(event_id)
    preferred=d/'OUT'/'preferred_mechanisms.csv'
    acceptable=d/'OUT'/'acceptable_mechanisms.csv'
    if not preferred.exists() or not acceptable.exists():
        raise FileNotFoundError(f'SKHASH output not found for {event_id}. Run SKHASH before plotting.')
    args=['SKHASH_Figure.py','--preferred',str(preferred),'--acceptable',str(acceptable),'--polarity',str(d/'polarity.csv'),'--stations',str(d/'stations.csv'),'--output',str(d/f'{event_id}_SKHASH_Figure.png')]
    if (d/'sp_amplitude.csv').exists():args+=['--sp-amplitude',str(d/'sp_amplitude.csv')]
    if topography is not None:args+=['--topography',str(topography)]
    if faults is not None:args+=['--faults',str(faults)]
    saved=sys.argv
    try:
        sys.argv=args
        mod.main()
    finally:sys.argv=saved


def run_travel_time_qc(data_dir,output_dir,run_mode="all",event_id=None,fit_mode="segmented",distance_edges=(0,50,100,150,200),min_points=30):
    from _core.travel_time_core import run
    return run(data_dir,output_dir,run_mode,event_id,fit_mode,distance_edges,min_points=min_points)
