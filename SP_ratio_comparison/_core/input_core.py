#!/usr/bin/env python3
"""
Build SKHASH input files for one Yeongdeok earthquake.

Inputs:
    test/20220409.174546/event_info
    test/20220409.174546/pickinfo.txt
    test/20220409.174546/stn1.dat
    test/20220409.174546/Korean_station_info.txt
    test/20220409.174546/KS_KG_metadata_1.0.2.xml
    test/20220409.174546/crustal_kim2011_AV.prm

Outputs:
    skhash_test/EVENT_ID/
        polarity.csv
        stations.csv
        kim2011.txt
        kim2011_model_QC.csv
        station_metadata_QC.csv
        control_file.txt
        sp_amplitude.csv       (only when USE_SP_RATIO=True)
        sp_amplitude_QC.csv    (only when USE_SP_RATIO=True)
        OUT/

SAC files are read only when USE_SP_RATIO=True. They are never modified or written.
"""

import functions as common
from pathlib import Path
from datetime import datetime
import csv
import re
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

DATA_DIR = Path("your data directory") ### must edit
EVENT_ID = "20220409.174546" ### must edit
STATION_METADATA_FILE = Path("/test/20220409.174546/Korean_station_info.txt")
#VELOCITY_MODEL_FILE = Path("crustal_kim2011_AV.prm")
OUTPUT_ROOT = Path("skhash_test")
USE_SP_RATIO = True

# S/P amplitude settings following Hardebeck & Shearer (2003).
FILTER_FREQMIN = 1.0
FILTER_FREQMAX = 15.0
P_WINDOW_MIN = 0.5
P_WINDOW_MAX = 0.5
S_WINDOW_AFTER = 2.0
NOISE_WINDOW = 1.5
NOISE_GAP_BEFORE_P = 1.0
RATMIN = 3.0

# Instrument response settings.
REMOVE_RESPONSE = True
RESPONSE_XML_FILE = Path("test/20220409.174546/KS_KG_metadata_1.0.2.xml")
RESPONSE_PREFILT = (0.5, 0.8, 18.0, 20.0)
RESPONSE_WATER_LEVEL = 60.0

# Research baseline settings.
NPOLMIN = 8
MAX_AGAP = 90
MAX_PGAP = 60
DANG = 5
NMC = 30
MAXOUT = 500
BADFRAC = 0.10
DELMAX = 200
CANGLE = 45
PROB_MAX = 0.25


# ============================================================
# UTILITIES
# ============================================================

def parse_yyyymmdd(value):
    value = value.strip()
    if not value or value.lower() in {"unknown", "none", "nan", "-"}:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def make_z_channel(channel):
    channel = channel.strip().upper()
    if len(channel) == 2:
        return channel + "Z"
    if len(channel) == 3:
        return channel[:2] + "Z"
    return channel


def parse_polarity(code):
    """IPU0 -> +1, IPD0 -> -1. Codes without U/D are excluded."""
    code = code.strip().upper()
    if "U" in code:
        return 1.0
    if "D" in code:
        return -1.0
    return None


# ============================================================
# EVENT INFO
# ============================================================

def read_event_info(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Invalid event_info: {path}")

    p = lines[1].split()
    if len(p) < 6:
        raise ValueError(f"Cannot parse event_info: {path}")

    origin = datetime.fromisoformat(f"{p[0]}T{p[1]}")
    return {
        "origin": origin,
        "origin_time": origin.isoformat(timespec="milliseconds"),
        "latitude": float(p[2]),
        "longitude": float(p[3]),
        "depth_km": float(p[4]),
        "rms": float(p[5])
    }



# ============================================================
# HYPOELLIPSE LOCATION UNCERTAINTY
# ============================================================

def read_hypoellipse_uncertainty(path):
    """
    Parse the HYPOELLIPSE line containing two horizontal standard errors (SEH)
    and one vertical standard error (SEZ), for example:

        seh = 0.55   seh = 1.65   sez = 0.82   quality = b

    SKHASH accepts one scalar horizontal hypocentral uncertainty and one
    vertical hypocentral uncertainty. Here the larger of the two HYPOELLIPSE
    horizontal principal-axis standard errors is used conservatively.
    """
    pattern = re.compile(
        r"seh\s*=\s*([0-9.+\-Ee]+).*?"
        r"seh\s*=\s*([0-9.+\-Ee]+).*?"
        r"sez\s*=\s*([0-9.+\-Ee]+).*?"
        r"quality\s*=\s*([A-Za-z0-9]+)",
        re.IGNORECASE
    )

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            seh1 = float(match.group(1))
            seh2 = float(match.group(2))
            sez = float(match.group(3))
            quality = match.group(4)

            if seh1 < 0 or seh2 < 0 or sez < 0:
                raise ValueError(f"Negative HYPOELLIPSE uncertainty in {path}")

            return {
                "seh1_km": seh1,
                "seh2_km": seh2,
                "sez_km": sez,
                "horz_uncert_km": max(seh1, seh2),
                "vert_uncert_km": sez,
                "quality": quality
            }

    raise ValueError(f"Could not find SEH/SEZ uncertainty line in {path}")

# ============================================================
# PICKINFO
# ============================================================

def read_pickinfo(path):
    """
    Supported formats:
        NET STA CHA REF P S POLARITY
        NET STA CHA REF P POLARITY

    All valid P picks are retained. P polarity may be None. S time may be None.
    This allows S/P measurements to use stations even when first-motion polarity
    is unavailable.
    """
    picks = []

    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.rstrip()
            if not raw.strip():
                continue

            p = raw.split()
            if len(p) == 7:
                network, station, channel = p[0], p[1], p[2]
                p_time, s_time, polarity_code = p[4], p[5], p[6]
            elif len(p) == 6:
                network, station, channel = p[0], p[1], p[2]
                p_time, s_time, polarity_code = p[4], None, p[5]
            else:
                print(f"WARNING: pickinfo line {lineno} skipped ({len(p)} columns): {raw}")
                continue

            if p_time.lower() == "none":
                continue
            if s_time is not None and s_time.lower() == "none":
                s_time = None

            polarity = parse_polarity(polarity_code)
            if polarity is None:
                print(f"WARNING: no usable polarity: {station} {polarity_code}")

            picks.append({
                "network": network.upper(),
                "station": station.upper(),
                "channel_family": channel.upper(),
                "channel": make_z_channel(channel),
                "p_polarity": polarity,
                "p_time": p_time,
                "s_time": s_time,
                "original_code": polarity_code
            })

    return picks


# ============================================================
# KOREAN STATION METADATA
# ============================================================

def read_korean_station_metadata(path):
    """
    Expected format:
        NETWORK STATION START END LAT LON ELEV ... ... ... CHANNEL

    Example:
        BO IZH 20010218 20991231 34.1359 129.2066 386 Unknown STS-2.5 Unknown HH

    Elevation is in meters.
    """
    records = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            p = raw.split()
            if len(p) < 7:
                continue

            network, station = p[0].upper(), p[1].upper()
            start_date, end_date = parse_yyyymmdd(p[2]), parse_yyyymmdd(p[3])

            try:
                latitude = float(p[4])
                longitude = float(p[5])
                elevation_m = float(p[6])
            except ValueError:
                continue

            channel_family = ""
            if len(p) >= 11:
                candidate = p[-1].upper()
                if candidate not in {"UNKNOWN", "NONE", "-"}:
                    channel_family = candidate

            records.append({
                "network": network,
                "station": station,
                "start_date": start_date,
                "end_date": end_date,
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elevation_m,
                "channel_family": channel_family,
                "line_number": lineno,
                "raw": raw
            })

    if not records:
        raise ValueError(f"No station records could be parsed from {path}")

    return records


def date_is_valid(record, event_date):
    start, end = record["start_date"], record["end_date"]
    if start is not None and event_date < start:
        return False
    if end is not None and event_date > end:
        return False
    return True


def channel_family_matches(metadata_channel, pick_channel):
    if not metadata_channel:
        return True
    return metadata_channel.upper()[:2] == pick_channel.upper()[:2]


def select_station_record(metadata, pick, event_date):
    """
    Match priority:
        1. station + network + date + channel
        2. station + network + date
        3. station + date + channel
        4. station + date
    """
    station = pick["station"]
    network = pick["network"]
    channel = pick["channel"]

    station_records = [r for r in metadata if r["station"] == station]
    if not station_records:
        return None, "station_not_found"

    date_records = [r for r in station_records if date_is_valid(r, event_date)]
    if not date_records:
        return None, "no_valid_installation_date"

    exact = [
        r for r in date_records
        if r["network"] == network and channel_family_matches(r["channel_family"], channel)
    ]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return exact[0], "multiple_exact"

    network_match = [r for r in date_records if r["network"] == network]
    if len(network_match) == 1:
        return network_match[0], "network_date"
    if len(network_match) > 1:
        return network_match[0], "multiple_network_date"

    channel_match = [
        r for r in date_records
        if channel_family_matches(r["channel_family"], channel)
    ]
    if len(channel_match) == 1:
        return channel_match[0], "station_date_channel"

    if len(date_records) == 1:
        return date_records[0], "station_date_only"

    return date_records[0], "ambiguous_station_date"


# ============================================================
# STN1.DAT - QC ONLY
# ============================================================

def read_stn1(path):
    stations = {}
    if not path.exists():
        return stations

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return stations

    station_lines = lines[1:] if lines[0].split()[0].isdigit() else lines

    for line in station_lines:
        p = line.split()
        if len(p) < 4:
            continue

        try:
            stations[p[0].upper()] = {
                "latitude": float(p[1]),
                "longitude": float(p[2]),
                "elevation_km": float(p[3])
            }
        except ValueError:
            continue

    return stations


# ============================================================
# KIM ET AL. (2011) VELOCITY MODEL
# ============================================================

def read_kim2011_model(path):
    """
    HYPOELLIPSE example:
        VELOC 2 1 5.63000 0.00000 1.66000

    Parsed as:
        Vp = 5.63 km/s
        depth = 0 km
        Vp/Vs = 1.66
    """
    model = []

    with open(path, "r") as f:
        for line in f:
            p = line.split()
            if not p or p[0].upper() != "VELOC" or len(p) < 6:
                continue

            try:
                vp = float(p[3])
                depth = float(p[4])
                vpvs = float(p[5])
            except ValueError:
                continue

            if vp <= 0 or vpvs <= 0:
                continue

            model.append({
                "depth_km": depth,
                "vp_km_s": vp,
                "vp_vs": vpvs,
                "vs_km_s": vp / vpvs
            })

    if not model:
        raise ValueError(f"No VELOC entries were found in {path}")

    model.sort(key=lambda x: x["depth_km"])
    return model


# ============================================================
# MATCH PICKS TO STATION METADATA
# ============================================================

def build_observations(picks, metadata, event, stn1):
    observations = []
    qc_rows = []
    event_date = event["origin"].date()

    for pick in picks:
        record, status = select_station_record(metadata, pick, event_date)

        if record is None:
            print(
                f"WARNING: station metadata not found: "
                f"{pick['network']} {pick['station']} {pick['channel']} ({status})"
            )
            qc_rows.append({
                "network": pick["network"],
                "station": pick["station"],
                "channel": pick["channel"],
                "match_status": status,
                "metadata_network": "",
                "metadata_channel": "",
                "start_date": "",
                "end_date": "",
                "latitude": "",
                "longitude": "",
                "elevation_m": "",
                "stn1_latitude": "",
                "stn1_longitude": "",
                "coordinate_difference_deg": ""
            })
            continue

        stn1_info = stn1.get(pick["station"])
        stn1_lat, stn1_lon, coord_diff = "", "", ""

        if stn1_info is not None:
            stn1_lat = stn1_info["latitude"]
            stn1_lon = stn1_info["longitude"]
            dlat = record["latitude"] - stn1_lat
            dlon = record["longitude"] - stn1_lon
            coord_diff = (dlat ** 2 + dlon ** 2) ** 0.5

            if coord_diff > 0.01:
                print(
                    f"WARNING: station coordinate difference >0.01 deg: "
                    f"{pick['station']} {coord_diff:.5f}"
                )

        observations.append({
            **pick,
            "station_latitude": record["latitude"],
            "station_longitude": record["longitude"],
            "station_elevation_m": record["elevation_m"],
            "metadata_network": record["network"],
            "metadata_channel": record["channel_family"],
            "metadata_start": record["start_date"],
            "metadata_end": record["end_date"],
            "match_status": status
        })

        qc_rows.append({
            "network": pick["network"],
            "station": pick["station"],
            "channel": pick["channel"],
            "match_status": status,
            "metadata_network": record["network"],
            "metadata_channel": record["channel_family"],
            "start_date": record["start_date"] or "",
            "end_date": record["end_date"] or "",
            "latitude": record["latitude"],
            "longitude": record["longitude"],
            "elevation_m": record["elevation_m"],
            "stn1_latitude": stn1_lat,
            "stn1_longitude": stn1_lon,
            "coordinate_difference_deg": coord_diff
        })

    return observations, qc_rows


# ============================================================
# WRITE SKHASH INPUT FILES
# ============================================================

def write_polarity_csv(path, observations, event, uncertainty):
    fields = [
        "event_id", "network", "station", "location", "channel",
        "p_polarity", "origin_latitude", "origin_longitude", "origin_depth_km",
        "horz_uncert_km", "vert_uncert_km"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for obs in observations:
            if obs["p_polarity"] is None:
                continue
            writer.writerow({
                "event_id": EVENT_ID,
                "network": obs["network"],
                "station": obs["station"],
                "location": "--",
                "channel": obs["channel"],
                "p_polarity": f"{obs['p_polarity']:.1f}",
                "origin_latitude": f"{event['latitude']:.6f}",
                "origin_longitude": f"{event['longitude']:.6f}",
                "origin_depth_km": f"{event['depth_km']:.3f}",
                "horz_uncert_km": f"{uncertainty['horz_uncert_km']:.3f}",
                "vert_uncert_km": f"{uncertainty['vert_uncert_km']:.3f}"
            })


def write_stations_csv(path, observations):
    fields = ["network", "station", "location", "channel", "latitude", "longitude", "elevation"]
    unique = {}

    for obs in observations:
        key = (obs["network"], obs["station"], obs["channel"])
        unique[key] = obs

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for key in sorted(unique):
            obs = unique[key]
            writer.writerow({
                "network": obs["network"],
                "station": obs["station"],
                "location": "--",
                "channel": obs["channel"],
                "latitude": f"{obs['station_latitude']:.6f}",
                "longitude": f"{obs['station_longitude']:.6f}",
                "elevation": f"{obs['station_elevation_m']:.1f}"
            })


def write_velocity_model(path, model):
    with open(path, "w") as f:
        for layer in model:
            f.write(f"{layer['depth_km']:.3f},{layer['vp_km_s']:.4f}\n")


def write_velocity_qc(path, model):
    fields = ["depth_km", "vp_km_s", "vp_vs", "vs_km_s"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(model)


def write_station_qc(path, rows):
    fields = [
        "network", "station", "channel", "match_status",
        "metadata_network", "metadata_channel", "start_date", "end_date",
        "latitude", "longitude", "elevation_m",
        "stn1_latitude", "stn1_longitude", "coordinate_difference_deg"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# S/P AMPLITUDE MEASUREMENTS
# ============================================================

_trace_id=common.trace_id

load_response_inventory=common.load_response_inventory

def _prepare_trace(tr,inventory=None):
    return common.prepare_trace(tr,inventory,FILTER_FREQMIN,FILTER_FREQMAX,RESPONSE_PREFILT,RESPONSE_WATER_LEVEL,REMOVE_RESPONSE,adapt_nyquist_10=True)

_window_data=common.window_data

_rms=common.rms

_vector_rms=common.vector_rms

def find_p_peak(r,z,ptime):
    rv=_window_data(r,ptime-P_WINDOW_MIN,ptime+P_WINDOW_MAX)
    zv=_window_data(z,ptime-P_WINDOW_MIN,ptime+P_WINDOW_MAX)

    if rv is None or zv is None:
        return None
    m=min(len(rv),len(zv))

    if m<2:
        return None
    amp=np.sqrt(rv[:m]**2 + zv[:m]**2)

    return float(np.max(amp))

def _s_three_component_peak(r,t,z,stime):
    rv=_window_data(r,stime,stime+S_WINDOW_AFTER)
    tv=_window_data(t,stime,stime+S_WINDOW_AFTER)
    zv=_window_data(z,stime,stime+S_WINDOW_AFTER)
    if rv is None or tv is None or zv is None:
        return None
    return float(max(np.max(np.abs(rv)),np.max(np.abs(tv)),np.max(np.abs(zv))))

_rotate_ne_to_rt=common.rotate_ne_to_rt

def build_sp_amplitudes(event_dir,observations,event,uncertainty):
    try:
        from obspy import read,UTCDateTime
    except ImportError as exc:
        raise ImportError("USE_SP_RATIO=True requires ObsPy and NumPy.") from exc
    inventory=load_response_inventory(RESPONSE_XML_FILE) if REMOVE_RESPONSE else None
    sac_dir=event_dir/"sac"
    if not sac_dir.exists():
        raise FileNotFoundError(f"SAC directory not found: {sac_dir}")
    traces={}
    for sac_path in sorted(sac_dir.iterdir()):
        if not sac_path.is_file():
            continue
        try:
            st=read(str(sac_path),headonly=False)
        except Exception:
            continue
        for tr in st:
            tr=common.normalize_trace_channel(tr)
            net,sta,cha=_trace_id(tr)
            if not sta or len(cha)<3:
                continue
            try:
                tr=_prepare_trace(tr,inventory)
            except Exception as exc:
                print(f"WARNING: S/P trace skipped {net}.{sta}.{cha}: {exc}")
                continue
            traces[(net,sta,cha)]=tr
    rows=[]
    qc_rows=[]
    for obs in observations:
        if not obs.get("s_time"):
            qc_rows.append({"network":obs["network"],"station":obs["station"],"status":"no_s_pick"})
            continue
        net=obs["network"]
        sta=obs["station"]
        fam=obs["channel_family"][:2]
        station_traces={cha:tr for (n,s,cha),tr in traces.items() if n==net and s==sta and cha[:2]==fam}
        if not station_traces:
            station_traces={cha:tr for (n,s,cha),tr in traces.items() if s==sta and cha[:2]==fam}
        z=next((tr for cha,tr in station_traces.items() if cha.endswith("Z")),None)
        n=next((tr for cha,tr in station_traces.items() if cha.endswith("N")),None)
        e=next((tr for cha,tr in station_traces.items() if cha.endswith("E")),None)
        if z is None or n is None or e is None:
            qc_rows.append({"network":net,"station":sta,"status":"missing_ENZ"})
            continue
        ptime=UTCDateTime(obs["p_time"])
        stime=UTCDateTime(obs["s_time"])
        if stime<=ptime:
            qc_rows.append({"network":net,"station":sta,"status":"s_not_after_p"})
            continue
        start=max(z.stats.starttime,n.stats.starttime,e.stats.starttime)
        end=min(z.stats.endtime,n.stats.endtime,e.stats.endtime)
        if end<=start:
            qc_rows.append({"network":net,"station":sta,"status":"no_common_ENZ"})
            continue
        z=z.copy().trim(start,end,pad=False)
        n=n.copy().trim(start,end,pad=False)
        e=e.copy().trim(start,end,pad=False)
        m=min(z.stats.npts,n.stats.npts,e.stats.npts)
        if m<2:
            qc_rows.append({"network":net,"station":sta,"status":"too_short"})
            continue
        z.data=np.asarray(z.data[:m],dtype=float)
        n.data=np.asarray(n.data[:m],dtype=float)
        e.data=np.asarray(e.data[:m],dtype=float)
        r,t,back_azimuth=_rotate_ne_to_rt(n,e,event["latitude"],event["longitude"],obs["station_latitude"],obs["station_longitude"])
        if r is None or t is None:
            qc_rows.append({"network":net,"station":sta,"status":"rotation_failed"})
            continue
        start=max(r.stats.starttime,t.stats.starttime,z.stats.starttime)
        end=min(r.stats.endtime,t.stats.endtime,z.stats.endtime)
        r.trim(start,end,pad=False)
        t.trim(start,end,pad=False)
        z.trim(start,end,pad=False)
        m=min(r.stats.npts,t.stats.npts,z.stats.npts)
        if m<2:
            qc_rows.append({"network":net,"station":sta,"status":"too_short_after_rotation"})
            continue
        r.data=np.asarray(r.data[:m],dtype=float)
        t.data=np.asarray(t.data[:m],dtype=float)
        z.data=np.asarray(z.data[:m],dtype=float)
        noise_end=ptime-NOISE_GAP_BEFORE_P
        noise_start=noise_end-NOISE_WINDOW
        amp_p=find_p_peak(r,z,ptime)
        amp_s=_s_three_component_peak(r,t,z,stime)
        noise_p=_vector_rms(r,z,noise_start,noise_end)
        noise_r=_rms(r,noise_start,noise_end)
        noise_t=_rms(t,noise_start,noise_end)
        noise_z=_rms(z,noise_start,noise_end)
        noise_s=None if any(v is None for v in (noise_r,noise_t,noise_z)) else max(noise_r,noise_t,noise_z)
        vals=[noise_p,noise_s,amp_p,amp_s]
        if any(v is None or not np.isfinite(v) or v<=0 for v in vals):
            qc_rows.append({"network":net,"station":sta,"status":"invalid_amplitude","back_azimuth":back_azimuth})
            continue
        snr_p=amp_p/noise_p
        snr_s=amp_s/noise_s
        sp_ratio=amp_s/amp_p
        log10_sp_ratio=np.log10(sp_ratio)
        status="pass_ratmin" if snr_p>=RATMIN and snr_s>=RATMIN else "below_ratmin"
        if status=="pass_ratmin":
            rows.append({"event_id":EVENT_ID,"network":net,"station":sta,"location":"--","channel":obs["channel"],"noise_p":noise_p,"noise_s":noise_s,"amp_p":amp_p,"amp_s":amp_s,"origin_latitude":event["latitude"],"origin_longitude":event["longitude"],"origin_depth_km":event["depth_km"],"horz_uncert_km":uncertainty["horz_uncert_km"],"vert_uncert_km":uncertainty["vert_uncert_km"]})
        qc_rows.append({"network":net,"station":sta,"status":status,"back_azimuth":back_azimuth,"amp_p":amp_p,"amp_s":amp_s,"noise_p":noise_p,"noise_s":noise_s,"snr_p":snr_p,"snr_s":snr_s,"sp_ratio_linear":sp_ratio,"log10_sp_ratio":log10_sp_ratio})
    return rows,qc_rows

def write_sp_csv(path,rows):
    fields=["event_id","network","station","location","channel","noise_p","noise_s","amp_p","amp_s","origin_latitude","origin_longitude","origin_depth_km","horz_uncert_km","vert_uncert_km"]
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_sp_qc(path,rows):
    fields=["network","station","status","back_azimuth","amp_p","amp_s","noise_p","noise_s","snr_p","snr_s","sp_ratio_linear","log10_sp_ratio"]
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# CONTROL FILE
# ============================================================

def write_control_file(path, polarity_file, station_file, velocity_file, output_dir, amp_file=None):
    # Build the optional S/P block outside the main f-string.
    # This is compatible with Python 3.8, where backslashes are not allowed
    # inside an f-string expression.
    if amp_file is not None:
        amp_block = "$ampfile\n{}\n\n$ratmin\n{}\n\n".format(amp_file, RATMIN)
    else:
        amp_block = ""

    content = f"""# SKHASH control file
# Yeongdeok one-event smoke test
# Event: {EVENT_ID}

$input_format
skhash

$fpfile
{polarity_file}

$stfile
{station_file}

{amp_block}$outfile1
{output_dir / "preferred_mechanisms.csv"}

$outfile2
{output_dir / "acceptable_mechanisms.csv"}

$vmodel_paths
{velocity_file}

$npolmin
{NPOLMIN}

$max_agap
{MAX_AGAP}

$max_pgap
{MAX_PGAP}

$dang
{DANG}

$nmc
{NMC}

$maxout
{MAXOUT}

$badfrac
{BADFRAC}

$delmax
{DELMAX}

$cangle
{CANGLE}

$prob_max
{PROB_MAX}
"""
    with open(path, "w") as f:
        f.write(content)


# ============================================================
# MAIN
# ============================================================

def main():
    event_dir = DATA_DIR / EVENT_ID
    event_info_file = event_dir / "event_info"
    pickinfo_file = event_dir / "pickinfo.txt"
    stn1_file = event_dir / "stn1.dat"
    hypoe_file = event_dir / "hypoellipse_1" / "hypoe.out"
    velocity_model_file = event_dir / "hypoellipse_1" / "crustal_kim2011_AV.prm"
    
    required = [
        event_info_file,
        pickinfo_file,
        STATION_METADATA_FILE,
        hypoe_file,
        velocity_model_file
    ]
    if USE_SP_RATIO and REMOVE_RESPONSE:
        required.append(RESPONSE_XML_FILE)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    output_dir = OUTPUT_ROOT / EVENT_ID
    skhash_out = output_dir / "OUT"
    output_dir.mkdir(parents=True, exist_ok=True)
    skhash_out.mkdir(parents=True, exist_ok=True)

    print(f"\nReading event: {EVENT_ID}")

    event = read_event_info(event_info_file)
    picks = read_pickinfo(pickinfo_file)
    metadata = read_korean_station_metadata(STATION_METADATA_FILE)
    stn1 = read_stn1(stn1_file)
    uncertainty = read_hypoellipse_uncertainty(hypoe_file)
    velocity_model = read_kim2011_model(velocity_model_file)

    print(f"P picks read: {len(picks)}")
    print(f"Usable P polarities: {sum(p['p_polarity'] is not None for p in picks)}")
    print(f"Station metadata records: {len(metadata)}")

    observations, qc_rows = build_observations(picks, metadata, event, stn1)

    print(f"Matched SKHASH observations: {len(observations)}")

    polarity_file = output_dir / "polarity.csv"
    station_file = output_dir / "stations.csv"
    velocity_file = output_dir / "kim2011.txt"
    velocity_qc_file = output_dir / "kim2011_model_QC.csv"
    station_qc_file = output_dir / "station_metadata_QC.csv"
    control_file = output_dir / "control_file.txt"
    amp_file = output_dir / "sp_amplitude.csv"
    amp_qc_file = output_dir / "sp_amplitude_QC.csv"

    write_polarity_csv(polarity_file, observations, event, uncertainty)
    write_stations_csv(station_file, observations)
    write_velocity_model(velocity_file, velocity_model)
    write_velocity_qc(velocity_qc_file, velocity_model)
    write_station_qc(station_qc_file, qc_rows)

    sp_rows = []
    sp_qc_rows = []
    if USE_SP_RATIO:
        sp_rows, sp_qc_rows = build_sp_amplitudes(event_dir, observations, event, uncertainty)
        write_sp_csv(amp_file, sp_rows)
        write_sp_qc(amp_qc_file, sp_qc_rows)
        if not sp_rows:
            raise ValueError("USE_SP_RATIO=True but no usable S/P amplitude measurements were produced.")
        write_control_file(control_file, polarity_file, station_file, velocity_file, skhash_out, amp_file)
    else:
        write_control_file(control_file, polarity_file, station_file, velocity_file, skhash_out, None)

    polarity_obs = [x for x in observations if x["p_polarity"] is not None]
    n_up = sum(x["p_polarity"] > 0 for x in polarity_obs)
    n_down = sum(x["p_polarity"] < 0 for x in polarity_obs)
    unmatched = len(picks) - len(observations)

    print("\n" + "=" * 60)
    print("SKHASH INPUT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Event ID:      {EVENT_ID}")
    print(f"Origin:        {event['origin_time']}")
    print(
        f"Hypocenter:    {event['latitude']:.5f}, "
        f"{event['longitude']:.5f}, {event['depth_km']:.2f} km"
    )
    print(f"RMS:           {event['rms']:.4f} s")
    print(f"HYPOELLIPSE:   SEH1={uncertainty['seh1_km']:.3f} km, "
          f"SEH2={uncertainty['seh2_km']:.3f} km, SEZ={uncertainty['sez_km']:.3f} km, "
          f"quality={uncertainty['quality']}")
    print(f"SKHASH uncert: H={uncertainty['horz_uncert_km']:.3f} km, "
          f"V={uncertainty['vert_uncert_km']:.3f} km")
    print(f"\nInput picks:   {len(picks)}")
    print(f"Matched picks: {len(observations)}")
    print(f"Unmatched:     {unmatched}")
    print(f"Up (+):        {n_up}")
    print(f"Down (-):      {n_down}")
    print(f"S/P ratios:    {'ON' if USE_SP_RATIO else 'OFF'}")
    if USE_SP_RATIO:
        n_sp_pass = sum(r.get("status") == "pass_ratmin" for r in sp_qc_rows)
        print(f"S/P measured:  {len(sp_rows)}")
        print(f"S/P SNR pass:  {n_sp_pass} (RATMIN={RATMIN:g})")
        print(f"S/P filter:    {FILTER_FREQMIN:g}-{FILTER_FREQMAX:g} Hz")
        print(f"Response rm:   {'ON' if REMOVE_RESPONSE else 'OFF'}")
        if REMOVE_RESPONSE:
            print(f"StationXML:    {RESPONSE_XML_FILE}")

    print("\nVelocity model:")
    for x in velocity_model:
        print(
            f"  depth={x['depth_km']:6.2f} km  "
            f"Vp={x['vp_km_s']:.3f} km/s  "
            f"Vp/Vs={x['vp_vs']:.3f}"
        )

    print(f"\nOutput: {output_dir}")

    if len(polarity_obs) < NPOLMIN:
        print(f"WARNING: fewer than {NPOLMIN} matched polarities.")
    else:
        print("Polarity count: PASS")

    if unmatched > 0:
        print("CHECK station_metadata_QC.csv before running SKHASH.")

    print("\nFiles created:")
    print(f"  {polarity_file}")
    print(f"  {station_file}")
    print(f"  {velocity_file}")
    print(f"  {velocity_qc_file}")
    print(f"  {station_qc_file}")
    print(f"  {control_file}")

    print("\nAfter checking station_metadata_QC.csv, run:")
    print(f"SKHASH {control_file}\n")


if __name__ == "__main__":
    main()

