# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:37:35 2026

@author: vijit
"""

# General Helper Functions
import util_functions

# Import python ABF library
import pyabf
from uuid import uuid4
import numpy as np
import os

# Plotter
import matplotlib.pyplot as plt

# Dataframe for opening CSVs
import pandas as pd

# IPFX Analyze
import warnings
import ipfx.subthresh_features as subf
import ipfx.time_series_utils as tsu
from ipfx.sweep import Sweep, SweepSet
from ipfx.feature_extractor import SpikeTrainFeatureExtractor, SpikeFeatureExtractor
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis

# -----------------------------
# Paths and constants
# -----------------------------

current_dir = os.getcwd()

input_abf_root = os.path.join(current_dir, "Human ZD")
patient_data_path = os.path.join(current_dir, "patientData", "patientData.csv")

clamp_modes = ["VoltageClamp", "CurrentClamp", ""]
V_CLAMP_MODE = 0
I_CLAMP_MODE = 1
I0_CLAMP_MODE = 2

species = "human"

# Keep only sweeps with stimulation amplitude <= 350 pA
MAX_STIM_AMP_PA = 350.0

# Tolerance between baseline and RMP for when RMP correction is needed
BASELINE_TOLERANCE= 10.0 # mV

## Dates Missing from patient files ##
dates_missing = set()

data_dict = {}
meta_dict = {}

def get_metadata(abf):
    metadata = {
        "file": {
            "version": abf.abfVersion,
            "versionStr": abf.abfVersionString,
            "protocol": abf.protocol,
            "creator": abf.creator,
            "ID": abf.abfID,
            'file_path': abf.abfFilePath,
            "comment": abf.abfFileComment,
            "guid": abf.fileGUID,
            "datetime": abf.abfDateTime,
            "datetimestr": abf.abfDateTimeString,
        },
        
        "acquisition": {
            "sample_rate": abf.dataRate,
            "sweeps": abf.sweepCount,
            "points_per_sweep": abf.sweepPointCount,
            "sweep_length_s": abf.sweepLengthSec,
        },
        
        "channels": {
            "count": abf.channelCount,
            "names": abf.adcNames,
            "units": abf.adcUnits,
            # "gains": abf.adcGains,
        },
        
        "clamp": {
            # "mode": abf.clampMode,
            # "mode_str": abf.clampModeString,
        },
    }
    
    return metadata

def estimate_tau_relaxed_ipfx_snr(
    sweeps,
    stim_amps,
    start_time,
    end_time,
    subthresh_min_amp,
    baseline_interval=0.008,
    tau_frac=0.1,
    min_snr=5.0,
):
    """
    Recompute tau using the same IPFX exponential fit logic,
    but with a relaxed SNR threshold.

    This is useful when IPFX LongSquareAnalysis returns tau=NaN
    because subf.time_constant() defaults to min_snr=20.
    """

    candidates = []

    for sweep, amp in zip(sweeps, stim_amps):
        if amp is None:
            continue

        amp = float(amp)

        # Match IPFX convention:
        # stim_amp < 0 and stim_amp > subthresh_min_amp
        if amp < 0 and amp > subthresh_min_amp:
            candidates.append((sweep, amp))

    if len(candidates) == 0:
        return np.nan

    # IPFX uses median peak hyperpolarization time as max_fit_end
    peak_times = []

    for sweep, amp in candidates:
        try:
            _, peak_index = subf.voltage_deflection(
                sweep.t,
                sweep.v,
                sweep.i,
                start_time,
                end_time,
                "min",
            )
            peak_times.append(sweep.t[peak_index])
        except Exception:
            pass

    if len(peak_times) == 0:
        return np.nan

    median_peak_time = float(np.nanmedian(peak_times))

    tau_values = []

    for sweep, amp in candidates:
        try:
            tau = subf.time_constant(
                sweep.t,
                sweep.v,
                sweep.i,
                start_time,
                end_time,
                max_fit_end=median_peak_time,
                frac=tau_frac,
                baseline_interval=baseline_interval,
                min_snr=min_snr,
            )

            if tau is not None and np.isfinite(tau):
                tau_values.append(float(tau))

        except Exception:
            continue

    if len(tau_values) == 0:
        return np.nan

    return float(np.nanmean(tau_values))

def analyze_abf(dirpath, filename, species = "human"):
    print(f"\Analyzing {filename}")
    recordings = []
    metadata = {}
    
    ## Make file directory strings to access ABF files and output folders for NWB files
    abf_path = os.path.join(dirpath, filename)
    abf = pyabf.ABF(abf_path)
    
    # Get ABF metadata
    metadata = get_metadata(abf)

    try:
        conditions = abf._tagSection.sComment
    except AttributeError:
        conditions = list(zip(abf.tagTimesSec, abf.tagComments))

    if len(conditions) < 1:
        conditions = 'No Description'
        
    else:
        conditions = conditions[0]
    
    ## if species is human then extract data from the patient_data.csv
    if species == "human":
        patient_date_db, csv_missing, tissue_type = util_functions.patient_data_parse(
            metadata['file']['datetimestr'],
            patient_data_path,
            metadata['file']['protocol']
        )
    
        if csv_missing:
            dates_missing.add(csv_missing)
    
        has_patient_info = not patient_date_db.empty

        structure, sex, age, tissue_type, tumour = util_functions.patient_data_extract(
            patient_date_db
        )
        
        model = tissue_type
        condition = tumour
        
        description = (
            f"{conditions}; "
            f"tissue type: {tissue_type}; "
            f"tumour status: {tumour}; "
            f"structure: {structure}"
        )
    ## for mice data what to do
    else:
        mouse_meta = util_functions.parse_mouse_metadata_from_path(dirpath, input_abf_root)

        has_patient_info = True
    
        structure = mouse_meta["structure"]
        condition = mouse_meta["condition"]
        model = mouse_meta["model"]
        cell_type = mouse_meta["cell_type"]
        tissue_type = mouse_meta["tissue_type"]
    
        sex = "unknown"
        age = "unknown"
        
        description = f"{condition}; tissue type: {tissue_type}; structure: {structure}"

    ## Finding Response details
    print("adcNames:", getattr(abf, "adcNames", None))
    print("adcUnits:", getattr(abf, "adcUnits", None))
    channel_info = util_functions.find_file_level_channels(abf)

    if channel_info is None:
        print(f"{filename}: could not determine response channel, skipping") 
        return None
    
    response_channel = channel_info["response_channel"]
    stim_channel = channel_info["stim_channel"]

    print(f"{filename}: response_channel={response_channel}, stim_channel={stim_channel}")
    
    # Go through the sweeps to find time frames where long square sweep may be available
    fallback_stim_start, fallback_stim_end = util_functions.infer_long_square_window_from_abf(abf)
    

for dirpath, dirnames, filenames in os.walk(input_abf_root):
    folder_dict = {}
    
    # Folder Name Creation 
    folder_name = os.path.relpath(dirpath, input_abf_root)
    if folder_name == ".":
        folder_name_safe = "root"
    else:
        folder_name_safe = folder_name.replace("\\", "__").replace("/", "__")

    for filename in filenames:
        if not filename.lower().endswith(".abf"):
            continue 
    
        try:
            # abf = pyabf.ABF(dirpath + '\\' + filename)

            # print("protocol name:", abf.protocol)

            # print("\nEpoch/DAC-related attributes:")
            # for attr in dir(abf):
            #     if "epoch" in attr.lower() or "dac" in attr.lower() or "protocol" in attr.lower():
            #         print(attr)
                    
            # for section_name in ["_epochPerDacSection", "_dacSection", "_protocolSection"]:
            #     section = getattr(abf, section_name, None)
            
            #     print("\n", section_name)
            #     print(section)
            
            #     if section is not None:
            #         for key, value in vars(section).items():
            #             print(key, "=", value)        
            result = analyze_abf(dirpath, filename, species)
        
            # if result is None:
            #     continue
            
            
            print("Success!")
        
        except Exception as e:
            print(f"Failed to analyze {filename}: {e}")
            
        abf = pyabf.ABF(dirpath + '\\' + filename)
    
        valid_sweeps = []
        start_times = []
        end_times = []
        stim_amps = []
        min_amp = None
        positive_stim_count = 0
        metadata = {}
        
        metadata = get_metadata(abf)
        
        try:
            conditions = abf._tagSection.sComment
        except AttributeError:
            conditions = list(zip(abf.tagTimesSec, abf.tagComments))
    
        if len(conditions) < 1:
            conditions = 'No Description'
        else:
            conditions = conditions[0]
            
        layer, layer_source = util_functions.extract_layer_from_abf_tags(abf)
        print(f"{filename}: layer={layer}, layer_source={layer_source}")
        
        true_rmp_mV, rmp_source = util_functions.extract_rmp_from_abf_tags(abf)
        print(f"{filename}: true_rmp_mV={true_rmp_mV}, rmp_source={rmp_source}")
    
        ## if species is human then extract data from the patient_data.csv
        if species == "human":
            patient_date_db, csv_missing, tissue_type = util_functions.patient_data_parse(
                metadata['file']['datetimestr'],
                patient_data_path,
                metadata['file']['protocol']
            )
        
            if csv_missing:
                dates_missing.add(csv_missing)
        
            has_patient_info = not patient_date_db.empty
            structure, sex, age, tissue_type, tumour = util_functions.patient_data_extract(
                patient_date_db
            )
            
            model = tissue_type
            
            description = (
                f"{conditions}; "
                f"tissue type: {tissue_type}; "
                f"tumour status: {tumour}; "
                f"structure: {structure}"
            )
                        
        ## for mice data what to do
        else:
            mouse_meta = util_functions.parse_mouse_metadata_from_path(dirpath, input_abf_root)

            has_patient_info = True
        
            structure = mouse_meta["structure"]
            condition = mouse_meta["condition"]
            model = mouse_meta["model"]
            cell_type = mouse_meta["cell_type"]
            tissue_type = mouse_meta["tissue_type"]
        
            sex = "unknown"
            age = "unknown"
            
            description = f"{condition}; tissue type: {tissue_type}; structure: {structure}"
        datetimestring = abf.abfDateTimeString
    
        # patient_date_db, csv_missing, tissue_type = util_functions.patient_data_parse(
        #     abf.abfDateTimeString, patient_data_path, conditions
        # )
    
        # # skipPatient = not patient_date_db.empty
        # dates_missing.add(csv_missing)
        
        # structure, sex, age = util_functions.patient_data_extract(patient_date_db)
        
        ## Finding Response details
        print("adcNames:", getattr(abf, "adcNames", None))
        print("adcUnits:", getattr(abf, "adcUnits", None))
        channel_info = util_functions.find_file_level_channels(abf)
    
        if channel_info is None:
            print(f"{filename}: could not determine response channel, skipping")
            continue
    
        response_channel = channel_info["response_channel"]
        stim_channel = channel_info["stim_channel"]
        
        # # If stim channel was not detected, use the pA channel as Current_in
        # if stim_channel is None:
        #     for ch, unit in enumerate(abf.adcUnits):
        #         name = str(abf.adcNames[ch]).lower()
        #         unit = str(unit).lower()
        
        #         if unit == "pa" or "current" in name:
        #             stim_channel = ch
        #             print(f"{filename}: manually assigned stim_channel={stim_channel} from adcNames/adcUnits")
        #             break
        # Always prefer the real Current_in pA channel as the fallback stim channel
        current_in_channel = None
        
        # Strong match: channel named Current_in with pA units
        for ch, name in enumerate(abf.adcNames):
            name_lower = str(name).lower()
            unit_lower = str(abf.adcUnits[ch]).lower()
        
            if "current_in" in name_lower and unit_lower == "pa":
                current_in_channel = ch
                break
        
        # Fallback: any pA channel
        if current_in_channel is None:
            for ch, unit in enumerate(abf.adcUnits):
                unit_lower = str(unit).lower()
        
                if unit_lower == "pa":
                    current_in_channel = ch
                    break
        
        if current_in_channel is not None:
            stim_channel = current_in_channel
            print(f"{filename}: assigned stim_channel={stim_channel} from Current_in/pA channel")
        else:
            print(f"{filename}: WARNING - no pA Current_in channel found")
    
        print(f"{filename}: response_channel={response_channel}, stim_channel={stim_channel}")
        
        # Go through the sweeps to find time frames where long square sweep may be available
        fallback_stim_start, fallback_stim_end = util_functions.infer_long_square_window_from_abf(abf)
        
        current_in_start, current_in_end = util_functions.infer_current_in_timing_for_file(
            abf=abf,
            stim_channel=stim_channel,
        )
        
        print(
            f"{filename}: current_in_start={current_in_start}, "
            f"current_in_end={current_in_end}"
        )
        
        for i in abf.sweepList:
            sampling_rate = abf.sampleRate
    
            # response
            try:
                abf.setSweep(i, channel=response_channel)
                dataX = abf.sweepX.copy()
                dataY = abf.sweepY.copy()
            
                dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source = (
                    util_functions.get_stim_from_sweepC_or_current_in(
                        abf=abf,
                        sweep_number=i,
                        response_channel=response_channel,
                        stim_channel=stim_channel,
                        dataX=dataX,
                        file_current_start=current_in_start,
                        file_current_end=current_in_end,
                    )
                )
            
                print(
                    f"{filename} sweep {i}: "
                    f"resp={response_channel}, stim={stim_channel}, "
                    f"stim_source={stim_source}, stim_mode={stim_mode}, stim_amp={stim_amp}"
                )
            
            except Exception as e:
                print(f"{filename} sweep {i}: response/stim read failed on channel {response_channel}: {e}")
                continue
            
            # restore response
            try:
                abf.setSweep(i, channel=response_channel)
                dataY = abf.sweepY.copy()
            except Exception as e:
                print(f"{filename} sweep {i}: response restore failed on channel {response_channel}: {e}")
                continue
            
            # # NOW apply RMP correction, after the final dataY reload
            # dataY, measured_baseline_mV, voltage_offset_mV = (
            #     util_functions.apply_rmp_baseline_correction(
            #         time=dataX,
            #         voltage=dataY,
            #         true_rmp_mV=true_rmp_mV,
            #         stim_start=stim_start,
            #         baseline_duration=0.05,
            #         correction_tolerance_mV=BASELINE_TOLERANCE,
            #     )
            # )
            
            # if true_rmp_mV is not None and measured_baseline_mV is not None:
            #     if abs(voltage_offset_mV) > 0:
            #         print(
            #             f"{filename} sweep {i}: RMP correction applied. "
            #             f"measured_baseline={measured_baseline_mV:.2f} mV, "
            #             f"true_rmp={true_rmp_mV:.2f} mV, "
            #             f"offset={voltage_offset_mV:.2f} mV"
            #         )
            #     else:
            #         print(
            #             f"{filename} sweep {i}: RMP correction skipped. "
            #             f"measured_baseline={measured_baseline_mV:.2f} mV, "
            #             f"true_rmp={true_rmp_mV:.2f} mV, "
            #             f"difference within tolerance"
            #         )
                        
            try:
                clampMode = clamp_modes[abf._adcSection.nTelegraphMode[response_channel]]
            except AttributeError:
                clampMode = util_functions.infer_clamp_mode(abf)
            
            if stim_start is None or stim_end is None or stim_mode != "Long":
                continue
            
            if stim_amp is None:
                continue
            
            if stim_amp > MAX_STIM_AMP_PA:
                print(
                    f"{filename} sweep {i}: excluded because "
                    f"stim_amp={stim_amp:.2f} pA > {MAX_STIM_AMP_PA} pA"
                )
                continue
    
            sweep = Sweep(
                t=dataX,
                v=dataY,
                i=dataStim,
                sampling_rate=sampling_rate,
                sweep_number=i,
                clamp_mode=clampMode,
            )
    
            valid_sweeps.append(sweep)
            start_times.append(stim_start)
            end_times.append(stim_end)
            stim_amps.append(stim_amp)
    
            if stim_amp is not None:
                if min_amp is None or stim_amp < min_amp:
                    min_amp = stim_amp
                if stim_amp > 0:
                    positive_stim_count += 1
    
        if not valid_sweeps or not start_times or not end_times:
            print(f"{filename}: no valid long-square sweeps found, skipping")
            continue
    
        start_time, end_time, keep_mask, timing_source = util_functions.refine_long_square_timing(
            start_times=start_times,
            end_times=end_times,
            anchor_start=current_in_start,
            anchor_end=current_in_end,
            start_end_tol_s=0.025,
            duration_tol_s=0.050,
            min_keep=3,
        )
        
        if start_time is None or end_time is None:
            print(f"{filename}: could not refine long-square timing, skipping")
            continue
        
        n_removed = len(valid_sweeps) - int(np.sum(keep_mask))
        
        if n_removed > 0:
            print(
                f"{filename}: removed {n_removed} timing outlier sweep(s) "
                f"before LongSquareAnalysis using {timing_source}"
            )
        
        # Remove the matching sweeps, times, and amplitudes
        valid_sweeps = [sw for sw, keep in zip(valid_sweeps, keep_mask) if keep]
        start_times = [x for x, keep in zip(start_times, keep_mask) if keep]
        end_times = [x for x, keep in zip(end_times, keep_mask) if keep]
        stim_amps = [x for x, keep in zip(stim_amps, keep_mask) if keep]
        
        if not valid_sweeps:
            print(f"{filename}: no valid sweeps remain after timing outlier removal, skipping")
            continue
        
        # Recompute these after filtering
        min_amp = min(stim_amps)
        positive_stim_count = sum(1 for amp in stim_amps if amp > 0)
        
        baseline_interval = 0.008
        
        # IPFX SpikeFeatureExtractor(filter=1) needs time before start_time.
        # If start_time is too close to t=0, it tries to read negative time.
        filter_pad_s = 0.030
        
        t0 = valid_sweeps[0].t[0]
        t1 = valid_sweeps[0].t[-1]
        
        start_time = max(start_time, t0 + filter_pad_s)
        
        # Keep end_time inside the sweep too
        end_time = min(end_time, t1)
        
        # Safety check
        if start_time >= end_time:
            print(
                f"{filename}: invalid analysis window after padding: "
                f"start={start_time:.4f}, end={end_time:.4f}, "
                f"t0={t0:.4f}, t1={t1:.4f}"
            )
            continue
        
        print(
            f"{filename}: analysis window start={start_time:.4f}, "
            f"end={end_time:.4f}, stim_start_min={min(start_times):.4f}"
        )
        
        sweepSet = SweepSet(valid_sweeps)
        
        spike_extractor = SpikeFeatureExtractor(
            start=start_time,
            end=end_time,
            filter=1
        )
        
        spike_train_extractor = SpikeTrainFeatureExtractor(
            start=start_time,
            end=end_time,
            baseline_interval=baseline_interval
        )
    
        if positive_stim_count <= 1:
            print(f"{filename}: not enough positive stim sweeps for LongSquareAnalysis")
            continue
    
        has_spiking_sweep = False
        for sweep in sweepSet.sweeps:
            spikes_df = spike_extractor.process(sweep.t, sweep.v, sweep.i)
            if spikes_df is not None and len(spikes_df) > 0:
                has_spiking_sweep = True
                break
    
        if not has_spiking_sweep:
            print(f"{filename}: no spiking long square sweeps found, skipping")
            continue
    
        if structure is None:
            structure = "No Value"
            sex = "No Value"
            age = "No Value"
            
        else:
            structure = structure
            sex = str(sex)
            age = str(age)
        
        if species == "human":
            meta = {
                "file": filename,
                "folder": folder_name_safe,
                "protocol": conditions,
                "abf_datetime": datetimestring,
                "structure": structure,
                "layer": layer,
                "layer_source": layer_source,
                "true_rmp_mV": true_rmp_mV,
                "rmp_source": rmp_source,
                "model": model,
                "sex": sex,
                "age": age,
                "tissue_type": tissue_type,
                "tumour": tumour,
            }
        
        else:
            meta = {
                "file": filename,
                "folder": folder_name_safe,
                "protocol": conditions,
                "abf_datetime": datetimestring,
                "structure": structure,
                "layer": layer,
                "layer_source": layer_source,
                "true_rmp_mV": true_rmp_mV,
                "rmp_source": rmp_source,
                "model": model,
                "sex": sex,
                "age": age,
                "tissue_type": tissue_type,
            }
    
        print("Analyzing...")
    
        try:
            lsa = LongSquareAnalysis(
                spx=spike_extractor,
                sptx=spike_train_extractor,
                subthresh_min_amp=min_amp,
            )
                
            lsa_results = lsa.analyze(sweepSet)
            
            # Run IPFX first
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Mean of empty slice",
                    category=RuntimeWarning,
                )
            
                lsa_results = lsa.analyze(sweepSet)
            
            # Check IPFX tau
            ipfx_tau = lsa_results.get("tau", np.nan)
            
            if ipfx_tau is None or not np.isfinite(ipfx_tau):
                fallback_tau = estimate_tau_relaxed_ipfx_snr(
                    sweeps=valid_sweeps,
                    stim_amps=stim_amps,
                    start_time=start_time,
                    end_time=end_time,
                    subthresh_min_amp=min_amp,
                    baseline_interval=baseline_interval,
                    tau_frac=0.1,
                    min_snr=5.0,
                )
            
                print(
                    f"{filename}: IPFX tau was NaN; "
                    f"relaxed-SNR fallback tau = {fallback_tau}"
                )
            
                if np.isfinite(fallback_tau):
                    lsa_results["tau"] = fallback_tau
                    lsa_results["tau_source"] = "relaxed_ipfx_snr5"
                else:
                    lsa_results["tau_source"] = "missing"
            else:
                lsa_results["tau_source"] = "ipfx"
                
            summarized_cell_features = util_functions.summarize_cell_ephys_features(lsa_results)
            meta["tau_source"] = lsa_results.get("tau_source", "unknown")
            
            meta_dict[filename] = meta
            data_dict[filename] = summarized_cell_features
            folder_dict[filename] = summarized_cell_features
    
        except Exception as e:
            print(f"{filename}: LongSquareAnalysis failed: {e}")
            continue
                    
        
        if len(folder_dict) == 0:
            print(f"Skipping folder (no analyzed files): {dirpath}")
            continue
        
    df = pd.DataFrame(folder_dict)
    
    out_csv = os.path.join(dirpath, f"{folder_name_safe}_example_multi.csv")
    df.to_csv(out_csv)
    
    mean_vals = df.mean(axis=1)
    std_vals = df.std(axis=1)
    cv_vals = std_vals / mean_vals.replace(0, np.nan)
    
    summary_df = pd.DataFrame({"mean": mean_vals, "std": std_vals, "cv": cv_vals})
    out_sum = os.path.join(dirpath, f"{folder_name_safe}_summary_stats.csv")
    summary_df.to_csv(out_sum)
        
    # summary_df.to_csv(dirpath +  '/_' + folder_name + '_summary_stats.csv')

## Features with Metadata
df_data = pd.DataFrame(data_dict)
df_feat = pd.DataFrame(data_dict).T  # rows = files, cols = features
df_meta = pd.DataFrame(meta_dict).T  # rows = files, cols = metadata

df_all = df_meta.join(df_feat)
df_all.to_csv(input_abf_root + "/all_features_with_meta.csv", index=True)

mean_vals = df_data.mean(axis=1)
std_vals = df_data.std(axis=1)
cv_vals = std_vals / mean_vals.replace(0, np.nan)

summary_all_df = pd.DataFrame({"mean": mean_vals, "std": std_vals, "cv": cv_vals})
out_all_sum = os.path.join(input_abf_root, "all_summary_stats.csv")
summary_all_df.to_csv(out_all_sum)
