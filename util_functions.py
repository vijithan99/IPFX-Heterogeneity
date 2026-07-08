# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:26:37 2026

@author: vijit
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import re

## File Details
experimenter = "Homeira Moradi Chameh"
lab = "Neuron to Brain Lab"

# Adjust if needed after reviewing your dataset
RMP_MIN_MV = -100.0
RMP_MAX_MV = -35.0

def _normalize_tag_text(text):
    """
    Normalize ABF tag text before regex parsing.
    Handles unicode minus signs and accidental double negatives.
    """
    text = str(text)

    # Normalize unicode dashes/minus signs to normal hyphen
    text = (
        text.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("-", "-")
    )

    # Handle accidental double signs before numbers:
    # RMP --62.1 -> RMP -62.1
    # RMP - -62.1 -> RMP -62.1
    text = re.sub(r"(?<![\d.])-\s*-\s*(?=\d)", "-", text)
    text = re.sub(r"(?<![\d.])\+\s*\+\s*(?=\d)", "+", text)
    text = re.sub(r"(?<![\d.])\+\s*-\s*(?=\d)", "-", text)
    text = re.sub(r"(?<![\d.])-\s*\+\s*(?=\d)", "-", text)

    return text


def _parse_float_token(token):
    token = _normalize_tag_text(token)
    token = token.replace(" ", "")

    try:
        return float(token)
    except Exception:
        return None


def _is_valid_rmp(value):
    """
    Safety gate for true RMP values.
    This rejects offset -20 mV, injected current -667.2, etc.
    """
    if value is None:
        return False

    return RMP_MIN_MV <= value <= RMP_MAX_MV


def _bad_standalone_context(text, start, end):
    """
    For fallback standalone mV values, reject values that are probably
    offsets, gains, current amplitudes, step sizes, etc.
    """
    window = text[max(0, start - 35): min(len(text), end + 35)].lower()

    bad_words = [
        "offset",
        "p offset",
        "poffset",
        "gain",
        "dc",
        "step",
        "step size",
        "current",
        "injected",
        "bridge",
        "balance",
        "pia",
        "um",
        "µm",
        "pa",
        "na",
    ]

    return any(word in window for word in bad_words)


def extract_rmp_from_text(text, allow_standalone=True):
    """
    Extract true RMP from messy ABF tag/comment text.

    Handles:
        'RMP -68.7 mV'
        'rmp -71.6 MV'
        'RMP: -68.7 mV'
        'RMP=-68.7 mV'
        'RMP --62.1 mV'
        'RMP -62.4'
        'C6-L3 -61.2 mv'
        'RMP -64.8 V'   # treats V as a typo for mV if in valid RMP range

    Returns:
        float RMP in mV, or None.
    """

    if text is None:
        return None

    text = _normalize_tag_text(text)

    # -----------------------------
    # PASS 1: explicit RMP keyword
    # -----------------------------
    rmp_pattern = re.compile(
        r"""
        \b
        (?:
            RMP |
            VREST |
            V_REST |
            VMREST |
            VM_REST |
            resting\s*membrane\s*potential
        )
        \b
        \s*[:=]?\s*
        (?P<num>[+\-]?\s*[+\-]?\s*\d+(?:\.\d+)?)
        \s*
        (?:
            m\s*v |
            mv |
            v
        )?
        \b
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    for match in rmp_pattern.finditer(text):
        value = _parse_float_token(match.group("num"))

        if _is_valid_rmp(value):
            return value

    # -----------------------------------------------------
    # PASS 2: fallback standalone physiological mV value
    # -----------------------------------------------------
    # Only use this for tags like:
    #     'C6-L3 -61.2 mv'
    #
    # Do not use arbitrary negative numbers without voltage units.
    if not allow_standalone:
        return None

    mv_pattern = re.compile(
        r"""
        (?P<num>[+\-]?\s*[+\-]?\s*\d+(?:\.\d+)?)
        \s*
        (?:
            m\s*v |
            mv |
            v
        )
        \b
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    candidates = []

    for match in mv_pattern.finditer(text):
        value = _parse_float_token(match.group("num"))

        if not _is_valid_rmp(value):
            continue

        if _bad_standalone_context(text, match.start(), match.end()):
            continue

        candidates.append(value)

    # Be conservative: if exactly one plausible standalone voltage exists, accept it.
    # If there are multiple, leave it as None for manual review.
    if len(candidates) == 1:
        return candidates[0]

    return None


def extract_rmp_from_abf_tags(abf):
    """
    Extract true RMP from ABF tag comments.

    Returns:
        rmp_mV, source_comment
    """

    tag_comments = []

    try:
        if hasattr(abf, "tagComments"):
            tag_comments.extend(list(abf.tagComments))
    except Exception:
        pass

    try:
        comments = abf._tagSection.sComment

        if isinstance(comments, str):
            tag_comments.append(comments)
        else:
            tag_comments.extend(list(comments))
    except Exception:
        pass

    # Optional extra places to search
    try:
        if hasattr(abf, "abfFileComment") and abf.abfFileComment:
            tag_comments.append(abf.abfFileComment)
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    clean_comments = []

    for comment in tag_comments:
        # Sometimes comments may come in as tuples like (time, comment)
        if isinstance(comment, (list, tuple)) and len(comment) > 0:
            comment = comment[-1]

        comment = str(comment)

        if comment not in seen:
            clean_comments.append(comment)
            seen.add(comment)

    # First pass: only accept explicit RMP-containing comments
    for comment in clean_comments:
        rmp = extract_rmp_from_text(comment, allow_standalone=False)

        if rmp is not None:
            return rmp, comment

    # Second pass: allow standalone '-61.2 mV'-type comments
    for comment in clean_comments:
        rmp = extract_rmp_from_text(comment, allow_standalone=True)

        if rmp is not None:
            return rmp, comment

    return None, None

def measure_voltage_baseline(time, voltage, stim_start=None, baseline_duration=0.05):
    """
    Measure apparent voltage baseline before the current step.

    If stim_start is known:
        uses the baseline_duration immediately before stim_start.

    If stim_start is unknown:
        uses the first baseline_duration of the sweep.
    """

    time = np.asarray(time)
    voltage = np.asarray(voltage)

    if time.size == 0 or voltage.size == 0:
        return None

    if stim_start is not None:
        baseline_start = max(time[0], stim_start - baseline_duration)
        baseline_end = stim_start
    else:
        baseline_start = time[0]
        baseline_end = time[0] + baseline_duration

    idx = (time >= baseline_start) & (time < baseline_end)

    if not np.any(idx):
        return None

    return float(np.nanmedian(voltage[idx]))

def apply_rmp_baseline_correction(
    time,
    voltage,
    true_rmp_mV,
    stim_start=None,
    baseline_duration=0.05,
    correction_tolerance_mV=10.0,
):
    """
    Shift voltage trace only if measured baseline differs meaningfully
    from the true RMP.

    If the difference is within correction_tolerance_mV, no correction is applied.

    Returns:
        corrected_voltage, measured_baseline, voltage_offset
    """

    if true_rmp_mV is None:
        return voltage, None, 0.0

    measured_baseline = measure_voltage_baseline(
        time=time,
        voltage=voltage,
        stim_start=stim_start,
        baseline_duration=baseline_duration,
    )

    if measured_baseline is None:
        return voltage, None, 0.0

    raw_offset = float(true_rmp_mV - measured_baseline)

    # No correction needed if baseline is already close enough to true RMP
    if abs(raw_offset) <= correction_tolerance_mV:
        return voltage, measured_baseline, 0.0

    corrected_voltage = np.asarray(voltage) + raw_offset

    return corrected_voltage, measured_baseline, raw_offset

def normalize_layer_label(layer_raw):
    """
    Normalize extracted layer strings into consistent labels.

    Examples:
        'L5'       -> 'L5'
        'layer 5'  -> 'L5'
        'L2/3'     -> 'L2/3'
        'L2&3'     -> 'L2/3'
        'L2-3'     -> 'L2/3'
        'L3c'      -> 'L3c'
    """

    if layer_raw is None:
        return None

    s = str(layer_raw).strip()
    s = s.replace(" ", "")
    s = s.replace("&", "/")
    s = s.replace("-", "/")

    # Remove full word prefix if present
    s = re.sub(r"(?i)^layer", "L", s)

    # Force uppercase L, preserve c as lowercase if L3c
    s = re.sub(r"(?i)^l", "L", s)

    if re.fullmatch(r"L[1-6]", s, flags=re.IGNORECASE):
        return s.upper()

    if re.fullmatch(r"L[1-6]/[1-6]", s, flags=re.IGNORECASE):
        return s.upper()

    if re.fullmatch(r"L3C", s, flags=re.IGNORECASE):
        return "L3c"

    return s


def extract_layer_from_text(text):
    """
    Extract cortical layer from a tag/comment string.

    Handles:
        L5
        l5
        Layer 5
        layer 5
        L2/3
        L2&3
        L2-3
        Layer 2/3
        L3c

    Returns:
        normalized layer string, e.g. 'L5', 'L2/3', 'L3c'
        or None if no layer is found.
    """

    if text is None:
        return None

    text = str(text)

    patterns = [
        # Layer 2/3, layer 2-3, layer 2&3
        r"\b(?:layer|Layer)\s*([1-6]\s*[/&-]\s*[1-6])\b",

        # L2/3, l2&3, L2-3
        r"\b[Ll]\s*([1-6]\s*[/&-]\s*[1-6])\b",

        # L3c / l3c
        r"\b[Ll]\s*(3\s*[cC])\b",

        # Layer 5, layer 2
        r"\b(?:layer|Layer)\s*([1-6])\b",

        # L5, l5, but not part of words
        r"\b[Ll]\s*([1-6])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            raw = "L" + match.group(1)
            return normalize_layer_label(raw)

    return None

def extract_layer_from_abf_tags(abf):
    """
    Extract cortical layer from ABF tag comments.

    Returns:
        layer, layer_source_text

    Example:
        ('L5', 'C1, RMP -63.7 mV L5, Synaptic blocke APv.CNQX, PTX,')
    """

    tag_comments = []

    # ABF2-ish / pyABF tag access
    try:
        if hasattr(abf, "tagComments"):
            tag_comments.extend(list(abf.tagComments))
    except Exception:
        pass

    # Older/private tag section access
    try:
        comments = abf._tagSection.sComment

        if isinstance(comments, str):
            tag_comments.append(comments)
        else:
            tag_comments.extend(list(comments))

    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    clean_comments = []

    for comment in tag_comments:
        comment = str(comment)

        if comment not in seen:
            clean_comments.append(comment)
            seen.add(comment)

    for comment in clean_comments:
        layer = extract_layer_from_text(comment)

        if layer is not None:
            return layer, comment

    return None, None

def infer_culture_days(protocol):
    """
    Infer culture duration from protocol string.

    Returns:
        1 for 24h culture
        2 for 48h culture
        3 for 72h culture
        None if no culture duration is found
    """
    protocol_lower = str(protocol).lower()

    if "culture" not in protocol_lower and "cultured" not in protocol_lower:
        return None

    hour_to_days = {
        "24": 1,
        "48": 2,
        "72": 3,
    }

    # Strong match: 24h, 24 h, 48hr, 72 hours, etc.
    explicit_hour_match = re.search(
        r"(?<![\d.])(?P<hours>24|48|72)(?![\d.])\s*(h|hr|hrs|hour|hours)\b",
        protocol_lower
    )

    if explicit_hour_match:
        hours = explicit_hour_match.group("hours")
        return hour_to_days[hours]

    # Weaker match: 48 post culture, culture 48, 72 post cultured, etc.
    candidate_pattern = r"(?<![\d.])(?P<hours>24|48|72)(?![\d.])"

    for match in re.finditer(candidate_pattern, protocol_lower):
        hours = match.group("hours")
        start, end = match.span()

        before = protocol_lower[max(0, start - 30):start]
        after = protocol_lower[end:min(len(protocol_lower), end + 30)]
        context = before + hours + after

        # Reject electrophysiology values like -72 mV, 72 mV, 48 pA, etc.
        unit_after = re.match(r"\s*(mv|v|pa|na|ua|ma|hz|khz)\b", after)
        if unit_after:
            continue

        # Reject if clearly part of RMP value nearby
        if "rmp" in before[-15:]:
            continue

        # Accept only if culture-related words are nearby
        culture_nearby = (
            "culture" in context
            or "cultured" in context
            or "post" in context
            or "after" in context
        )

        if culture_nearby:
            return hour_to_days[hours]

    return None

def parse_mouse_metadata_from_path(dirpath, input_abf_root):
    """
    Extract mouse metadata from folder structure.

    Expected structure:
        miceData / structure / condition / model / cell_type

    Example:
        miceData / mPFC / KA / Focal / Interneuron
    """
    relative_path = os.path.relpath(dirpath, input_abf_root)
    parts = relative_path.split(os.sep)

    mouse_meta = {
        "structure": None,
        "condition": None,
        "model": None,
        "cell_type": None,
        "tissue_type": "Mouse slice",
    }

    if len(parts) > 0:
        mouse_meta["structure"] = parts[0]

    if len(parts) > 1:
        mouse_meta["condition"] = parts[1]

    if len(parts) > 2:
        mouse_meta["model"] = parts[2]

    if len(parts) > 3:
        mouse_meta["cell_type"] = parts[3]

    return mouse_meta

def patient_data_parse(abf_datetime_string, patient_data_path, protocol):
    """
    Match an ABF recording date to patient metadata.

    Tries:
        1. Same date, ISO format: YYYY-MM-DD
        2. Previous/culture-adjusted date, ISO format: YYYY-MM-DD
        3. Same date, old CSV format: M/D/YYYY
        4. Previous/culture-adjusted date, old CSV format: M/D/YYYY

    Notes:
        For slice culture protocols, the resection date may be 24h, 48h, or 72h
        before the recording date.

    Returns:
        patient_date_db, csv_missing, tissue_type
    """
    csv_missing = None

    # Extract date and remove time
    date_str = abf_datetime_string.split("T")[0]

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    prev_date_obj = date_obj - timedelta(days=1)

    protocol_lower = str(protocol).lower()
    
    culture_days = infer_culture_days(protocol)

    if culture_days is not None:
        prev_date_obj = date_obj - timedelta(days=culture_days)
    elif "culture" in protocol_lower or "cultured" in protocol_lower:
        # Default assumption if culture is mentioned but no duration is clear
        prev_date_obj = date_obj - timedelta(days=1)
    else:
        prev_date_obj = date_obj - timedelta(days=1)
    # if "culture" in protocol_lower:
    #     if "48" in protocol_lower:
    #         prev_date_obj = date_obj - timedelta(days=2)
    #     elif "72" in protocol_lower:
    #         prev_date_obj = date_obj - timedelta(days=3)
    #     else:
    #         prev_date_obj = date_obj - timedelta(days=1)

    date_iso = date_obj.strftime("%Y-%m-%d")
    prev_date_iso = prev_date_obj.strftime("%Y-%m-%d")

    date_old = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
    prev_date_old = f"{prev_date_obj.month}/{prev_date_obj.day}/{prev_date_obj.year}"

    patient_db = pd.read_csv(patient_data_path)

    resection_dates = patient_db["Resection Date"].astype(str)

    search_attempts = [
        (date_iso, "Acute Slice", "New Date Format - Acute Slice"),
        (prev_date_iso, "Slice Culture", "New Date Format - Slice Culture"),
        (date_old, "Acute Slice", "Old Date Format - Acute Slice"),
        (prev_date_old, "Slice Culture", "Old Date Format - Slice Culture"),
    ]

    for search_date, tissue_type, message in search_attempts:
        patient_date_db = patient_db.loc[
            resection_dates.str.contains(search_date, na=False, regex=False)
        ]

        if not patient_date_db.empty:
            print(message)
            return patient_date_db, csv_missing, tissue_type

    print("Date Empty")
    csv_missing = date_iso
    tissue_type = "No Info"

    return patient_date_db, csv_missing, tissue_type



def patient_data_extract(patient_data):
    """
    Extract one row of patient metadata as plain Python values.
    """
    if patient_data is None or patient_data.empty:
        return None, "U", None, "No Info", "No Info"

    row = patient_data.iloc[0]

    tissue_info = str(row.get("Valiante tissue", "")).lower()

    if "tumour" in tissue_info or "tumor" in tissue_info:
        tumour = "Tumour"
    else:
        tumour = "No Tumour"

    if "non" in tissue_info and "epileptogenic" in tissue_info:
        tissue_type = "Non-Epileptogenic"
    elif "non" in tissue_info and "epileptic" in tissue_info:
        tissue_type = "Non-Epileptic"
    elif "epileptogenic" in tissue_info:
        tissue_type = "Epileptogenic"
    elif "epileptic" in tissue_info:
        tissue_type = "Epileptic"
    else:
        tissue_type = "No Info"

    structure = row.get("TissueLocation", None)
    sex = row.get("Sex", "U")
    age = row.get("Age At OR ", None)

    return structure, sex, age, tissue_type, tumour

def infer_long_square_window_from_abf(abf, threshold=1e-12):
    starts = []
    ends = []

    for sweep in abf.sweepList:
        abf.setSweep(sweep)

        stim = abf.sweepC
        time = abf.sweepX

        nonzero = np.where(np.abs(stim) > threshold)[0]

        if nonzero.size > 0:
            starts.append(float(time[nonzero[0]]))
            ends.append(float(time[nonzero[-1]]))

    if len(starts) == 0:
        return None, None

    stim_start = float(np.median(starts))
    stim_end = float(np.median(ends))

    return stim_start, stim_end

def refine_long_square_timing(
    start_times,
    end_times,
    anchor_start=None,
    anchor_end=None,
    start_end_tol_s=0.025,
    duration_tol_s=0.050,
    min_keep=3,
):
    """
    Remove outlier stim_start/stim_end values, then return cleaned min/max timing.

    If anchor_start/anchor_end are provided, they are treated as the trusted
    file-level current injection timing.

    Returns:
        refined_start, refined_end, keep_mask, timing_source
    """

    starts = np.asarray(start_times, dtype=float)
    ends = np.asarray(end_times, dtype=float)

    finite = (
        np.isfinite(starts)
        & np.isfinite(ends)
        & (ends > starts)
    )

    if not np.any(finite):
        return None, None, np.zeros(len(starts), dtype=bool), "no_valid_times"

    # Use file-level current timing as the anchor if available.
    # Otherwise use the median of collected sweep times.
    if anchor_start is not None and anchor_end is not None:
        ref_start = float(anchor_start)
        ref_end = float(anchor_end)
        timing_source = "file_current_in_anchor"
    else:
        ref_start = float(np.median(starts[finite]))
        ref_end = float(np.median(ends[finite]))
        timing_source = "median_anchor"

    ref_duration = ref_end - ref_start
    durations = ends - starts

    keep = (
        finite
        & (np.abs(starts - ref_start) <= start_end_tol_s)
        & (np.abs(ends - ref_end) <= start_end_tol_s)
        & (np.abs(durations - ref_duration) <= duration_tol_s)
    )

    # If the filter is too aggressive, fall back safely.
    if np.sum(keep) < min_keep:
        if anchor_start is not None and anchor_end is not None:
            # Trust file-level timing, but do not discard all sweeps.
            keep = finite.copy()
            return ref_start, ref_end, keep, "file_current_in_anchor_fallback"
        else:
            # Use median timing instead of dangerous min/max.
            keep = finite.copy()
            return ref_start, ref_end, keep, "median_fallback"

    # This is the compromise you asked for:
    # use min/max, but only after removing outliers.
    refined_start = float(np.min(starts[keep]))
    refined_end = float(np.max(ends[keep]))

    return refined_start, refined_end, keep, timing_source

def stim_range(stim, time, threshold=1e-12, fallback_start=None, fallback_end=None):
    active_idx = np.where(np.abs(stim) > threshold)[0]

    if active_idx.size == 0:
        if fallback_start is not None and fallback_end is not None:
            return fallback_start, fallback_end, 0.0, "Long"
        return None, None, None, "None"

    start_idx = active_idx[0]
    end_idx = active_idx[-1]

    start = float(time[start_idx])
    end = float(time[end_idx])

    active_stim = stim[active_idx]

    stim_start_amp = float(stim[start_idx])
    stim_end_amp = float(stim[end_idx])

    if np.isclose(stim_start_amp, stim_end_amp, atol=threshold):
        stim_mode = "Long"
        stim_amp = float(np.median(active_stim))
    else:
        stim_mode = "Ramp"
        stim_amp = stim_end_amp

    return start, end, stim_amp, stim_mode

def measure_stim_amp_from_window(time, stim_trace, stim_start, stim_end, baseline_interval=0.05):
    """
    Measure current step amplitude from recorded Current_in channel.

    Returns:
        stim_amp = median current during step - median current before step
    """
    time = np.asarray(time)
    stim_trace = np.asarray(stim_trace)

    baseline_start = max(time[0], stim_start - baseline_interval)
    baseline_end = stim_start

    baseline_idx = (time >= baseline_start) & (time < baseline_end)
    stim_idx = (time >= stim_start) & (time <= stim_end)

    if not np.any(baseline_idx) or not np.any(stim_idx):
        return None

    baseline_current = np.median(stim_trace[baseline_idx])
    step_current = np.median(stim_trace[stim_idx])

    return float(step_current - baseline_current)

def get_protocol_stim_robust(abf, sweep_number, time):
    """
    Robust protocol-based stimulation reconstruction.

    Priority:
        1. ABF2 _epochPerDacSection
        2. ABF1 / old Homeira I-V defaults from protocol name
        3. Explicit protocol-name parsing

    Returns:
        dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source
        or None
    """

    # -----------------------------
    # 1. ABF2 epoch-per-DAC section
    # -----------------------------
    result = get_protocol_stim_from_epoch_per_dac(
        abf=abf,
        sweep_number=sweep_number,
        time=time,
        target_dac=None,
    )

    if result is not None:
        return result

    # -----------------------------
    # 2. Protocol-name / old ABF1 fallback
    # -----------------------------
    protocol_info = parse_iv_protocol_from_name(
        getattr(abf, "protocol", "")
    )

    if protocol_info is not None:
        return get_protocol_stim_from_parsed_info(
            sweep_number=sweep_number,
            time=time,
            protocol_info=protocol_info,
        )

    return None

def parse_iv_protocol_from_name(protocol_name):
    """
    Parse I-V protocol names and provide ABF1/ABF2 defaults.

    Examples:
        'I-V curve ,-400 pA'
        'I-V curve ,-300 pA'
        'I-V curve.pro'
    """

    protocol_name = str(protocol_name)
    protocol_lower = protocol_name.lower()

    if "i-v" not in protocol_lower and "iv" not in protocol_lower:
        return None

    # Case 1: protocol name explicitly contains starting amplitude
    # Example: "I-V curve ,-400 pA"
    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*pA",
        protocol_name,
        re.IGNORECASE
    )

    if match:
        first_level_pa = float(match.group(1))

        # Your newer ABF2 Homeira files used this timing:
        # A = 1000 samples = ~51 ms
        # B = 12000 samples = ~612 ms
        return {
            "first_level_pa": first_level_pa,
            "delta_level_pa": 50.0,
            "pre_duration_s": 0.051,
            "step_duration_s": 0.612,
            "source": "protocol_name",
        }

    # Case 2: older ABF1 Homeira protocol name is generic:
    # "I-V curve.pro"
    # From your ABF1 protocol table:
    # A = 3200 samples = 192 ms
    # B = 12000 samples = 720 ms
    # First level = -400 pA
    # Delta = 50 pA
    return {
        "first_level_pa": -400.0,
        "delta_level_pa": 50.0,
        "pre_duration_s": 0.192,
        "step_duration_s": 0.720,
        "source": "old_abf1_iv_default",
    }

def get_old_abf1_protocol_fallback(abf, sweep_number, time):
    """
    Fallback for older ABF1 Homeira protocols where pyABF does not expose
    _epochPerDacSection, but protocol naming is consistent.
    """

    protocol = str(getattr(abf, "protocol", "")).lower()

    # Long Square 1S - 220-330-40 low R
    # Actual protocol table:
    # First level = -240 pA, delta = 40 pA, duration = 1000 ms
    if "long square" in protocol:
        first_level_pa = -240.0
        delta_level_pa = 40.0
        pre_duration_s = 0.005
        step_duration_s = 1.000

        stim_amp = first_level_pa + delta_level_pa * sweep_number
        stim_start = pre_duration_s
        stim_end = stim_start + step_duration_s

        dataStim = np.zeros_like(time, dtype=float)
        dataStim[(time >= stim_start) & (time <= stim_end)] = stim_amp

        return dataStim, stim_start, stim_end, stim_amp, "Long", "old_abf1_long_square_protocol"

    # Old I-V curve.pro
    if "i-v curve" in protocol or "iv curve" in protocol:
        first_level_pa = -400.0
        delta_level_pa = 50.0
        pre_duration_s = 0.192
        step_duration_s = 0.720

        stim_amp = first_level_pa + delta_level_pa * sweep_number
        stim_start = pre_duration_s
        stim_end = stim_start + step_duration_s

        dataStim = np.zeros_like(time, dtype=float)
        dataStim[(time >= stim_start) & (time <= stim_end)] = stim_amp

        return dataStim, stim_start, stim_end, stim_amp, "Long", "old_abf1_iv_protocol"

    return None

def get_stim_from_sweepC_or_current_in(
    abf,
    sweep_number,
    response_channel,
    stim_channel,
    dataX,
    file_current_start=None,
    file_current_end=None,
    sweepC_threshold=1e-12,
):
    """
    Robust stimulation extraction for ABF1 and ABF2.

    Priority:
        1. Pure protocol reconstruction, only if timing is consistent
        2. Protocol amplitude + Current_in timing
        3. Valid sweepC
        4. Old ABF1 protocol fallback, only if timing is consistent or no Current_in timing exists
        5. Raw Current_in fallback
    """

    def timing_matches_file_timing(stim_start, stim_end, tol=0.075):
        """
        Return True if protocol timing agrees with file-level Current_in timing.

        If file-level timing is unavailable, allow the candidate.
        """
        if file_current_start is None or file_current_end is None:
            return True

        if stim_start is None or stim_end is None:
            return False

        start_ok = abs(stim_start - file_current_start) <= tol
        end_ok = abs(stim_end - file_current_end) <= tol

        return start_ok and end_ok

    # --------------------------------------------------
    # 1. Pure protocol/header reconstruction
    #    Use only if timing agrees with Current_in timing.
    # --------------------------------------------------
    protocol_result = get_protocol_stim_robust(
        abf=abf,
        sweep_number=sweep_number,
        time=dataX,
    )

    if protocol_result is not None:
        dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source = protocol_result

        if timing_matches_file_timing(stim_start, stim_end):
            return protocol_result

    # --------------------------------------------------
    # 2. Best fallback for your current dataset:
    #    protocol amplitude + file-level Current_in timing
    # --------------------------------------------------
    hybrid_result = get_stim_protocol_amp_current_timing(
        abf=abf,
        sweep_number=sweep_number,
        stim_channel=stim_channel,
        dataX=dataX,
        file_current_start=file_current_start,
        file_current_end=file_current_end,
    )

    if hybrid_result is not None:
        return hybrid_result

    # --------------------------------------------------
    # 3. sweepC, only if real and long enough
    # --------------------------------------------------
    try:
        abf.setSweep(sweep_number, channel=response_channel)
        commandStim = abf.sweepC.copy()

        if np.nanmax(np.abs(commandStim)) > sweepC_threshold:
            stim_start, stim_end, stim_amp, stim_mode = stim_range(
                commandStim,
                dataX,
                threshold=sweepC_threshold,
                fallback_start=None,
                fallback_end=None,
            )

            if (
                stim_start is not None
                and stim_end is not None
                and stim_amp is not None
                and stim_mode == "Long"
                and (stim_end - stim_start) > 0.2
            ):
                return (
                    commandStim,
                    stim_start,
                    stim_end,
                    stim_amp,
                    stim_mode,
                    "sweepC",
                )

    except Exception:
        pass

    # --------------------------------------------------
    # 4. Old ABF1 fallback
    #    Only use if it agrees with Current_in timing,
    #    or if Current_in timing is unavailable.
    # --------------------------------------------------
    old_protocol_result = get_old_abf1_protocol_fallback(
        abf=abf,
        sweep_number=sweep_number,
        time=dataX,
    )

    if old_protocol_result is not None:
        dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source = old_protocol_result

        if timing_matches_file_timing(stim_start, stim_end):
            return old_protocol_result

    # --------------------------------------------------
    # 5. Last fallback: raw Current_in trace
    # --------------------------------------------------
    try:
        if stim_channel is None:
            return np.zeros_like(dataX), None, None, None, "None", "none"

        abf.setSweep(sweep_number, channel=stim_channel)
        current_trace = abf.sweepY.copy()

        stim_start, stim_end, stim_amp, stim_mode = detect_long_square_from_current_trace(
            time=dataX,
            current_trace=current_trace,
            baseline_interval=0.03,
            edge_buffer=0.02,
            min_step_duration=0.05,
            min_step_amp=5.0,
        )

        return (
            current_trace,
            stim_start,
            stim_end,
            stim_amp,
            stim_mode,
            "Current_in",
        )

    except Exception:
        return np.zeros_like(dataX), None, None, None, "None", "none"

def infer_current_in_timing_for_file(
    abf,
    stim_channel,
    min_step_duration=0.05,
    min_step_amp=5.0,
):
    """
    Detect stim timing from Current_in across all nonzero sweeps.
    Returns median stim_start and stim_end across detectable sweeps.
    """

    if stim_channel is None:
        return None, None

    starts = []
    ends = []

    for sweep_number in abf.sweepList:
        try:
            abf.setSweep(sweep_number, channel=stim_channel)
            time = abf.sweepX.copy()
            current_trace = abf.sweepY.copy()

            stim_start, stim_end, stim_amp, stim_mode = detect_long_square_from_current_trace(
                time=time,
                current_trace=current_trace,
                baseline_interval=0.03,
                edge_buffer=0.02,
                min_step_duration=min_step_duration,
                min_step_amp=min_step_amp,
            )

            if stim_start is not None and stim_end is not None and stim_mode == "Long":
                starts.append(stim_start)
                ends.append(stim_end)

        except Exception:
            continue

    if len(starts) == 0:
        return None, None

    return float(np.median(starts)), float(np.median(ends))

def get_protocol_amp_only(abf, sweep_number):
    """
    Return protocol-derived stimulation amplitude only.
    Does not return timing.
    """

    # ABF2 epoch-per-DAC
    eps = getattr(abf, "_epochPerDacSection", None)

    if eps is not None:
        dac_nums = np.asarray(eps.nDACNum)
        epoch_nums = np.asarray(eps.nEpochNum)
        first_levels = np.asarray(eps.fEpochInitLevel, dtype=float)
        delta_levels = np.asarray(eps.fEpochLevelInc, dtype=float)
        first_durations = np.asarray(eps.lEpochInitDuration, dtype=float)

        best = None

        for dac in sorted(set(dac_nums)):
            rows = []

            for idx in range(len(dac_nums)):
                if dac_nums[idx] != dac:
                    continue

                rows.append({
                    "epoch_num": int(epoch_nums[idx]),
                    "first_level": float(first_levels[idx]),
                    "delta_level": float(delta_levels[idx]),
                    "duration": float(first_durations[idx]),
                })

            candidates = [
                r for r in rows
                if r["duration"] > 0 and (abs(r["first_level"]) > 0 or abs(r["delta_level"]) > 0)
            ]

            if not candidates:
                continue

            stim_epoch = max(candidates, key=lambda r: r["duration"])
            score = stim_epoch["duration"] + 1000 * abs(stim_epoch["delta_level"])

            if best is None or score > best["score"]:
                best = {
                    "first_level_pa": stim_epoch["first_level"],
                    "delta_level_pa": stim_epoch["delta_level"],
                    "source": f"epochPerDac_DAC{dac}_amp",
                    "score": score,
                }

        if best is not None:
            amp = best["first_level_pa"] + best["delta_level_pa"] * sweep_number
            return float(amp), best["source"]

    # ABF1 long square protocol by name
    protocol = str(getattr(abf, "protocol", "")).lower()

    if "long square" in protocol:
        amp = -240.0 + 40.0 * sweep_number
        return float(amp), "old_abf1_long_square_amp"

    if "i-v curve" in protocol or "iv curve" in protocol:
        amp = -400.0 + 50.0 * sweep_number
        return float(amp), "old_abf1_iv_amp"

    return None, None

def get_stim_protocol_amp_current_timing(
    abf,
    sweep_number,
    stim_channel,
    dataX,
    file_current_start=None,
    file_current_end=None,
):
    """
    Use protocol/header for amplitude and file-level Current_in timing.

    This avoids sweep-by-sweep timing outliers from contaminating IPFX.
    """

    stim_amp, amp_source = get_protocol_amp_only(abf, sweep_number)

    if stim_amp is None:
        return None

    # Prefer file-level median timing
    sweep_start = file_current_start
    sweep_end = file_current_end

    if sweep_start is None or sweep_end is None:
        return None

    dataStim = np.zeros_like(dataX, dtype=float)
    dataStim[(dataX >= sweep_start) & (dataX <= sweep_end)] = stim_amp

    return (
        dataStim,
        sweep_start,
        sweep_end,
        stim_amp,
        "Long",
        f"{amp_source}+file_Current_in_timing",
    )

def get_protocol_stim_from_epoch_per_dac(abf, sweep_number, time, target_dac=None):
    """
    Reconstruct long-square stimulation from ABF2 _epochPerDacSection.

    This uses the protocol epochs, for example:
        Epoch A: baseline
        Epoch B: current step with first level + delta per sweep
        Epoch C: return to baseline

    Returns:
        dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source
    """

    eps = getattr(abf, "_epochPerDacSection", None)

    if eps is None:
        return None

    dac_nums = np.asarray(eps.nDACNum)
    epoch_nums = np.asarray(eps.nEpochNum)
    epoch_types = np.asarray(eps.nEpochType)
    first_levels = np.asarray(eps.fEpochInitLevel, dtype=float)
    delta_levels = np.asarray(eps.fEpochLevelInc, dtype=float)
    first_durations = np.asarray(eps.lEpochInitDuration, dtype=float)
    delta_durations = np.asarray(eps.lEpochDurationInc, dtype=float)

    # Candidate DACs
    if target_dac is None:
        candidate_dacs = sorted(set(dac_nums))
    else:
        candidate_dacs = [target_dac]

    best_info = None

    for dac in candidate_dacs:
        rows = []

        for idx in range(len(dac_nums)):
            if dac_nums[idx] != dac:
                continue

            rows.append({
                "epoch_num": int(epoch_nums[idx]),
                "epoch_type": int(epoch_types[idx]),
                "first_level": float(first_levels[idx]),
                "delta_level": float(delta_levels[idx]),
                "first_duration_samples": float(first_durations[idx]),
                "delta_duration_samples": float(delta_durations[idx]),
            })

        if not rows:
            continue

        rows = sorted(rows, key=lambda x: x["epoch_num"])

        # Find the stimulation epoch:
        # It should have nonzero duration and either nonzero level or nonzero delta.
        candidates = []
        for row in rows:
            duration = row["first_duration_samples"]
            level = row["first_level"]
            delta = row["delta_level"]

            if duration > 0 and (abs(level) > 0 or abs(delta) > 0):
                candidates.append(row)

        if not candidates:
            continue

        # For I-V curves, the main step is usually the longest nonzero/changing epoch.
        stim_epoch = max(candidates, key=lambda x: x["first_duration_samples"])

        pre_duration_samples = 0.0
        for row in rows:
            if row["epoch_num"] < stim_epoch["epoch_num"]:
                pre_duration_samples += row["first_duration_samples"]

        stim_amp = (
            stim_epoch["first_level"]
            + stim_epoch["delta_level"] * sweep_number
        )

        stim_start = pre_duration_samples / abf.sampleRate
        stim_end = stim_start + stim_epoch["first_duration_samples"] / abf.sampleRate

        # Prefer DACs with a long step and nonzero delta
        score = stim_epoch["first_duration_samples"] + 1000 * abs(stim_epoch["delta_level"])

        info = {
            "dac": dac,
            "stim_amp": stim_amp,
            "stim_start": stim_start,
            "stim_end": stim_end,
            "score": score,
            "rows": rows,
        }

        if best_info is None or info["score"] > best_info["score"]:
            best_info = info

    if best_info is None:
        return None

    dataStim = np.zeros_like(time, dtype=float)
    stim_idx = (time >= best_info["stim_start"]) & (time <= best_info["stim_end"])
    dataStim[stim_idx] = best_info["stim_amp"]

    return (
        dataStim,
        best_info["stim_start"],
        best_info["stim_end"],
        best_info["stim_amp"],
        "Long",
        f"epochPerDac_DAC{best_info['dac']}",
    )

def measure_step_current_from_recorded_trace(
    time,
    current_trace,
    stim_start,
    stim_end,
    baseline_interval=0.05,
    edge_buffer=0.02,
):
    """
    Measure current step amplitude from recorded Current_in channel.

    Uses:
        stim_amp = median current during stable step - median current before step

    edge_buffer avoids including onset/offset transients.
    """

    if stim_start is None or stim_end is None:
        return None

    time = np.asarray(time)
    current_trace = np.asarray(current_trace)

    # Baseline window before current step
    baseline_start = max(time[0], stim_start - baseline_interval)
    baseline_end = stim_start

    # Stable part of current step, avoiding onset/offset edges
    step_start = stim_start + edge_buffer
    step_end = stim_end - edge_buffer

    if step_end <= step_start:
        step_start = stim_start
        step_end = stim_end

    baseline_idx = (time >= baseline_start) & (time < baseline_end)
    step_idx = (time >= step_start) & (time <= step_end)

    if not np.any(baseline_idx) or not np.any(step_idx):
        return None

    baseline_current = np.median(current_trace[baseline_idx])
    step_current = np.median(current_trace[step_idx])

    return float(step_current - baseline_current)

def get_iv_curve_protocol_stim(
    sweep_number,
    time,
    first_level_pa=-250.0,
    delta_level_pa=50.0,
    pre_duration_s=0.051,
    step_duration_s=0.612,
):
    """
    Reconstruct I-V curve stimulation from protocol.

    Protocol:
        Epoch A: 0 pA for 51 ms
        Epoch B: starts at -250 pA, increments by 50 pA per sweep, duration 612 ms
        Epoch C: 0 pA for 1 ms

    Returns:
        dataStim, stim_start, stim_end, stim_amp, stim_mode, stim_source
    """

    stim_amp = first_level_pa + delta_level_pa * sweep_number

    stim_start = pre_duration_s
    stim_end = pre_duration_s + step_duration_s

    dataStim = np.zeros_like(time, dtype=float)
    stim_idx = (time >= stim_start) & (time <= stim_end)
    dataStim[stim_idx] = stim_amp

    return dataStim, stim_start, stim_end, stim_amp, "Long", "protocol"

def detect_long_square_from_current_trace(
    time,
    current_trace,
    baseline_interval=0.03,
    edge_buffer=0.02,
    min_step_duration=0.05,
    min_step_amp=5.0,
):
    """
    Detect a long-square current step from recorded Current_in.

    Returns:
        stim_start, stim_end, stim_amp, stim_mode

    stim_amp is measured as:
        median stable current during step - median baseline current
    """

    time = np.asarray(time)
    current_trace = np.asarray(current_trace)

    if time.size == 0 or current_trace.size == 0:
        return None, None, None, "None"

    # Use early sweep as baseline
    baseline_end_time = time[0] + baseline_interval
    baseline_idx = time < baseline_end_time

    if not np.any(baseline_idx):
        return None, None, None, "None"

    baseline_current = np.median(current_trace[baseline_idx])
    delta_current = current_trace - baseline_current

    # Robust threshold based on actual current deflection
    max_deflection = np.nanmax(np.abs(delta_current))

    if max_deflection < min_step_amp:
        return None, None, None, "None"

    threshold = max(min_step_amp, 0.25 * max_deflection)

    active_idx = np.where(np.abs(delta_current) >= threshold)[0]

    if active_idx.size == 0:
        return None, None, None, "None"

    # Find contiguous active regions
    breaks = np.where(np.diff(active_idx) > 1)[0]

    region_starts = np.r_[active_idx[0], active_idx[breaks + 1]]
    region_ends = np.r_[active_idx[breaks], active_idx[-1]]

    # Pick the longest active region
    durations = time[region_ends] - time[region_starts]
    best_region = np.argmax(durations)

    start_idx = region_starts[best_region]
    end_idx = region_ends[best_region]

    stim_start = float(time[start_idx])
    stim_end = float(time[end_idx])
    stim_duration = stim_end - stim_start

    if stim_duration < min_step_duration:
        return None, None, None, "None"

    # Measure stable part of step, avoiding onset/offset edges
    step_start = stim_start + edge_buffer
    step_end = stim_end - edge_buffer

    if step_end <= step_start:
        step_start = stim_start
        step_end = stim_end

    stable_idx = (time >= step_start) & (time <= step_end)

    if not np.any(stable_idx):
        return None, None, None, "None"

    step_current = np.median(current_trace[stable_idx])
    stim_amp = float(step_current - baseline_current)

    return stim_start, stim_end, stim_amp, "Long"

def infer_clamp_mode(abf):
    y_units = str(getattr(abf, "sweepUnitsY", "")).strip().lower()
    c_units = str(getattr(abf, "sweepUnitsC", "")).strip().lower()

    if y_units == "mv" and c_units == "pa":
        return "Current Clamp"
    if y_units == "pa" and c_units == "mv":
        return "Voltage Clamp"
    if y_units == "mv":
        return "Current Clamp"
    if y_units == "pa":
        return "Voltage Clamp"

    return "Unknown"

def summarize_cell_ephys_features(lsa_results):
    hero_keep_features = ['adapt', 'avg_rate', 'first_isi', 'isi_cv', 'latency', 'mean_isi', 'median_isi', 'stim_amp']
    rheo_keep_features = ['threshold_v', 'peak_v', 'trough_v', 'fast_trough_v', 'adp_v', 'width', 'upstroke_downstroke_ratio', 'peak_t', 'fast_trough_t', 'trough_t']
    overall_cell_keep_features = ['v_baseline', 'rheobase_i', 'fi_fit_slope', 'sag', 'vm_for_sag', 'input_resistance', 'tau']
    
    hero_small_dict = lsa_results['hero_sweep'][hero_keep_features]
    rheobase_sweep_index = lsa_results['rheobase_sweep'].name
    rheobase_sweep = lsa_results['spikes_set'][rheobase_sweep_index].iloc[0]
    
    rheo_spike_small_dict = rheobase_sweep[rheo_keep_features]
    rheo_first_isi = lsa_results['rheobase_sweep']['first_isi']
    rheo_spike_small_dict['rheo_first_isi'] = rheo_first_isi
    
    spike_comb_dict = {**hero_small_dict, **rheo_spike_small_dict}
    
    overall_cell_features = {x: lsa_results[x] for x in overall_cell_keep_features if x in lsa_results}
    final_cell_feature_dict = {**spike_comb_dict, **overall_cell_features}
    
    return(final_cell_feature_dict)

def find_file_level_channels(abf, zero_tol=1e-9, max_probe_sweeps=3):
    channels = abf.channelList if hasattr(abf, "channelList") else list(range(abf.channelCount))
    adc_names = getattr(abf, "adcNames", [])
    adc_units = getattr(abf, "adcUnits", [])

    response_scores = {ch: [] for ch in channels}
    stim_scores = {ch: [] for ch in channels}

    probe_sweeps = abf.sweepList[:min(max_probe_sweeps, len(abf.sweepList))]

    for i in probe_sweeps:
        for ch in channels:
            # probe response
            try:
                abf.setSweep(i, channel=ch)
                dataY = abf.sweepY.copy()
                y_range = float(np.nanmax(dataY) - np.nanmin(dataY))
            
                if ch < len(adc_units):
                    unit = str(adc_units[ch]).lower()
                    if unit != "mv":
                        continue   # Only voltage channels allowed
            
                score = y_range
            
                if ch < len(adc_names):
                    name = str(adc_names[ch]).lower()
            
                    if "iclamp" in name:
                        score += 2000
                    elif "vm" in name:
                        score += 1500
                    elif "volt" in name:
                        score += 1000
                    elif "mem" in name:
                        score += 800
            
                response_scores[ch].append(score)
            
            except Exception:
                pass

            # probe stim
            try:
                abf.setSweep(i, channel=ch)
                dataX = abf.sweepX.copy()
                dataStim = abf.sweepC.copy()
                stim_start, stim_end, stim_amp, stim_mode = stim_range(dataStim, dataX, threshold=zero_tol)

                if stim_start is not None and stim_end is not None:
                    stim_duration = stim_end - stim_start
                    stim_score = stim_duration + abs(stim_amp if stim_amp is not None else 0)

                    if stim_mode == "Long":
                        stim_score += 1000

                    stim_scores[ch].append(stim_score)

            except Exception:
                pass

    # best response channel
    valid_response = {ch: vals for ch, vals in response_scores.items() if len(vals) > 0}
    if not valid_response:
        return None

    response_channel = max(valid_response, key=lambda ch: np.mean(valid_response[ch]))

    # best stim channel
    valid_stim = {ch: vals for ch, vals in stim_scores.items() if len(vals) > 0}
    stim_channel = None
    if valid_stim:
        stim_channel = max(valid_stim, key=lambda ch: np.mean(valid_stim[ch]))

    # Override stim channel if a clear Current_in pA channel exists
    current_in_channel = None
    
    for ch, name in enumerate(adc_names):
        name_lower = str(name).lower()
        unit_lower = str(adc_units[ch]).lower() if ch < len(adc_units) else ""
    
        if "current_in" in name_lower and unit_lower == "pa":
            current_in_channel = ch
            break
    
    if current_in_channel is None:
        for ch, unit in enumerate(adc_units):
            if str(unit).lower() == "pa":
                current_in_channel = ch
                break
    
    if current_in_channel is not None:
        stim_channel = current_in_channel
        
    return {
        "response_channel": response_channel,
        "stim_channel": stim_channel,
    }

def parse_iv_protocol_from_text(protocol_text):
    """
    Parse Clampex ABF protocol/header text for I-V curve stimulation.

    Returns:
        dict with first_level_pa, delta_level_pa, pre_duration_s, step_duration_s, sweep_count
    """

    text = protocol_text

    # Sweep count
    sweep_match = re.search(r"Trial length:\s*.*?(\d+)\s+sweeps", text, re.IGNORECASE)
    sweep_count = int(sweep_match.group(1)) if sweep_match else None

    # Find Analog Output #1 block
    ao1_match = re.search(
        r"Analog Output #1\s+Waveform:(.*?)(?:Analog Output #2|Statistics measurements|Channel math|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if not ao1_match:
        return None

    ao1 = ao1_match.group(1)

    def extract_numeric_row(row_name):
        pattern = rf"{re.escape(row_name)}\s+(.+)"
        match = re.search(pattern, ao1, re.IGNORECASE)

        if not match:
            return None

        row_text = match.group(1)

        # Extract numbers, including negatives and decimals
        nums = re.findall(r"[-+]?\d*\.?\d+", row_text)
        return [float(x) for x in nums]

    first_levels = extract_numeric_row("First level (pA)")
    delta_levels = extract_numeric_row("Delta level (pA)")
    durations_ms = extract_numeric_row("First duration (ms)")

    if first_levels is None or delta_levels is None or durations_ms is None:
        return None

    # Pick the main stimulation epoch:
    # usually the epoch with nonzero duration and either nonzero first level or nonzero delta
    candidate_epochs = []

    for idx, duration_ms in enumerate(durations_ms):
        first_level = first_levels[idx] if idx < len(first_levels) else 0.0
        delta_level = delta_levels[idx] if idx < len(delta_levels) else 0.0

        if duration_ms > 0 and (abs(first_level) > 0 or abs(delta_level) > 0):
            candidate_epochs.append(idx)

    if not candidate_epochs:
        return None

    stim_epoch_idx = candidate_epochs[0]

    # Pre-duration is the sum of all earlier epoch durations
    pre_duration_ms = sum(durations_ms[:stim_epoch_idx])
    step_duration_ms = durations_ms[stim_epoch_idx]

    return {
        "sweep_count": sweep_count,
        "stim_epoch_idx": stim_epoch_idx,
        "first_level_pa": first_levels[stim_epoch_idx],
        "delta_level_pa": delta_levels[stim_epoch_idx],
        "pre_duration_s": pre_duration_ms / 1000.0,
        "step_duration_s": step_duration_ms / 1000.0,
    }

def get_protocol_stim_from_parsed_info(sweep_number, time, protocol_info):
    """
    Reconstruct command current using parsed protocol information.
    """

    first_level_pa = protocol_info["first_level_pa"]
    delta_level_pa = protocol_info["delta_level_pa"]
    pre_duration_s = protocol_info["pre_duration_s"]
    step_duration_s = protocol_info["step_duration_s"]

    stim_amp = first_level_pa + delta_level_pa * sweep_number

    stim_start = pre_duration_s
    stim_end = pre_duration_s + step_duration_s

    dataStim = np.zeros_like(time, dtype=float)
    stim_idx = (time >= stim_start) & (time <= stim_end)
    dataStim[stim_idx] = stim_amp

    return dataStim, stim_start, stim_end, stim_amp, "Long", "protocol"


def find_best_response_and_stim_channels(abf, sweep_number, zero_tol=1e-9):
    response_candidates = []
    stim_candidates = []

    channels = abf.channelList if hasattr(abf, "channelList") else list(range(abf.channelCount))
    adc_names = getattr(abf, "adcNames", [])
    adc_units = getattr(abf, "adcUnits", [])

    def response_score(ch, dataY):
        score = 0
        y_range = float(np.nanmax(dataY) - np.nanmin(dataY))

        if ch < len(adc_units):
            unit = str(adc_units[ch]).lower()
            if unit == "mv":
                score += 100
            elif unit == "pa":
                score += 20

        if ch < len(adc_names):
            name = str(adc_names[ch]).lower()
            if "mem" in name or "volt" in name or "primary" in name:
                score += 50

        score += y_range
        return score

    for ch in channels:
        try:
            abf.setSweep(sweep_number, channel=ch)
            dataX = abf.sweepX.copy()
            dataY = abf.sweepY.copy()
            y_range = float(np.nanmax(dataY) - np.nanmin(dataY))
            score = response_score(ch, dataY)

            response_candidates.append({
                "channel": ch,
                "dataX": dataX,
                "dataY": dataY,
                "y_range": y_range,
                "score": score,
            })
        except Exception as e:
            print(f"sweep {sweep_number}, channel {ch} response failed: {e}")
            continue

        try:
            dataStim = abf.sweepC.copy()
            stim_start, stim_end, stim_amp, stim_mode = stim_range(
                dataStim, dataX, threshold=zero_tol
            )

            if stim_start is not None and stim_end is not None:
                stim_range_val = float(np.nanmax(dataStim) - np.nanmin(dataStim))
                stim_duration = stim_end - stim_start

                stim_candidates.append({
                    "channel": ch,
                    "dataStim": dataStim,
                    "stim_range_val": stim_range_val,
                    "stim_start": stim_start,
                    "stim_end": stim_end,
                    "stim_amp": stim_amp,
                    "stim_mode": stim_mode,
                    "stim_duration": stim_duration,
                })

        except Exception as e:
            print(f"sweep {sweep_number}, channel {ch} stim failed: {e}")

    if not response_candidates:
        return None

    response_best = max(response_candidates, key=lambda c: c["score"])

    long_stim_candidates = [c for c in stim_candidates if c["stim_mode"] == "Long"]

    if long_stim_candidates:
        stim_best = max(
            long_stim_candidates,
            key=lambda c: (c["stim_duration"], abs(c["stim_amp"]) if c["stim_amp"] is not None else -np.inf)
        )
    elif stim_candidates:
        stim_best = max(
            stim_candidates,
            key=lambda c: (c["stim_duration"], abs(c["stim_amp"]) if c["stim_amp"] is not None else -np.inf)
        )
    else:
        stim_best = {
            "channel": None,
            "dataStim": np.zeros_like(response_best["dataX"], dtype=float),
            "stim_start": None,
            "stim_end": None,
            "stim_amp": 0.0,
            "stim_mode": "None",
            "stim_duration": 0.0,
        }

    return {
        "response_channel": response_best["channel"],
        "stim_channel": stim_best["channel"],
        "dataX": response_best["dataX"],
        "dataY": response_best["dataY"],
        "dataStim": stim_best["dataStim"],
        "stim_start": stim_best["stim_start"],
        "stim_end": stim_best["stim_end"],
        "stim_amp": stim_best["stim_amp"],
        "stim_mode": stim_best["stim_mode"],
    }