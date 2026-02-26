"""
Metrics computation for the Unsupervised Spindle Detector evaluation.

For each subject, channels are matched by normalized name (case-insensitive, 
stripping 'EEG ' prefix). Metrics are computed at IoU threshold tau=0.2.

Ground truth files: /mnt/project/{1,2,3,4,5_1,5_2,6}.csv
Detection files:    /mnt/user-data/uploads/start_end_per_channel_{1,2,3,4,5_1,5_2,6}.csv

Special cases handled:
  - GT3: some rows have Start > End (swapped by annotation error) -> swap to fix
  - GT4: channels 'P<' and 'P6' treated as annotation artefacts -> kept but 
         may not match any detection channel
  - Patient 5: two N2 phases (5_1, 5_2) merged before evaluation
  - Channel name normalisation: strip 'EEG ' prefix, uppercase
  - Only channels present in BOTH GT and detections are evaluated
"""

import pandas as pd
import numpy as np
from itertools import product

IOU_THRESH = 0.2
SUBJECTS = ['1', '2', '3', '4', '5_1', '5_2', '6']


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def norm_ch(ch: str) -> str:
    ch = str(ch).strip()
    if ch.upper().startswith('EEG '):
        ch = ch[4:]
    return ch.upper()


def compute_iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, a_start) - min(a_start, b_start) + \
            max(b_end, b_start) - min(b_start, b_end) - inter
    # simpler:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_events(gt_events, det_events, iou_thresh=IOU_THRESH):
    """
    Greedy matching: each GT event matched to at most one detection, 
    each detection to at most one GT event, by descending IoU.
    Returns (TP, FP, FN, list_of_matched_ious).
    """
    if len(gt_events) == 0 and len(det_events) == 0:
        return 0, 0, 0, []
    if len(gt_events) == 0:
        return 0, len(det_events), 0, []
    if len(det_events) == 0:
        return 0, 0, len(gt_events), []

    # Build IoU matrix
    iou_matrix = np.zeros((len(gt_events), len(det_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ds, de) in enumerate(det_events):
            iou_matrix[i, j] = compute_iou(gs, ge, ds, de)

    matched_gt  = set()
    matched_det = set()
    matched_ious = []

    # Greedy: pick highest IoU pairs first
    flat_indices = np.argsort(-iou_matrix.ravel())
    for idx in flat_indices:
        i, j = divmod(idx, len(det_events))
        if iou_matrix[i, j] < iou_thresh:
            break
        if i in matched_gt or j in matched_det:
            continue
        matched_gt.add(i)
        matched_det.add(j)
        matched_ious.append(iou_matrix[i, j])

    TP = len(matched_gt)
    FN = len(gt_events) - TP
    FP = len(det_events) - len(matched_det)
    return TP, FP, FN, matched_ious


def prf(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --------------------------------------------------------------------------
# load & clean
# --------------------------------------------------------------------------

def load_gt(subject_id):
    gt = pd.read_csv(f'/mnt/project/{subject_id}.csv')
    gt.columns = [c.strip() for c in gt.columns]
    gt['Channel'] = gt['Channel'].apply(norm_ch)
    # Fix swapped start/end (annotation artefact in some files)
    mask = gt['Start_Time(s)'] > gt['End_Time(s)']
    gt.loc[mask, ['Start_Time(s)', 'End_Time(s)']] = \
        gt.loc[mask, ['End_Time(s)', 'Start_Time(s)']].values
    # Drop zero-duration events
    gt = gt[gt['End_Time(s)'] > gt['Start_Time(s)']].copy()
    return gt


def load_det(subject_id):
    det = pd.read_csv(f'/mnt/user-data/uploads/start_end_per_channel_{subject_id}.csv')
    det.columns = [c.strip() for c in det.columns]
    det['Channel'] = det['Canale'].apply(norm_ch)
    det = det[det['End_Time(s)'] > det['Start_Time(s)']].copy()
    return det


# --------------------------------------------------------------------------
# per-channel metrics for a single recording
# --------------------------------------------------------------------------

def evaluate_recording(subject_id):
    gt  = load_gt(subject_id)
    det = load_det(subject_id)

    gt_channels  = set(gt['Channel'].unique())
    det_channels = set(det['Channel'].unique())
    common_channels = gt_channels & det_channels

    rows = []
    for ch in sorted(common_channels):
        gt_ev  = list(zip(gt[gt['Channel']  == ch]['Start_Time(s)'],
                          gt[gt['Channel']  == ch]['End_Time(s)']))
        det_ev = list(zip(det[det['Channel'] == ch]['Start_Time(s)'],
                          det[det['Channel'] == ch]['End_Time(s)']))

        TP, FP, FN, ious = match_events(gt_ev, det_ev)
        prec, rec, f1    = prf(TP, FP, FN)
        miou = float(np.mean(ious)) if ious else 0.0

        rows.append({
            'Subject':   subject_id,
            'Channel':   ch,
            'GT_Events': len(gt_ev),
            'Det_Events': len(det_ev),
            'TP': TP, 'FP': FP, 'FN': FN,
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1':        round(f1,   4),
            'mIoU':      round(miou, 4),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():

    all_records = []

    # Standard subjects (one recording each)
    for sid in ['1', '2', '3', '4', '6']:
        df = evaluate_recording(sid)
        all_records.append(df)

    # Patient 5: two N2 phases evaluated separately then averaged
    df5_1 = evaluate_recording('5_1')
    df5_2 = evaluate_recording('5_2')
    # Common channels across both phases
    common5 = set(df5_1['Channel']) & set(df5_2['Channel'])
    merged5_rows = []
    for ch in sorted(common5):
        r1 = df5_1[df5_1['Channel'] == ch].iloc[0]
        r2 = df5_2[df5_2['Channel'] == ch].iloc[0]
        TP = r1['TP'] + r2['TP']
        FP = r1['FP'] + r2['FP']
        FN = r1['FN'] + r2['FN']
        prec, rec, f1 = prf(TP, FP, FN)
        # mIoU weighted by TP
        tp_total = r1['TP'] + r2['TP']
        if tp_total > 0:
            miou = (r1['mIoU']*r1['TP'] + r2['mIoU']*r2['TP']) / tp_total
        else:
            miou = 0.0
        merged5_rows.append({
            'Subject':   '5',
            'Channel':   ch,
            'GT_Events': r1['GT_Events'] + r2['GT_Events'],
            'Det_Events': r1['Det_Events'] + r2['Det_Events'],
            'TP': TP, 'FP': FP, 'FN': FN,
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1':        round(f1,   4),
            'mIoU':      round(miou, 4),
        })
    df5 = pd.DataFrame(merged5_rows)
    # Also include channels only in one phase (treated with zeros for the other)
    only1 = set(df5_1['Channel']) - common5
    only2 = set(df5_2['Channel']) - common5
    for ch in sorted(only1):
        r = df5_1[df5_1['Channel'] == ch].iloc[0].copy()
        r['Subject'] = '5'
        merged5_rows.append(r)
    for ch in sorted(only2):
        r = df5_2[df5_2['Channel'] == ch].iloc[0].copy()
        r['Subject'] = '5'
        merged5_rows.append(r)
    df5 = pd.DataFrame(merged5_rows)
    all_records.append(df5)

    results = pd.concat(all_records, ignore_index=True)

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------

    print("=" * 80)
    print("PER-SUBJECT MACRO-AVERAGE (averaged over matched channels)")
    print("=" * 80)
    subj_summary = results.groupby('Subject').agg(
        Channels=('Channel','count'),
        GT_Events=('GT_Events','sum'),
        Det_Events=('Det_Events','sum'),
        Precision=('Precision','mean'),
        Recall=('Recall','mean'),
        F1=('F1','mean'),
        mIoU=('mIoU','mean'),
    ).round(4)
    print(subj_summary)
    print()

    print("=" * 80)
    print("OVERALL (macro-average across all subject-channel pairs)")
    print("=" * 80)
    print(f"  Mean Precision : {results['Precision'].mean():.4f}  ± {results['Precision'].std():.4f}")
    print(f"  Mean Recall    : {results['Recall'].mean():.4f}  ± {results['Recall'].std():.4f}")
    print(f"  Mean F1        : {results['F1'].mean():.4f}  ± {results['F1'].std():.4f}")
    print(f"  Mean mIoU      : {results['mIoU'].mean():.4f}  ± {results['mIoU'].std():.4f}")
    print(f"  Total GT events evaluated: {results['GT_Events'].sum()}")
    print(f"  Total Det events:          {results['Det_Events'].sum()}")
    print()

    print("=" * 80)
    print("PER-CHANNEL MACRO-AVERAGE (averaged over subjects)")
    print("=" * 80)
    ch_summary = results.groupby('Channel').agg(
        N_Subjects=('Subject','count'),
        Precision=('Precision','mean'),
        Recall=('Recall','mean'),
        F1=('F1','mean'),
        mIoU=('mIoU','mean'),
    ).round(4).sort_values('F1', ascending=False)
    print(ch_summary)
    print()

    print("=" * 80)
    print("FULL PER-SUBJECT-CHANNEL TABLE")
    print("=" * 80)
    print(results.sort_values(['Subject','Channel']).to_string(index=False))

    # Save
    results.to_csv('Data/evaluations/unsupervised_metrics_full.csv', index=False)
    subj_summary.to_csv('Data/evaluations/unsupervised_metrics_per_subject.csv')
    ch_summary.to_csv('Data/evaluations/unsupervised_metrics_per_channel.csv')
    print("\nSaved: unsupervised_metrics_full.csv, _per_subject.csv, _per_channel.csv")


if __name__ == '__main__':
    main()