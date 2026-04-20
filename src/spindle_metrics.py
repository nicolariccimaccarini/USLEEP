"""
spindle_metrics.py
==================
Calcola le seguenti metriche per ogni fase N2 presente in un file CSV di spindles EEG:

  - Tasso degli spindles  (spindles/minuto, per canale e globale)
  - Durata degli spindles (min, max, media, std — per canale e globale)
  - ISI (Inter-Spindle Interval)
      Definizione: tempo (s) tra la FINE di un fuso e l'INIZIO del successivo nello STESSO canale.
      Calcolo globale: per ogni canale con almeno 10 fusi si calcola la media del 10% degli ISI
      più bassi (periodo refrattario); il valore globale è la media di questi su tutti i canali
      idonei. (Kwon et al., 2023)
  - Ritardo interemisferico (Interhemispheric spindle lag)
      Definizione: tempo (s) tra il PRIMO fuso rilevato in un canale di un emisfero e il PRIMO
      fuso rilevato nel canale OMOLOGO dell'emisfero opposto, calcolato per ogni coppia di canali
      frontopolari o centroparietali (FP1/FP2, F3/F4, C3/C4, P3/P4).
      Applicabile solo se i fusi sono presenti in almeno 2 canali frontopolari o centroparietali
      in ENTRAMBI gli emisferi; altrimenti NA.
      Il valore globale è la media dei ritardi su tutte le coppie disponibili (per età).
      NA è fisiologico fino al 30% dei fusi. (Grigg-Damberger et al., 2007)

Formato atteso del CSV (può contenere N fasi N2 affiancate, ognuna su 5 colonne):
  Riga 1 : Inizio NREM, HH:MM:SS.mmm, Fine NREM, HH:MM:SS.mmm, [vuoto], Inizio NREM, ...
  Riga 2 : vuota
  Riga 3 : Canale, Start_time, End_time, [vuoto], [vuoto], Canale, Start_time, End_time, ...
  Riga 4+: dati spindles

Output: per ogni fase N2 viene generato un file CSV separato
  <nome_originale>_<N>.csv  (es. PAZIENTE_1.csv, PAZIENTE_2.csv, ...)
con i dati originali + un blocco di metriche in fondo.

Opzioni da riga di comando:
  --global-only   Stampa nel CSV solo le metriche globali (omette la tabella per canale)
"""

import os
import sys
import math
import warnings
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configurazione emisferi
# ─────────────────────────────────────────────
EMISFERO_SX = {"FP1", "F7", "F3", "T3", "C3", "T5", "P3", "O1"}
EMISFERO_DX = {"FP2", "F8", "F4", "T4", "C4", "T6", "P4", "O2"}

# Coppie omologhe valide per il ritardo interemisferico
# (solo canali frontopolari e centroparietali, come da Grigg-Damberger et al. 2007)
COPPIE_INTEREMISFERICHE = [
    ("FP1", "FP2"),
    ("F3",  "F4"),
    ("C3",  "C4"),
    ("P3",  "P4"),
]

# Soglia minima fusi per calcolo ISI refrattario
ISI_MIN_FUSI = 10
# Percentile basso per ISI (10% degli ISI più bassi)
ISI_PERCENTILE = 10


# ─────────────────────────────────────────────
# Utility: parsing del tempo
# ─────────────────────────────────────────────
def parse_time(t) -> float | None:
    """
    Converte un timestamp HH:MM:SS.mmm in secondi dall'inizio della giornata.
    Restituisce None se il valore non è parsabile.
    """
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    s = str(t).strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            continue
    return None


def seconds_to_hms(sec: float) -> str:
    """Formatta secondi in HH:MM:SS.mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ─────────────────────────────────────────────
# Parsing del file CSV
# ─────────────────────────────────────────────
def parse_input_file(filepath: str) -> list[dict]:
    """
    Legge il CSV e restituisce una lista di dizionari, uno per fase N2:
      {
        'phase_num'   : int,
        'inizio_nrem' : float  (secondi),
        'fine_nrem'   : float  (secondi),
        'spindles'    : pd.DataFrame con colonne [Canale, Start_s, End_s, Duration_s]
      }
    """
    df_raw = pd.read_csv(filepath, header=None, dtype=str)

    # ── Riga 0: individua i blocchi di fase N2 ──────────────────────────
    row0 = df_raw.iloc[0].tolist()
    phases_meta = []  # (col_start, inizio_nrem_sec, fine_nrem_sec)

    col = 0
    while col < len(row0):
        val = row0[col]
        if isinstance(val, str) and "inizio" in val.lower():
            try:
                inizio = parse_time(row0[col + 1])
                fine   = parse_time(row0[col + 3])
                phases_meta.append((col, inizio, fine))
                col += 5
            except IndexError:
                col += 1
        else:
            col += 1

    if not phases_meta:
        raise ValueError(
            f"Nessuna riga 'Inizio NREM / Fine NREM' trovata in {filepath}.\n"
            "Assicurati che il file abbia la struttura attesa."
        )

    # ── Righe 3+: dati spindles ─────────────────────────────────────────
    results = []
    for phase_idx, (col_start, inizio_nrem, fine_nrem) in enumerate(phases_meta):
        cols = [col_start, col_start + 1, col_start + 2]
        # Prende solo le colonne del blocco (dalla riga 3 in poi)
        block = df_raw.iloc[3:, cols].copy()
        block.columns = ["Canale", "Start_time", "End_time"]
        block = block.reset_index(drop=True)

        valid_rows = []
        for _, row in block.iterrows():
            canale = str(row["Canale"]).strip() if pd.notna(row["Canale"]) else ""
            if canale in ("", "nan", "None"):
                continue

            start_s = parse_time(row["Start_time"])
            end_s   = parse_time(row["End_time"])

            # Skip silenzioso se i tempi non sono parsabili o end <= start
            if start_s is None or end_s is None:
                continue
            if end_s <= start_s:
                continue

            valid_rows.append({
                "Canale":     canale,
                "Start_s":    start_s,
                "End_s":      end_s,
                "Duration_s": end_s - start_s,
            })

        spindles_df = pd.DataFrame(valid_rows)

        results.append({
            "phase_num":   phase_idx + 1,
            "inizio_nrem": inizio_nrem,
            "fine_nrem":   fine_nrem,
            "spindles":    spindles_df,
        })

    return results


# Ordine standard dei 19 canali (come da montaggio 10-20 usato nella distribuzione spaziale)
CANALI_STANDARD_ORDINE = [
    "FP1", "FP2",
    "F7",  "F3",  "FZ",  "F4",  "F8",
    "T3",  "C3",  "CZ",  "C4",  "T4",
    "T5",  "P3",  "PZ",  "P4",  "T6",
    "O1",  "O2",
]


# ─────────────────────────────────────────────
# Calcolo metriche
# ─────────────────────────────────────────────

N_CANALI_STANDARD = 19  # numero fisso di canali per normalizzazione del tasso


def calc_tasso(spindles_df: pd.DataFrame, durata_nrem_min: float,
               canale: str | None = None) -> float | None:
    """
    Tasso degli spindles (#/min).

    Definizione (Kwon et al., 2023):
      Tasso = N_fusi_totali / durata_N2_min / 19_canali

    Modalità GLOBALE (canale=None):
      Somma tutti gli spindles su tutti i canali, divide per durata e per 19.
      Questo equivale a: quanti spindles per minuto per canale in media.

    Modalità per CANALE (canale!=None):
      N_spindles_canale / durata_N2_min  (spindles del singolo canale per minuto).
    """
    if durata_nrem_min <= 0:
        return None

    if canale:
        sub = spindles_df[spindles_df["Canale"] == canale]
        if sub.empty:
            return None
        return len(sub) / durata_nrem_min
    else:
        if spindles_df.empty:
            return None
        return len(spindles_df) / durata_nrem_min / N_CANALI_STANDARD


def calc_durata(spindles_df: pd.DataFrame,
                canale: str | None = None) -> dict | None:
    """Min, Max, Media, Std della durata in secondi."""
    if canale:
        sub = spindles_df[spindles_df["Canale"] == canale]["Duration_s"]
    else:
        sub = spindles_df["Duration_s"]
    if sub.empty:
        return None
    return {
        "min": sub.min(),
        "max": sub.max(),
        "mean": sub.mean(),
        "std":  sub.std() if len(sub) > 1 else 0.0,
    }



def calc_spindle_time_percentage(spindles_df: pd.DataFrame,
                                  durata_nrem_s: float) -> dict | None:
    """
    Percentuale di Spindle Time (Spindle Time %).

    Definizione corretta:
      Tempo totale coperto da almeno uno spindle (su qualsiasi canale) / durata_N2 * 100

    Metodo: union of intervals.
      Gli spindles simultanei su canali diversi rappresentano lo stesso evento biologico
      visto da elettrodi diversi. Sommare le durate canale per canale conterebbe più volte
      lo stesso secondo di attività cerebrale, gonfiando il risultato.
      Si uniscono quindi tutti gli intervalli [Start_s, End_s] di tutti i canali,
      si fondono quelli sovrapposti (merge), e si misura il tempo totale coperto.

    Restituisce un dict con:
      'percentuale'     : float (%)
      'tempo_coperto_s' : float (secondi effettivamente coperti dopo merge)
      'n_intervalli'    : int   (numero di intervalli distinti dopo merge)
    """
    if durata_nrem_s is None or durata_nrem_s <= 0 or spindles_df.empty:
        return None

    # Ordina tutti gli intervalli per Start_s e fai il merge
    intervals = sorted(zip(spindles_df["Start_s"].values, spindles_df["End_s"].values))
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:          # sovrapposizione: estendi
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([start, end])

    tempo_coperto = sum(e - s for s, e in merged)
    return {
        "percentuale":     tempo_coperto / durata_nrem_s * 100,
        "tempo_coperto_s": tempo_coperto,
        "n_intervalli":    len(merged),
    }


def calc_distribuzione_spaziale(spindles_df: pd.DataFrame) -> dict:
    """
    Distribuzione spaziale dei fusi del sonno.

    Per ogni canale del montaggio standard (19 canali):
      % canale = N_spindles_canale / N_spindles_totali * 100

    Restituisce un dict {canale: percentuale | None} per tutti i 19 canali standard.
    I canali non presenti nel dataframe ricevono 0.0%.
    """
    n_tot = len(spindles_df)
    if n_tot == 0:
        return {ch: 0.0 for ch in CANALI_STANDARD_ORDINE}

    conteggi = spindles_df["Canale"].value_counts()
    return {
        ch: round(conteggi.get(ch, 0) / n_tot * 100, 2)
        for ch in CANALI_STANDARD_ORDINE
    }


def calc_isi(spindles_df: pd.DataFrame,
             canale: str | None = None) -> dict | None:
    """
    ISI (Inter-Spindle Interval) — Periodo refrattario.

    Per ogni canale con almeno ISI_MIN_FUSI fusi:
      1. Calcola tutti gli ISI del canale = End_s[i] → Start_s[i+1] (solo valori >= 0)
      2. Prende la media del 10% degli ISI più bassi (periodo refrattario del canale)

    Modalità canale singolo  (canale != None):
      Restituisce il valore refrattario del singolo canale, oppure None se fusi < soglia.

    Modalità globale (canale == None):
      Calcola il valore refrattario per ogni canale idoneo, poi ne fa la media.
      Restituisce un dict con:
        'globale'        : media tra canali idonei
        'per_canale'     : {nome_canale: valore o None}
        'canali_idonei'  : numero di canali usati nel calcolo globale
    """
    def _refrattario_canale(df_ch: pd.DataFrame) -> float | None:
        """Media del 10% degli ISI più bassi per un singolo canale."""
        df_ch = df_ch.sort_values("Start_s")
        if len(df_ch) < ISI_MIN_FUSI:
            return None
        starts = df_ch["Start_s"].values
        ends   = df_ch["End_s"].values
        isis = [starts[i] - ends[i - 1] for i in range(1, len(starts))
                if starts[i] - ends[i - 1] >= 0]
        if not isis:
            return None
        # 10% più bassi (minimo 1 valore)
        n_low = max(1, int(np.ceil(len(isis) * ISI_PERCENTILE / 100)))
        isis_sorted = sorted(isis)
        return float(np.mean(isis_sorted[:n_low]))

    if canale is not None:
        sub = spindles_df[spindles_df["Canale"] == canale]
        return _refrattario_canale(sub)

    # Globale
    canali = spindles_df["Canale"].unique()
    per_canale = {}
    valori_globali = []
    for ch in sorted(canali):
        sub = spindles_df[spindles_df["Canale"] == ch]
        val = _refrattario_canale(sub)
        per_canale[ch] = val
        if val is not None:
            valori_globali.append(val)

    return {
        "globale":       float(np.mean(valori_globali)) if valori_globali else None,
        "per_canale":    per_canale,
        "canali_idonei": len(valori_globali),
    }


def _spindles_simultanei(sp_a: pd.DataFrame, sp_b: pd.DataFrame) -> list[tuple]:
    """
    Trova tutte le coppie di spindles sovrapposti temporalmente tra il canale A e il canale B.

    Due spindles si considerano simultanei se si sovrappongono, cioè se:
        Start_A < End_B  AND  Start_B < End_A
    (condizione standard di overlap tra due intervalli)

    Per ogni coppia sovrapposta restituisce (Start_A, Start_B) in secondi,
    in modo che il chiamante possa calcolare il ritardo come Start_B - Start_A.

    Restituisce una lista di tuple (start_a, start_b).
    """
    risultati = []
    for _, row_a in sp_a.iterrows():
        for _, row_b in sp_b.iterrows():
            # Overlap: l'uno inizia prima che l'altro finisca, e viceversa
            if row_a["Start_s"] < row_b["End_s"] and row_b["Start_s"] < row_a["End_s"]:
                risultati.append((row_a["Start_s"], row_b["Start_s"]))
    return risultati


def calc_ritardo_interemisferico(spindles_df: pd.DataFrame) -> dict:
    """
    Ritardo interemisferico (Interhemispheric spindle lag).

    Algoritmo:
      - Considera solo le coppie frontopolari/centroparietali: FP1/FP2, F3/F4, C3/C4, P3/P4
      - Applicabile solo se fusi presenti in >= 2 canali frontopolari o centroparietali
        in ENTRAMBI gli emisferi (condizione minima)
      - Per ogni coppia omologa (es. C3/C4):
          Si cercano tutte le coppie di spindles SOVRAPPOSTI temporalmente
          (uno inizia mentre l'altro è già presente sull'omologo).
          Per ogni coppia sovrapposta: ritardo = Start_DX - Start_SX
          (positivo → SX anticipa DX; negativo → DX anticipa SX)
          Il ritardo della coppia è la media di tutti i ritardi delle coppie sovrapposte.
          Se non esiste alcuna sovrapposizione → NA per quella coppia (spindles asimmetrici)
      - Il valore globale è la media dei ritardi non-NA su tutte le coppie disponibili
      - % NA = proporzione di coppie senza sovrapposizioni (fisiologico ≤ 30%)

    Restituisce un dict con:
      'globale'           : float o None  (media ritardi non-NA tra coppie)
      'per_coppia'        : {coppia_str: {'ritardo': float|None,
                                          'n_sovrapposizioni': int,
                                          'ritardi_singoli': list[float]}}
      'na_count'          : numero coppie NA (nessuna sovrapposizione)
      'tot_coppie'        : numero coppie verificate
      'perc_na'           : percentuale NA
      'applicabile'       : bool
      'nota'              : stringa descrittiva
    """
    canali_presenti = set(spindles_df["Canale"].unique())

    # Canali frontopolari/centroparietali per emisfero
    sx_fp_cp = {"FP1", "F3", "C3", "P3"}
    dx_fp_cp = {"FP2", "F4", "C4", "P4"}

    canali_sx_ok = canali_presenti & sx_fp_cp
    canali_dx_ok = canali_presenti & dx_fp_cp

    # Condizione minima: >= 2 canali idonei per emisfero
    applicabile = len(canali_sx_ok) >= 2 and len(canali_dx_ok) >= 2

    if not applicabile:
        return {
            "globale":     None,
            "per_coppia":  {f"{sx}/{dx}": {"ritardo": None,
                                            "n_sovrapposizioni": 0,
                                            "ritardi_singoli": []}
                            for sx, dx in COPPIE_INTEREMISFERICHE},
            "na_count":    len(COPPIE_INTEREMISFERICHE),
            "tot_coppie":  len(COPPIE_INTEREMISFERICHE),
            "perc_na":     100.0,
            "applicabile": False,
            "nota": (f"NA — condizione non soddisfatta: "
                     f"canali SX idonei={sorted(canali_sx_ok)}, "
                     f"canali DX idonei={sorted(canali_dx_ok)} "
                     f"(richiesti >= 2 per emisfero)"),
        }

    per_coppia = {}
    ritardi_validi = []

    # Contatori per NA% basata su spindles (non su coppie di canali)
    tot_spindles_nelle_coppie = 0  # spindles totali sui canali delle 4 coppie
    tot_spindles_accoppiati   = 0  # spindles che hanno trovato un omologo sovrapposto

    for sx, dx in COPPIE_INTEREMISFERICHE:
        sp_sx = spindles_df[spindles_df["Canale"] == sx].copy()
        sp_dx = spindles_df[spindles_df["Canale"] == dx].copy()
        n_sx  = len(sp_sx)
        n_dx  = len(sp_dx)
        tot_spindles_nelle_coppie += n_sx + n_dx

        # Se uno dei due canali non ha spindles → tutti asimmetrici
        if sp_sx.empty or sp_dx.empty:
            per_coppia[f"{sx}/{dx}"] = {
                "ritardo":           None,
                "n_sovrapposizioni": 0,
                "n_sx":              n_sx,
                "n_dx":              n_dx,
                "ritardi_singoli":   [],
            }
            continue

        # Trova tutte le coppie sovrapposte (SX rispetto a DX)
        # ritardo = Start_DX - Start_SX  (positivo → SX anticipa DX)
        coppie_sovrapposte = _spindles_simultanei(sp_sx, sp_dx)

        if not coppie_sovrapposte:
            # Nessuna sovrapposizione → spindles asimmetrici → NA
            per_coppia[f"{sx}/{dx}"] = {
                "ritardo":           None,
                "n_sovrapposizioni": 0,
                "n_sx":              n_sx,
                "n_dx":              n_dx,
                "ritardi_singoli":   [],
            }
            continue

        ritardi_coppia = [start_dx - start_sx
                          for start_sx, start_dx in coppie_sovrapposte]
        ritardo_medio  = float(np.mean(ritardi_coppia))

        per_coppia[f"{sx}/{dx}"] = {
            "ritardo":           ritardo_medio,
            "n_sovrapposizioni": len(coppie_sovrapposte),
            "n_sx":              n_sx,
            "n_dx":              n_dx,
            "ritardi_singoli":   ritardi_coppia,
        }
        ritardi_validi.append(ritardo_medio)
        # Ogni sovrapposizione "consuma" 2 spindles (uno SX + uno DX)
        tot_spindles_accoppiati += len(coppie_sovrapposte) * 2

    # NA% = spindles non accoppiati / spindles totali nei canali delle coppie
    # Formula tutor: (tot_spindles_nelle_coppie - coppie_trovate*2) / tot_spindles_nelle_coppie
    if tot_spindles_nelle_coppie > 0:
        spindles_na = tot_spindles_nelle_coppie - tot_spindles_accoppiati
        perc_na     = spindles_na / tot_spindles_nelle_coppie * 100
    else:
        spindles_na = 0
        perc_na     = 100.0

    nota = (f"% NA = {perc_na:.1f}% "
            f"({'fisiologico' if perc_na <= 30 else 'ATTENZIONE: > 30%'}) "
            f"| spindles accoppiati: {tot_spindles_accoppiati}, "
            f"non accoppiati: {spindles_na}, "
            f"totale nei canali delle coppie: {tot_spindles_nelle_coppie}")

    return {
        "globale":                    float(np.mean(ritardi_validi)) if ritardi_validi else None,
        "per_coppia":                 per_coppia,
        "tot_spindles_nelle_coppie":  tot_spindles_nelle_coppie,
        "tot_spindles_accoppiati":    tot_spindles_accoppiati,
        "spindles_na":                spindles_na,
        "perc_na":                    perc_na,
        "applicabile":                True,
        "nota":                       nota,
    }


# ─────────────────────────────────────────────
# Costruzione output CSV
# ─────────────────────────────────────────────
def build_output_rows(phase: dict, global_only: bool = False) -> list[list]:
    """
    Costruisce le righe da aggiungere in fondo al CSV originale.
    Se global_only=True stampa solo la sezione GLOBALE, omettendo la tabella per canale.
    """
    spindles_df   = phase["spindles"]
    inizio_nrem   = phase["inizio_nrem"]
    fine_nrem     = phase["fine_nrem"]
    durata_nrem_s = fine_nrem - inizio_nrem if (inizio_nrem and fine_nrem) else None
    durata_nrem_m = durata_nrem_s / 60.0 if durata_nrem_s else None

    rows = []

    # ── intestazione ────────────────────────────────────────────────────
    rows.append([])
    rows.append(["=== METRICHE ==="])
    rows.append(["Durata fase N2",
                 f"{durata_nrem_s:.3f} s" if durata_nrem_s else "N/A",
                 f"({durata_nrem_m:.2f} min)" if durata_nrem_m else ""])
    rows.append([])

    # ── metriche GLOBALI ────────────────────────────────────────────────
    rows.append(["--- GLOBALE ---"])
    rows.append(["Metrica", "Valore", "Note"])

    # Tasso
    tasso_g = calc_tasso(spindles_df, durata_nrem_m or 0)
    n_tot   = len(spindles_df)
    rows.append([
        "Tasso spindles (sp/min/canale)",
        f"{tasso_g:.4f}" if tasso_g is not None else "N/A",
        f"= {n_tot} spindles / {durata_nrem_m:.4f} min / {N_CANALI_STANDARD} canali",
    ])

    # Durata
    dur_g = calc_durata(spindles_df)
    if dur_g:
        rows.append(["Durata media (s)", f"{dur_g['mean']:.4f}", ""])
        rows.append(["Durata min (s)",   f"{dur_g['min']:.4f}",  ""])
        rows.append(["Durata max (s)",   f"{dur_g['max']:.4f}",  ""])
        rows.append(["Durata std (s)",   f"{dur_g['std']:.4f}",  ""])
    else:
        rows.append(["Durata", "N/A", ""])

    # ISI (periodo refrattario)
    isi_res = calc_isi(spindles_df)
    isi_glob = isi_res["globale"] if isi_res else None
    n_idonei = isi_res["canali_idonei"] if isi_res else 0
    rows.append([
        f"ISI periodo refrattario (s)",
        f"{isi_glob:.4f}" if isi_glob is not None else "N/A",
        f"media 10% ISI più bassi su {n_idonei} canali con >= {ISI_MIN_FUSI} fusi",
    ])

    # Spindle Time %
    stp = calc_spindle_time_percentage(spindles_df, durata_nrem_s)
    if stp is not None:
        rows.append([
            "Spindle Time (%)",
            f"{stp['percentuale']:.4f}",
            (f"= {stp['tempo_coperto_s']:.3f} s coperti (union of intervals, "
             f"{stp['n_intervalli']} intervalli distinti) / {durata_nrem_s:.3f} s N2 * 100"),
        ])
    else:
        rows.append(["Spindle Time (%)", "N/A", ""])

    # Ritardo interemisferico
    rit = calc_ritardo_interemisferico(spindles_df)
    rit_glob = rit["globale"]
    rows.append([
        "Ritardo interemisferico medio (s)",
        f"{rit_glob:.4f}" if rit_glob is not None else "N/A",
        rit["nota"],
    ])
    rows.append([])

    # Dettaglio ritardo per coppia omologa
    rows.append(["--- RITARDO INTEREMISFERICO PER COPPIA (s) ---"])
    if not rit["applicabile"]:
        rows.append(["Non applicabile", rit["nota"], "", "", ""])
    else:
        rows.append([
            "Coppia",
            "Ritardo medio (DX - SX, s)",
            "N spindles SX", "N spindles DX",
            "N sovrapposizioni (accoppiati)",
            "Interpretazione",
        ])
        for coppia, info in rit["per_coppia"].items():
            ritardo = info["ritardo"]
            n_sovr  = info["n_sovrapposizioni"]
            n_sx    = info["n_sx"]
            n_dx    = info["n_dx"]
            if ritardo is None:
                motivo = ("nessun fuso in uno dei due canali"
                          if n_sx == 0 or n_dx == 0
                          else "spindles asimmetrici (nessuna sovrapposizione)")
                rows.append([coppia, "NA", n_sx, n_dx, n_sovr, motivo])
            else:
                interp = ("SX anticipa DX" if ritardo > 0
                          else ("DX anticipa SX" if ritardo < 0 else "sincroni"))
                rows.append([coppia, f"{ritardo:.4f}", n_sx, n_dx, n_sovr, interp])

        # Riepilogo NA% basata su spindles
        rows.append([])
        rows.append([
            "Riepilogo NA%",
            f"Spindles totali nei canali delle coppie: {rit['tot_spindles_nelle_coppie']}",
            f"Accoppiati: {rit['tot_spindles_accoppiati']}",
            f"Non accoppiati (NA): {rit['spindles_na']}",
            f"% NA = {rit['perc_na']:.1f}%",
            "fisiologico se <= 30%",
        ])
    rows.append([])

    # ── metriche PER CANALE (opzionale) ─────────────────────────────────
    if not global_only:
        isi_per_canale = isi_res["per_canale"] if isi_res else {}
        canali = sorted(spindles_df["Canale"].unique())
        rows.append(["--- PER CANALE ---"])
        rows.append([
            "Canale", "N spindles",
            "Tasso (sp/min)",
            "Durata media (s)", "Durata min (s)", "Durata max (s)", "Durata std (s)",
            f"ISI refrattario (s) [>=10 fusi, 10% ISI min]",
            "Emisfero",
        ])

        for canale in canali:
            n_sp    = len(spindles_df[spindles_df["Canale"] == canale])
            tasso_c = calc_tasso(spindles_df, durata_nrem_m or 0, canale)
            dur_c   = calc_durata(spindles_df, canale)
            isi_c   = isi_per_canale.get(canale)
            emisfero = ("SX" if canale in EMISFERO_SX
                        else ("DX" if canale in EMISFERO_DX else "Centrale/Mediano"))
            rows.append([
                canale,
                n_sp,
                f"{tasso_c:.4f}" if tasso_c is not None else "N/A",
                f"{dur_c['mean']:.4f}" if dur_c else "N/A",
                f"{dur_c['min']:.4f}"  if dur_c else "N/A",
                f"{dur_c['max']:.4f}"  if dur_c else "N/A",
                f"{dur_c['std']:.4f}"  if dur_c else "N/A",
                f"{isi_c:.4f}" if isi_c is not None else f"N/A (< {ISI_MIN_FUSI} fusi)",
                emisfero,
            ])

    # ── distribuzione spaziale ──────────────────────────────────────────
    rows.append([])
    rows.append(["--- DISTRIBUZIONE SPAZIALE ---"])
    rows.append(["Canale", "N spindles", "% sul totale", "Emisfero"])

    distr = calc_distribuzione_spaziale(spindles_df)
    n_tot_sp = len(spindles_df)
    for canale in CANALI_STANDARD_ORDINE:
        n_ch = int(spindles_df[spindles_df["Canale"] == canale].shape[0])
        perc = distr[canale]
        emisfero = ("SX" if canale in EMISFERO_SX
                    else ("DX" if canale in EMISFERO_DX else "Centrale/Mediano"))
        rows.append([
            canale,
            n_ch,
            f"{perc:.2f}" if n_ch > 0 else "0.00",
            emisfero,
        ])
    rows.append(["TOTALE", n_tot_sp, "100.00", ""])

    return rows


def write_phase_csv(filepath: str, phase: dict, output_dir: str,
                    global_only: bool = False) -> str:
    """
    Scrive il file CSV di output per una singola fase N2.
    Contiene i dati originali + le metriche in fondo.
    """
    spindles_df = phase["spindles"]

    # Righe dati originali
    original_rows = [["Canale", "Start_time", "End_time"]]
    for _, row in spindles_df.iterrows():
        original_rows.append([
            row["Canale"],
            seconds_to_hms(row["Start_s"]),
            seconds_to_hms(row["End_s"]),
        ])

    # Info N2
    info_row = [
        "Inizio NREM", seconds_to_hms(phase["inizio_nrem"]) if phase["inizio_nrem"] else "N/A",
        "Fine NREM",   seconds_to_hms(phase["fine_nrem"])   if phase["fine_nrem"]   else "N/A",
    ]

    # Metriche
    metric_rows = build_output_rows(phase, global_only=global_only)

    # Costruisco DataFrame output
    all_rows = [info_row, [], original_rows[0]] + original_rows[1:] + metric_rows

    # Determino il numero max di colonne
    max_cols = max(len(r) for r in all_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in all_rows]

    out_df = pd.DataFrame(padded)

    # Percorso output
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_name = f"{base}_{phase['phase_num']}.csv"
    out_path = os.path.join(output_dir, out_name)

    out_df.to_csv(out_path, index=False, header=False, encoding="utf-8-sig")
    return out_path


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def process_file(filepath: str, output_dir: str, global_only: bool = False) -> None:
    print(f"\n{'─'*60}")
    print(f"File: {filepath}")

    try:
        phases = parse_input_file(filepath)
    except Exception as e:
        print(f"  [ERRORE] {e}")
        return

    print(f"  Fasi N2 trovate: {len(phases)}")

    for phase in phases:
        n_sp = len(phase["spindles"])
        dur  = (phase["fine_nrem"] - phase["inizio_nrem"]) if (phase["inizio_nrem"] and phase["fine_nrem"]) else None
        if dur:
            print(f"  Fase {phase['phase_num']}: {n_sp} spindles validi, "
                  f"durata NREM = {dur:.1f}s ({dur/60:.2f} min)")
        else:
            print(f"  Fase {phase['phase_num']}: {n_sp} spindles validi, durata NREM = N/A")

        out_path = write_phase_csv(filepath, phase, output_dir, global_only=global_only)
        print(f"    → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calcola metriche spindles EEG da CSV con fasi N2."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Uno o più file CSV (o cartelle contenenti CSV) da processare."
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Cartella di output (default: stessa cartella del file input)."
    )
    parser.add_argument(
        "--global-only", action="store_true",
        help="Stampa nel CSV solo le metriche globali (omette la tabella per canale)."
    )
    args = parser.parse_args()

    # Raccolgo tutti i file CSV da processare
    csv_files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            for fname in os.listdir(inp):
                if fname.lower().endswith(".csv"):
                    csv_files.append(os.path.join(inp, fname))
        elif os.path.isfile(inp):
            csv_files.append(inp)
        else:
            print(f"[AVVISO] '{inp}' non trovato, saltato.")

    if not csv_files:
        print("Nessun file CSV trovato.")
        sys.exit(1)

    print(f"File da processare: {len(csv_files)}")
    if args.global_only:
        print("Modalità: solo metriche globali")

    for filepath in csv_files:
        out_dir = args.output_dir if args.output_dir else os.path.dirname(filepath) or "."
        os.makedirs(out_dir, exist_ok=True)
        process_file(filepath, out_dir, global_only=args.global_only)

    print(f"\n{'─'*60}")
    print("Completato.")


if __name__ == "__main__":
    main()