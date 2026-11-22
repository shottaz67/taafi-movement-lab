#!/usr/bin/env python3
"""
Movement Lab v2 - objective movement screening from a single phone video.

video -> MediaPipe Pose (open source, Google) -> joint angles -> rule-based findings
      -> annotated video + key frames + bilingual (EN/AR) 2-page PDF report.
No AI service and no internet needed once installed.

Usage:
  python movement_lab.py video.mp4 --movement squat --view side --patient "Name"
  python movement_lab.py video2.mp4 --movement squat --view side --compare output/squat_results.json
  movements: squat | shoulder_raise | knee_bend | elbow_flexion | hip_flexion | sit_to_stand | lunge | single_leg_balance | shoulder_flexion | hip_abduction | straight_leg_raise | gait | tug | posture | neck | wrist      view: side | front
"""
import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
KEYS = [L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "taafi_logo.png")   # Taafi identity: green #70B000, olive #3F6200
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

MOVES = {
    "squat": ("Squat", "القرفصاء (سكوات)"),
    "shoulder_raise": ("Shoulder abduction (arm raised to the side)", "تبعيد الكتف (رفع الذراع جانباً)"),
    "shoulder_flexion": ("Shoulder flexion (arm raised forward)", "ثني الكتف (رفع الذراع للأمام)"),
    "hip_abduction": ("Hip abduction (leg out to the side)", "تبعيد الورك (رفع الساق جانباً)"),
    "straight_leg_raise": ("Straight leg raise", "رفع الساق المستقيمة"),
    "gait": ("Gait (walking)", "المشي"),
    "tug": ("Timed Up and Go", "اختبار الوقوف والمشي الموقّت"),
    "posture": ("Posture screen (standing still)", "فحص وضعية الجسم (وقوفاً)"),
    "neck": ("Neck range of motion", "مدى حركة الرقبة"),
    "wrist": ("Wrist bend", "ثني الرسغ"),
    "knee_bend": ("Knee flexion / extension", "ثني ومد الركبة"),
    "elbow_flexion": ("Elbow flexion / extension (arm curl)", "ثني ومد المرفق (الذراع)"),
    "hip_flexion": ("Hip flexion (standing knee raise)", "ثني الورك (رفع الركبة وقوفاً)"),
    "sit_to_stand": ("Sit-to-stand (5 repetitions)", "الجلوس والوقوف (٥ تكرارات)"),
    "lunge": ("Lunge", "الطعنة (لانج)"),
    "single_leg_balance": ("Single-leg balance", "التوازن على ساق واحدة"),
}
STATUS_EN = {"ok": "Within expected", "watch": "Worth monitoring", "flag": "Notable", "info": "Informational"}
STATUS_AR = {"ok": "ضمن المتوقع", "watch": "يستحق المتابعة", "flag": "ملحوظ", "info": "معلومة"}
SIDE_EN = {"L": "left", "R": "right"}
SIDE_AR = {"L": "الأيسر", "R": "الأيمن"}


# ----------------------------------------------------------------- tracking
def track(video_path, max_width=960):
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, max_width / W0)
    W, H = int(W0 * scale), int(H0 * scale)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (W, H))
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        a = np.full((33, 3), np.nan)
        if res.pose_landmarks:
            for i, l in enumerate(res.pose_landmarks.landmark):
                a[i] = (l.x * W, l.y * H, l.visibility)
        out.append(a)
    cap.release()
    return np.array(out), fps, (W, H)


def clean(lm, fps, keys=KEYS):
    P = lm[:, :, :2].copy()
    vis = lm[:, :, 2]
    P[vis < 0.5] = np.nan
    ok_frames = ~np.isnan(P[:, keys, 0]).any(axis=1)
    detect_rate = float(ok_frames.mean()) if len(P) else 0.0
    t = np.arange(len(P))
    k = max(3, int(fps / 6) | 1)
    for j in range(33):
        for c in range(2):
            y = P[:, j, c]
            good = ~np.isnan(y)
            if good.sum() < 2:
                continue
            y = np.interp(t, t[good], y[good])
            y = medfilt(y, 5)
            P[:, j, c] = np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return P, detect_rate


def ang(a, b, c):
    v1, v2 = a - b, c - b
    cosv = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def series(P):
    mid_sh = (P[:, L_SH] + P[:, R_SH]) / 2
    mid_hip = (P[:, L_HIP] + P[:, R_HIP]) / 2
    d = mid_sh - mid_hip
    sw = np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=1) + 1e-9

    def lean(sh, hip):
        dd = P[:, sh] - P[:, hip]
        return np.degrees(np.arctan2(np.abs(dd[:, 0]), -dd[:, 1]))

    def shin(kn, an):
        dd = P[:, kn] - P[:, an]
        return np.degrees(np.arctan2(np.abs(dd[:, 0]), np.abs(dd[:, 1]) + 1e-9))

    def upper_arm(sh, el):
        dd = P[:, el] - P[:, sh]
        return np.degrees(np.arctan2(np.abs(dd[:, 0]), np.abs(dd[:, 1]) + 1e-9))

    return {
        "knee_L": ang(P[:, L_HIP], P[:, L_KN], P[:, L_AN]),
        "knee_R": ang(P[:, R_HIP], P[:, R_KN], P[:, R_AN]),
        "sh_L": ang(P[:, L_HIP], P[:, L_SH], P[:, L_EL]),
        "sh_R": ang(P[:, R_HIP], P[:, R_SH], P[:, R_EL]),
        "trunk": np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1])),
        "trunk_L": lean(L_SH, L_HIP), "trunk_R": lean(R_SH, R_HIP),
        "shin_L": shin(L_KN, L_AN), "shin_R": shin(R_KN, R_AN),
        "hip_L": ang(P[:, L_SH], P[:, L_HIP], P[:, L_KN]),
        "hip_R": ang(P[:, R_SH], P[:, R_HIP], P[:, R_KN]),
        "elb_L": ang(P[:, L_SH], P[:, L_EL], P[:, L_WR]),
        "elb_R": ang(P[:, R_SH], P[:, R_EL], P[:, R_WR]),
        "arm_L": upper_arm(L_SH, L_EL), "arm_R": upper_arm(R_SH, R_EL),
        "sh_diff": np.abs(P[:, L_SH, 1] - P[:, R_SH, 1]) / sw * 100,
    }


def medial_knee_dev(P, idx, side):
    hip, kn, an = (L_HIP, L_KN, L_AN) if side == "L" else (R_HIP, R_KN, R_AN)
    mid_x = (P[idx, L_HIP, 0] + P[idx, R_HIP, 0]) / 2
    h, k, a = P[idx, hip], P[idx, kn], P[idx, an]
    frac = (k[1] - h[1]) / (a[1] - h[1] + 1e-9)
    line_x = h[0] + frac * (a[0] - h[0])
    sign = np.sign(mid_x - h[0]) or 1.0
    return float((k[0] - line_x) * sign / (np.linalg.norm(a - h) + 1e-9) * 100)


def visible_side(lm):
    def m(ids):
        v = float(np.nanmean(lm[:, ids, 2]))
        return -1.0 if np.isnan(v) else v
    return "L" if m([L_HIP, L_KN, L_AN]) >= m([R_HIP, R_KN, R_AN]) else "R"


def visible_arm_side(lm):
    def m(ids):
        v = float(np.nanmean(lm[:, ids, 2]))
        return -1.0 if np.isnan(v) else v
    return "L" if m([L_SH, L_EL, L_WR]) >= m([R_SH, R_EL, R_WR]) else "R"


def needed(movement, side):
    if movement in ("lunge", "single_leg_balance", "gait"):
        return [L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
    if movement == "tug":
        return [L_HIP, R_HIP]
    if movement == "shoulder_flexion":
        if side:
            return list((L_SH, L_EL, L_HIP) if side == "L" else (R_SH, R_EL, R_HIP))
        return [L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP]
    if movement == "posture":
        if side:
            return [7 if side == "L" else 8] + list((L_SH, L_HIP) if side == "L" else (R_SH, R_HIP))
        return [7, 8, L_SH, R_SH, L_HIP, R_HIP]
    if movement == "neck":
        if side:
            return [0, 7 if side == "L" else 8, L_SH if side == "L" else R_SH]
        return [0, 7, 8, L_SH, R_SH]
    if movement == "wrist":
        return list((L_EL, L_WR, 19) if side == "L" else (R_EL, R_WR, 20)) if side else [L_EL, R_EL, L_WR, R_WR, 19, 20]
    if movement == "elbow_flexion":
        if side:
            return list((L_SH, L_EL, L_WR, L_HIP) if side == "L" else (R_SH, R_EL, R_WR, R_HIP))
        return [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]
    if side:
        return list((L_SH, L_HIP, L_KN, L_AN) if side == "L" else (R_SH, R_HIP, R_KN, R_AN))
    if movement == "shoulder_raise":
        return [L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP]
    if movement == "knee_bend":
        return [L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
    return [L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]


# ------------------------------------------------------------------ metrics
def asym(l, r):
    return abs(l - r) / max(abs(l), abs(r), 1e-9) * 100


def asym_status(a):
    return "ok" if a < 10 else ("watch" if a < 15 else "flag")


def rep_windows(f, pk, thr=10):
    """For each peak (max of movement signal f) find where the movement started and ended."""
    base = np.percentile(f, 10)
    out = []
    for i in pk:
        a = int(i)
        while a > 0 and f[a] > base + thr:
            a -= 1
        b = int(i)
        while b < len(f) - 1 and f[b] > base + thr:
            b += 1
        out.append((a, int(i), b))
    return out


def longest_run(mask):
    best, i, n = (0, 0), 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def windows_frac(g, pk, frac=0.25):
    """Windows around peaks of g where g is above base + frac*(peak-base)."""
    base = np.percentile(g, 10)
    out = []
    for i in pk:
        lvl = base + frac * (g[i] - base)
        a = int(i)
        while a > 0 and g[a] > lvl:
            a -= 1
        b = int(i)
        while b < len(g) - 1 and g[b] > lvl:
            b += 1
        out.append((a, int(i), b))
    return out


def compute(P, fps, movement, view, side=None):
    S = series(P)
    rows, notes, checks, chips = [], [], [], []
    reps, key_idx, wins, f, trunk_sig = 0, 0, [], None, S["trunk"]
    plot = None

    def add(key, en, ar, status, **kw):
        d = dict(key=key, en=en, ar=ar, status=status, left=None, right=None, value=None, text=None,
                 asym=None, side=None, unit="", ref_en="", ref_ar="")
        d.update(kw)
        rows.append(d)
        return d

    def pair_row(key, en, ar, l, r, unit, ref_en, ref_ar, good=None, watch=None):
        a = asym(l, r)
        st, low = asym_status(a), False
        if good is not None:
            lo = min(l, r)
            st_r = "ok" if lo >= good else ("watch" if lo >= watch else "flag")
            order = ("ok", "watch", "flag")
            if order.index(st_r) > order.index(st):
                st, low = st_r, True
        return add(key, en, ar, st, left=round(float(l), 1), right=round(float(r), 1),
                   asym=round(a, 1), unit=unit, ref_en=ref_en, ref_ar=ref_ar, low=low)

    def side_row(key, en, ar, v, sd, good, watch, ref_en, ref_ar):
        st = "ok" if v >= good else ("watch" if v >= watch else "flag")
        return add(key, en, ar, st, side=sd, value=round(float(v), 1), unit="deg",
                   left=round(float(v), 1) if sd == "L" else None,
                   right=round(float(v), 1) if sd == "R" else None, ref_en=ref_en, ref_ar=ref_ar)

    def single_row(key, en, ar, v, unit, watch, flag, ref_en, ref_ar):
        st = "ok" if v < watch else ("watch" if v < flag else "flag")
        return add(key, en, ar, st, value=round(float(v), 1), unit=unit, ref_en=ref_en, ref_ar=ref_ar)

    def info_row(key, en, ar, text, ref_en="", ref_ar=""):
        return add(key, en, ar, "info", text=text, ref_en=ref_en, ref_ar=ref_ar)

    def chip(en, ar, value, status):
        chips.append(dict(en=en, ar=ar, value=value, status=status))

    if movement == "squat":
        kn = S[f"knee_{side}"] if side else (S["knee_L"] + S["knee_R"]) / 2
        f = 180 - kn
        trunk_sig = S[f"trunk_{side}"] if side else S["trunk"]
        pk, _ = find_peaks(f, prominence=25, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the deepest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعمق نقطة في الفيديو."))
        wins = rep_windows(f, pk)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        depth = float(np.mean(f[pk]))
        if side:
            r1 = side_row("squat_flex", "Knee flexion at deepest point", "ثني الركبة عند أعمق نقطة", depth, side, 85, 60,
                          f"Joint angle at bottom: {180 - depth:.0f} deg (flexion = 180 - joint angle). Deep squat ~110-130 flexion",
                          "الثني = 180 - زاوية المفصل")
            chip("Depth", "العمق", f"{depth:.0f}°", r1["status"])
            tr = float(np.max(trunk_sig[pk]))
            r2 = single_row("trunk", "Trunk lean at the bottom", "ميلان الجذع عند أدنى نقطة", tr, "deg", 40, 50,
                            "Forward lean up to ~40 deg is common", "ميلان للأمام حتى ~40 درجة شائع")
            chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
            sh = float(np.mean(S[f"shin_{side}"][pk]))
            info_row("shin", "Shin forward tilt at the bottom (ankle mobility indicator)",
                     "ميلان الساق للأمام عند أدنى نقطة (مؤشر حركة الكاحل)", f"{sh:.0f} deg",
                     "Typical deep squat ~25-45 deg; low values can suggest limited ankle motion")
        else:
            notes.append(("Front view cannot measure squat depth (the thighs point toward the camera), so only knee alignment "
                          "and trunk lean are graded. Film from the side for depth.",
                          "التصوير الأمامي لا يقيس عمق القرفصاء (الفخذان يتجهان نحو الكاميرا)، لذلك يُقيَّم فقط محاذاة الركبة وميلان الجذع. صوّر من الجانب لقياس العمق."))
            tr = float(np.max(trunk_sig[pk]))
            r2 = single_row("trunk", "Sideways trunk lean at the bottom", "ميلان الجذع جانبياً", tr, "deg", 6, 10,
                            "Should stay near vertical", "يجب أن يبقى قريباً من العمودي")
            vals = []
            for sd in ("L", "R"):
                v = float(np.mean([medial_knee_dev(P, i, sd) for i in pk]))
                vals.append(v)
                single_row(f"valgus_{sd}", f"Knee falling inward ({SIDE_EN[sd]})", f"ميل الركبة للداخل ({SIDE_AR[sd]})",
                           v, "% leg", 5, 10, "Knee should track over the foot", "الركبة تتبع اتجاه القدم")
            worst = max(rows[-2:], key=lambda r: r["value"])
            chip("Knee alignment", "محاذاة الركبة", f"{worst['value']:.0f}%", worst["status"])
            chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        if len(pk) >= 2 and side:
            rng = float(np.max(f[pk]) - np.min(f[pk]))
            r3 = single_row("rep_range", "Depth difference between repetitions", "فرق العمق بين التكرارات", rng, "deg", 5, 10,
                            "Similar depth each rep suggests steady control", "عمق متقارب يدل على ثبات التحكم")
            chip("Consistency", "الثبات", f"±{rng:.0f}°", r3["status"])
        elif side:
            chip("Consistency", "الثبات", "n/a", "info")
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Descent / ascent time (average)", "زمن النزول / الصعود (المتوسط)", f"{dn:.1f} s / {up:.1f} s",
                 "Timing of the controlled lowering and the rise")

    elif movement == "shoulder_raise":
        f = (S["sh_L"] + S["sh_R"]) / 2
        pk, _ = find_peaks(f, prominence=40, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the highest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعلى نقطة في الفيديو."))
        wins = rep_windows(f, pk)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        r1 = pair_row("sh_elev", "Maximum arm elevation", "أقصى ارتفاع للذراع", float(np.mean(S["sh_L"][pk])),
                      float(np.mean(S["sh_R"][pk])), "deg", "Full elevation ~150-180 deg", "الرفع الكامل ~150-180 درجة", 140, 110)
        chip("Arm raise L/R", "رفع الذراع", f"{r1['asym']}%", r1["status"])
        tr = float(np.max(S["trunk"][pk]))
        r2 = single_row("trunk", "Sideways trunk lean at the top", "ميلان الجذع جانبياً عند القمة", tr, "deg", 6, 10,
                        "Should stay near vertical", "يجب أن يبقى قريباً من العمودي")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        hk = float(np.max(S["sh_diff"][pk]))
        r3 = single_row("hike", "Shoulder height difference at the top", "فرق ارتفاع الكتفين عند القمة", hk, "% width",
                        12, 20, "Shoulders should stay level", "يجب أن يبقى الكتفان بمستوى واحد")
        chip("Shoulders level", "مستوى الكتفين", f"{hk:.0f}%", r3["status"])
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Raise / lower time (average)", "زمن الرفع / الإنزال (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        trunk_sig = S["trunk"]
        if view != "front":
            notes.append(("Shoulder raise is best filmed from the front; results from a side view are less reliable.",
                          "يفضل تصوير رفع الذراع من الأمام؛ نتائج التصوير الجانبي أقل دقة."))

    elif movement == "knee_bend":
        if side:
            f = 180 - S[f"knee_{side}"]
            m = float(f.max())
            r1 = side_row("knee_flex", "Maximum knee flexion", "أقصى ثني للركبة", m, side, 120, 90,
                          "Normal ~130-140 deg", "الطبيعي ~130-140 درجة")
            chip("Knee flexion", "ثني الركبة", f"{m:.0f}°", r1["status"])
        else:
            fl_L, fl_R = 180 - S["knee_L"], 180 - S["knee_R"]
            f = np.maximum(fl_L, fl_R)
            mL, mR = float(fl_L.max()), float(fl_R.max())
            if min(mL, mR) < 25:
                weak = "L" if mL < mR else "R"
                notes.append((f"The {SIDE_EN[weak]} knee barely bent, so a left/right comparison is not meaningful. "
                              "Film both legs bending.",
                              f"الركبة {SIDE_AR[weak]} لم تنثنِ تقريباً، لذلك لا يصح المقارنة بين الجانبين. صوّر ثني الساقين."))
            r1 = pair_row("knee_flex", "Maximum knee flexion", "أقصى ثني للركبة", mL, mR, "deg",
                          "Normal ~130-140 deg", "الطبيعي ~130-140 درجة")
            chip("Left/right flexion", "تماثل الثني", f"{r1['asym']}%", r1["status"])
        key_idx = int(np.argmax(f))
        pk, _ = find_peaks(f, prominence=25, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([key_idx])
        wins = rep_windows(f, pk)
        reps = len(pk)
        trunk_sig = S["trunk"]
        sides_ = [side] if side else ["L", "R"]
        ext = {sd: float(np.percentile(180 - S["knee_" + sd], 5)) for sd in sides_}
        worst = max(ext, key=ext.get)
        r2 = single_row("knee_ext", "Knee extension (how straight the knee gets)", "مد الركبة (مدى استقامتها)", ext[worst], "deg",
                        10, 20, "Full extension ~0 deg; a remaining bend above ~10 deg is an extension deficit",
                        "المد الكامل ~0 درجة؛ بقاء ثني أكثر من ~10 درجات يعني نقصاً في المد")
        chip("Extension", "المد", f"{ext[worst]:.0f}° bend left", r2["status"])
        rom = {sd: float(np.max(180 - S["knee_" + sd]) - ext[sd]) for sd in sides_}
        info_row("knee_rom", "Total range of motion (flexion minus extension)", "مدى الحركة الكلي (الثني ناقص المد)",
                 "  ".join(f"{SIDE_EN[sd]} {rom[sd]:.0f} deg" for sd in sides_))
    elif movement == "elbow_flexion":
        sides_ = [side] if side else ["L", "R"]
        fl = {sd: 180 - S["elb_" + sd] for sd in sides_}
        f = fl[side] if side else np.maximum(fl["L"], fl["R"])
        key_idx = int(np.argmax(f))
        pk, _ = find_peaks(f, prominence=30, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([key_idx])
            notes.append(("No clear repetitions were detected; the point of maximum bend was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد نقطة أقصى ثني."))
        wins = rep_windows(f, pk)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        if side:
            m = float(np.mean(fl[side][pk]))
            r1 = side_row("elbow_flex", "Maximum elbow flexion", "أقصى ثني للمرفق", m, side, 125, 100,
                          "Normal ~140-150 deg", "الطبيعي ~140-150 درجة")
        else:
            mL, mR = float(np.mean(fl["L"][pk])), float(np.mean(fl["R"][pk]))
            if min(mL, mR) < 30:
                weak = "L" if mL < mR else "R"
                notes.append((f"The {SIDE_EN[weak]} elbow barely bent, so a left/right comparison is not meaningful. "
                              "Film both arms bending.",
                              f"المرفق {SIDE_AR[weak]} لم ينثنِ تقريباً، لذلك لا يصح المقارنة بين الجانبين. صوّر ثني الذراعين."))
            r1 = pair_row("elbow_flex", "Maximum elbow flexion", "أقصى ثني للمرفق", mL, mR, "deg",
                          "Normal ~140-150 deg", "الطبيعي ~140-150 درجة")
        chip("Elbow flexion" if side else "Left/right flexion", "ثني المرفق" if side else "تماثل الثني",
             f"{m:.0f}°" if side else f"{r1['asym']}%", r1["status"])
        ext = {sd: float(np.percentile(fl[sd], 5)) for sd in sides_}
        worst = max(ext, key=ext.get)
        r2 = single_row("elbow_ext", "Elbow extension (how straight the arm gets)", "مد المرفق (مدى استقامة الذراع)", ext[worst],
                        "deg", 10, 20, "Full extension ~0 deg; a remaining bend above ~10 deg is an extension deficit",
                        "المد الكامل ~0 درجة؛ بقاء ثني أكثر من ~10 درجات يعني نقصاً في المد")
        chip("Extension", "المد", f"{ext[worst]:.0f}° bend left", r2["status"])
        info_row("elbow_rom", "Total range of motion (flexion minus extension)", "مدى الحركة الكلي (الثني ناقص المد)",
                 "  ".join(f"{SIDE_EN[sd]} {float(np.max(fl[sd]) - ext[sd]):.0f} deg" for sd in sides_))
        sw = max(float(np.percentile(S["arm_" + sd], 95) - np.percentile(S["arm_" + sd], 5)) for sd in sides_)
        r4 = single_row("arm_swing", "Upper-arm movement during the curl (should stay by the side)",
                        "حركة العضد أثناء الثني (يجب أن يبقى بجانب الجسم)", sw, "deg", 25, 40,
                        "Large movement suggests the shoulder is helping", "الحركة الكبيرة تعني أن الكتف يساعد")
        trunk_sig = S[f"trunk_{side}"] if side else S["trunk"]
        tr = float(np.max(trunk_sig[pk]))
        r3 = single_row("trunk", "Trunk lean / swing at the top of the curl", "ميلان الجذع عند قمة الثني", tr, "deg", 15, 25,
                        "Body should stay upright", "يجب أن يبقى الجسم منتصباً")
        chip("Compensation", "التعويض", f"{max(tr, sw / 2):.0f}°", "ok" if r3["status"] == r4["status"] == "ok" else
             ("flag" if "flag" in (r3["status"], r4["status"]) else "watch"))
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Bend / straighten time (average)", "زمن الثني / المد (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        if view == "front":
            notes.append(("Elbow flexion is measured most reliably from the side; from the front the forearm points toward "
                          "the camera and angles can be under-estimated.",
                          "يُقاس ثني المرفق بدقة أكبر من الجانب؛ من الأمام يتجه الساعد نحو الكاميرا وقد تقل الزوايا المقيسة."))
    elif movement == "hip_flexion":
        sides_ = [side] if side else ["L", "R"]
        hf = {sd: 180 - S["hip_" + sd] for sd in sides_}
        f = hf[side] if side else np.maximum(hf["L"], hf["R"])
        pk, _ = find_peaks(f, prominence=20, distance=max(1, int(fps * 0.6)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the highest knee point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعلى نقطة للركبة في الفيديو."))
        wins = rep_windows(f, pk, thr=8)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        ref_en, ref_ar = ("Active hip flexion with the knee raised ~90-120 deg",
                          "ثني الورك الفعّال مع رفع الركبة ~90-120 درجة")
        if side:
            m = float(np.mean(hf[side][pk]))
            r1 = side_row("hip_flex", "Maximum hip flexion", "أقصى ثني للورك", m, side, 80, 60, ref_en, ref_ar)
            chip("Hip flexion", "ثني الورك", f"{m:.0f}°", r1["status"])
        else:
            mL, mR = float(np.percentile(hf["L"], 98)), float(np.percentile(hf["R"], 98))
            if min(mL, mR) < 25:
                weak = "L" if mL < mR else "R"
                notes.append((f"The {SIDE_EN[weak]} knee was barely lifted, so a left/right comparison is not meaningful. "
                              "Lift both knees in turn.",
                              f"الركبة {SIDE_AR[weak]} لم تُرفع تقريباً، لذلك لا يصح المقارنة بين الجانبين. ارفع الركبتين بالتناوب."))
            r1 = pair_row("hip_flex", "Maximum hip flexion", "أقصى ثني للورك", mL, mR, "deg", ref_en, ref_ar)
            chip("Left/right hip", "تماثل الورك", f"{r1['asym']}%", r1["status"])
            notes.append(("From the front the thigh points toward the camera, so hip angles are under-estimated; "
                          "side view is more reliable.",
                          "من الأمام يتجه الفخذ نحو الكاميرا فتقل الزوايا المقيسة؛ التصوير الجانبي أدق."))
        trunk_sig = S[f"trunk_{side}"] if side else S["trunk"]
        tr = float(np.max(trunk_sig[pk]))
        r2 = single_row("trunk", "Trunk lean while lifting the knee", "ميلان الجذع أثناء رفع الركبة", tr, "deg", 10, 20,
                        "Body should stay upright", "يجب أن يبقى الجسم منتصباً")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Lift / lower time (average)", "زمن الرفع / الإنزال (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        plot = dict(title="Hip flexion over time", ylabel="flexion (deg)",
                    lines=[((("Left" if sd == "L" else "Right") if side else ("Left" if sd == "L" else "Right")), hf[sd],
                            "#3F6200" if side else ("#2f80ed" if sd == "L" else "#f2994a")) for sd in sides_])

    elif movement == "sit_to_stand":
        g = S[f"knee_{side}"] if side else (S["knee_L"] + S["knee_R"]) / 2      # 180 = standing straight
        pk, _ = find_peaks(g, prominence=25, distance=max(1, int(fps * 0.6)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(g))])
            notes.append(("No clear stand-ups were detected. Film the whole body, seated to standing, 5 times.",
                          "لم يتم رصد حالات وقوف واضحة. صوّر الجسم كاملاً من الجلوس إلى الوقوف ٥ مرات."))
        wins = windows_frac(g, pk, 0.25)
        reps = len(pk)
        f = g
        key_idx = int(pk[len(pk) // 2])
        span = (pk[-1] - wins[0][0]) / fps
        rise0 = max((pk[0] - wins[0][0]) / fps, 0.1)
        cyc = (pk[-1] - pk[0]) / (reps - 1) / fps if reps >= 2 else 2 * rise0
        t5 = rise0 + 4 * cyc            # first seat-off to the 5th stand
        if reps < 2:
            notes.append(("Only one stand was counted, so the 5-stand time is a rough estimate. Film 5 full repetitions.",
                          "تم رصد وقفة واحدة فقط، لذا زمن ٥ تكرارات تقدير تقريبي. صوّر ٥ تكرارات كاملة."))
        r1 = single_row("sts_time", "Time for the stands (scaled to 5 repetitions)", "زمن الوقوف (محسوب لـ ٥ تكرارات)", t5, "s", 12, 15,
                        "Adults usually take under ~12 s for 5 stands; over ~15 s is linked to higher fall risk in older adults",
                        "عادةً أقل من ~12 ثانية لـ ٥ تكرارات؛ أكثر من ~15 ثانية يرتبط بخطر سقوط أعلى لدى كبار السن")
        chip("5-stand time", "زمن ٥ تكرارات", f"{t5:.1f} s", r1["status"])
        info_row("sts_reps", "Stands counted", "عدد مرات الوقوف", f"{reps} in {span:.1f} s")
        ext = float(np.mean(180 - g[pk]))
        r2 = single_row("sts_ext", "Knee bend left when fully standing", "ثني الركبة المتبقي عند الوقوف الكامل", ext, "deg", 15, 25,
                        "Should stand fully upright (about 0-10 deg of bend)", "يجب الوقوف باعتدال كامل (~0-10 درجات)")
        chip("Standing up", "الوقوف", f"{ext:.0f}° bend left", r2["status"])
        trunk_sig = S[f"trunk_{side}"] if side else S["trunk"]
        tr = float(np.max([np.max(trunk_sig[a:i + 1]) for a, i, b in wins]))
        r3 = single_row("trunk", "Forward trunk lean while rising", "ميلان الجذع للأمام أثناء النهوض", tr, "deg", 55, 70,
                        "A moderate forward lean (about 30-50 deg) is normal", "الميلان المعتدل للأمام (~30-50 درجة) طبيعي")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r3["status"])
        if len(pk) >= 2:
            rises = [(i - a) / fps for a, i, b in wins]
            info_row("tempo", "Rise time per stand (fastest / slowest)", "زمن النهوض (الأسرع / الأبطأ)",
                     f"{min(rises):.1f} s / {max(rises):.1f} s", "A slowing pattern can reflect fatigue")
        plot = dict(title="Knee angle over time (180 = standing straight)", ylabel="knee angle (deg)",
                    lines=[("Knee angle", g, "#3F6200")])
        notes.append(("Standard test: arms crossed on the chest, same chair each time, stand fully then sit, 5 times.",
                      "الاختبار القياسي: الذراعان متقاطعتان على الصدر، نفس الكرسي كل مرة، وقوف كامل ثم جلوس، ٥ مرات."))

    elif movement == "lunge":
        fl = {sd: 180 - S["knee_" + sd] for sd in ("L", "R")}
        f = np.maximum(fl["L"], fl["R"])
        pk, _ = find_peaks(f, prominence=25, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the deepest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعمق نقطة في الفيديو."))
        wins = rep_windows(f, pk)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        mL, mR = float(np.mean(fl["L"][pk])), float(np.mean(fl["R"][pk]))
        fs = "L" if mL >= mR else "R"
        if view == "front":
            fs = "L" if float(np.mean(P[pk, L_KN, 1])) >= float(np.mean(P[pk, R_KN, 1])) else "R"
        front, back = max(mL, mR), min(mL, mR)
        if view == "front":
            notes.append(("Front view cannot measure lunge depth (the front thigh points toward the camera), so only knee "
                          "alignment and trunk lean are graded. Film from the side for depth.",
                          "التصوير الأمامي لا يقيس عمق الطعنة (الفخذ الأمامي يتجه نحو الكاميرا)، لذلك يُقيَّم فقط محاذاة الركبة وميلان الجذع. صوّر من الجانب لقياس العمق."))
        else:
            r1 = side_row("lunge_front", "Front knee flexion at the bottom", "ثني الركبة الأمامية عند أدنى نقطة", front, fs, 85, 65,
                          "Front knee ~90-110 deg of flexion at the bottom of a lunge", "الركبة الأمامية ~90-110 درجة عند أدنى نقطة")
            chip("Front knee", "الركبة الأمامية", f"{front:.0f}°", r1["status"])
            info_row("lunge_back", "Back knee flexion at the bottom", "ثني الركبة الخلفية عند أدنى نقطة", f"{back:.0f} deg")
        tr = float(np.max(S["trunk"][pk]))
        if view == "front":
            r2 = single_row("trunk", "Sideways trunk lean at the bottom", "ميلان الجذع جانبياً", tr, "deg", 6, 10,
                            "Should stay near vertical", "يجب أن يبقى قريباً من العمودي")
            v = float(np.mean([medial_knee_dev(P, i, fs) for i in pk]))
            r3 = single_row("valgus", "Front knee falling inward", "ميل الركبة الأمامية للداخل", v, "% leg", 5, 10,
                            "Knee should track over the foot", "الركبة تتبع اتجاه القدم")
            chip("Knee alignment", "محاذاة الركبة", f"{v:.0f}%", r3["status"])
        else:
            r2 = single_row("trunk", "Trunk lean at the bottom", "ميلان الجذع عند أدنى نقطة", tr, "deg", 25, 40,
                            "Trunk should stay fairly upright", "يجب أن يبقى الجذع منتصباً نسبياً")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        if len(pk) >= 2 and view != "front":
            rng = float(np.max(f[pk]) - np.min(f[pk]))
            r4 = single_row("rep_range", "Depth difference between repetitions", "فرق العمق بين التكرارات", rng, "deg", 5, 10,
                            "Similar depth each rep suggests steady control", "عمق متقارب يدل على ثبات التحكم")
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Descent / ascent time (average)", "زمن النزول / الصعود (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        info_row("lunge_leg", "Front leg", "الساق الأمامية", "left" if fs == "L" else "right")
        plot = dict(title="Knee flexion over time", ylabel="flexion (deg)",
                    lines=[("Left knee", fl["L"], "#2f80ed"), ("Right knee", fl["R"], "#f2994a")],
                    trunk_title="Trunk lean from vertical (sideways)" if view == "front" else "Trunk lean from vertical (forward)")

    elif movement == "single_leg_balance":
        aL, aR = P[:, L_AN, 1], P[:, R_AN, 1]
        stance = "L" if np.nanmedian(aL) > np.nanmedian(aR) else "R"
        lift_sd = "R" if stance == "L" else "L"
        h_st, a_st = (L_HIP, L_AN) if stance == "L" else (R_HIP, R_AN)
        h_lf, a_lf = (L_HIP, L_AN) if lift_sd == "L" else (R_HIP, R_AN)
        leg_len = float(np.nanmedian(np.abs(P[:, a_st, 1] - P[:, h_st, 1]))) + 1e-9
        hip_w = float(np.nanmedian(np.abs(P[:, L_HIP, 0] - P[:, R_HIP, 0]))) + 1e-9
        lift = (P[:, a_st, 1] - P[:, a_lf, 1]) / leg_len * 100
        f = lift
        a, b = longest_run(lift > 12)
        n_ = len(lift)
        if b - a < int(fps * 0.5):
            a, b = 0, n_
            notes.append(("No clearly lifted foot was detected. Film the front view, one foot lifted off the floor, "
                          "holding as long as possible.",
                          "لم يتم رصد قدم مرفوعة بوضوح. صوّر من الأمام مع رفع قدم واحدة عن الأرض والثبات أطول وقت."))
            hold = 0.0
        else:
            hold = (b - a) / fps
        seg = slice(a, b)
        st_h = "ok" if hold >= 10 else ("watch" if hold >= 5 else "flag")
        add("hold", "Hold time (foot lifted)", "مدة الثبات (القدم مرفوعة)", st_h, value=round(hold, 1), unit="s",
            ref_en="Eyes open: about 10 s or more is expected for most adults under 60",
            ref_ar="عينان مفتوحتان: يُتوقع حوالي ١٠ ثوانٍ أو أكثر لمعظم البالغين دون الستين")
        chip("Hold time", "مدة الثبات", f"{hold:.0f} s", st_h)
        drop = float(np.nanmean((P[seg, h_lf, 1] - P[seg, h_st, 1]) / hip_w * 100))
        r2 = single_row("pelvic_drop", "Pelvis drop on the lifted side", "هبوط الحوض في جهة الساق المرفوعة", max(drop, 0.0), "% width",
                        10, 20, "Pelvis should stay level (a drop suggests hip abductor weakness)",
                        "يجب أن يبقى الحوض مستوياً (الهبوط يوحي بضعف العضلات المبعدة للورك)")
        chip("Pelvic drop", "هبوط الحوض", f"{max(drop, 0):.0f}%", r2["status"])
        cx = (P[seg, L_HIP, 0] + P[seg, R_HIP, 0]) / 2
        sway = float((np.nanpercentile(cx, 95) - np.nanpercentile(cx, 5)) / hip_w * 100)
        r3 = single_row("sway", "Side-to-side sway of the hips", "التمايل الجانبي للحوض", sway, "% width", 20, 40,
                        "Less sway means steadier balance", "التمايل الأقل يعني توازناً أثبت")
        chip("Sway", "التمايل", f"{sway:.0f}%", r3["status"])
        trk = float(np.nanmax(S["trunk"][seg]))
        single_row("trunk", "Sideways trunk lean during the hold", "ميلان الجذع جانبياً أثناء الثبات", trk, "deg", 6, 10,
                   "Trunk should stay near vertical", "يجب أن يبقى الجذع قريباً من العمودي")
        info_row("stance", "Standing leg", "الساق الثابتة", "left" if stance == "L" else "right")
        reps = 1
        key_idx = int((a + b) // 2)
        wins = [(int(a), key_idx, int(b - 1))]
        plot = dict(title="Height of the lifted foot over time (% of leg length)", ylabel="lift (%)",
                    lines=[("Lifted foot", lift, "#3F6200")], trunk_title="Trunk lean from vertical (sideways)")
        if view == "side":
            notes.append(("Single-leg balance is best filmed from the front so pelvis drop and sway can be seen.",
                          "يفضل تصوير التوازن على ساق واحدة من الأمام لرؤية هبوط الحوض والتمايل."))
    elif movement == "shoulder_flexion":
        sides_ = [side] if side else ["L", "R"]
        f = S["sh_" + side] if side else (S["sh_L"] + S["sh_R"]) / 2
        pk, _ = find_peaks(f, prominence=40, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the highest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعلى نقطة في الفيديو."))
        wins = rep_windows(f, pk)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        ref_en, ref_ar = "Full forward flexion ~150-180 deg", "الثني الأمامي الكامل ~150-180 درجة"
        if side:
            m = float(np.mean(S["sh_" + side][pk]))
            r1 = side_row("sh_flex", "Maximum arm elevation (forward)", "أقصى ارتفاع للذراع (للأمام)", m, side, 140, 110, ref_en, ref_ar)
            chip("Arm raise", "رفع الذراع", f"{m:.0f}°", r1["status"])
        else:
            r1 = pair_row("sh_flex", "Maximum arm elevation (forward)", "أقصى ارتفاع للذراع (للأمام)",
                          float(np.mean(S["sh_L"][pk])), float(np.mean(S["sh_R"][pk])), "deg", ref_en, ref_ar)
            chip("Arm raise L/R", "رفع الذراع", f"{r1['asym']}%", r1["status"])
            notes.append(("Forward arm raises are best filmed from the side.", "يفضل تصوير رفع الذراع للأمام من الجانب."))
        trunk_sig = S[f"trunk_{side}"] if side else S["trunk"]
        tr = float(np.max(trunk_sig[pk]))
        r2 = single_row("trunk", "Trunk lean (leaning back to lift)", "ميلان الجذع (الميل للخلف للرفع)", tr, "deg", 10, 20,
                        "Body should stay upright", "يجب أن يبقى الجسم منتصباً")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Raise / lower time (average)", "زمن الرفع / الإنزال (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        plot = dict(title="Arm elevation over time", ylabel="elevation (deg)", hud="Arm",
                    lines=[((("Left" if sd == "L" else "Right")), S["sh_" + sd], "#3F6200" if side else ("#2f80ed" if sd == "L" else "#f2994a"))
                           for sd in sides_])

    elif movement == "hip_abduction":
        def abd(h, a):
            dd = P[:, a] - P[:, h]
            return np.degrees(np.arctan2(np.abs(dd[:, 0]), np.abs(dd[:, 1]) + 1e-9))
        aL, aR = abd(L_HIP, L_AN), abd(R_HIP, R_AN)
        f = np.maximum(aL, aR)
        pk, _ = find_peaks(f, prominence=12, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the widest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أوسع نقطة في الفيديو."))
        wins = rep_windows(f, pk, thr=6)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        mL, mR = float(np.percentile(aL, 98)), float(np.percentile(aR, 98))
        ref_en, ref_ar = "Normal ~30-45 deg", "الطبيعي ~30-45 درجة"
        if min(mL, mR) < 12:
            sd = "L" if mL >= mR else "R"
            r1 = side_row("hip_abd", "Maximum hip abduction", "أقصى تبعيد للورك", max(mL, mR), sd, 30, 20, ref_en, ref_ar)
            chip("Hip abduction", "تبعيد الورك", f"{max(mL, mR):.0f}°", r1["status"])
            notes.append(("Only one leg was lifted, so no left/right comparison was made. Repeat with the other leg.",
                          "تم رفع ساق واحدة فقط، لذلك لم تُجرَ مقارنة بين الجانبين. كرر الاختبار بالساق الأخرى."))
        else:
            r1 = pair_row("hip_abd", "Maximum hip abduction", "أقصى تبعيد للورك", mL, mR, "deg", ref_en, ref_ar, 30, 20)
            chip("Left/right abduction", "تماثل التبعيد", f"{r1['asym']}%", r1["status"])
        tr = float(np.max(S["trunk"][pk]))
        r2 = single_row("trunk", "Sideways trunk lean while lifting the leg", "ميلان الجذع جانبياً أثناء رفع الساق", tr, "deg", 8, 12,
                        "Leaning the body sideways can fake extra range", "الميل الجانبي للجسم قد يعطي مدى زائفاً")
        chip("Trunk lean", "ميلان الجذع", f"{tr:.0f}°", r2["status"])
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Raise / lower time (average)", "زمن الرفع / الإنزال (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        plot = dict(title="Leg angle from vertical over time", ylabel="abduction (deg)", hud="Abduction",
                    lines=[("Left leg", aL, "#2f80ed"), ("Right leg", aR, "#f2994a")])
        if view == "side":
            notes.append(("Hip abduction must be filmed from the front.", "يجب تصوير تبعيد الورك من الأمام."))

    elif movement == "straight_leg_raise":
        hfL, hfR = 180 - ang(P[:, L_SH], P[:, L_HIP], P[:, L_AN]), 180 - ang(P[:, R_SH], P[:, R_HIP], P[:, R_AN])
        sd = side or ("L" if np.nanmax(hfL) >= np.nanmax(hfR) else "R")
        f = hfL if sd == "L" else hfR
        pk, _ = find_peaks(f, prominence=20, distance=max(1, int(fps * 0.8)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the highest point of the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أعلى نقطة في الفيديو."))
        wins = rep_windows(f, pk, thr=8)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        m = float(np.mean(f[pk]))
        lying = float(np.nanmedian(S["trunk_" + sd])) > 45
        if lying:
            r1 = side_row("slr", "Maximum leg raise (hip flexion)", "أقصى رفع للساق (ثني الورك)", m, sd, 70, 50,
                          "Lying straight-leg-raise range ~70-90 deg", "المدى الطبيعي لرفع الساق المستقيمة استلقاءً ~70-90 درجة")
        else:
            r1 = info_row("slr", "Maximum leg raise (hip flexion), standing", "أقصى رفع للساق (ثني الورك) وقوفاً", f"{m:.0f} deg",
                          "Not graded: the ~70-90 deg reference applies to the lying test; a standing active raise is normally lower",
                          "غير مُقيَّم: المرجع ~70-90 درجة خاص بالاختبار استلقاءً؛ الرفع الفعّال وقوفاً أقل عادةً")
            notes.append(("The leg raise was done standing, so it is reported but not graded. For the standard test, film lying on the back from the side.",
                          "تم رفع الساق وقوفاً، لذلك تُعرض القيمة دون تقييم. للاختبار القياسي صوّر الشخص مستلقياً على ظهره من الجانب."))
        chip("Leg raise", "رفع الساق", f"{m:.0f}°", r1["status"])
        kl = float(np.mean((180 - S["knee_" + sd])[pk]))
        r2 = single_row("knee_lag", "Knee bend while the leg is raised (should stay straight)", "ثني الركبة أثناء رفع الساق (يجب أن تبقى مستقيمة)",
                        kl, "deg", 10, 20, "Knee should stay straight (about 0-10 deg of bend)", "يجب أن تبقى الركبة مستقيمة (~0-10 درجات)")
        chip("Knee straight", "استقامة الركبة", f"{kl:.0f}° bend", r2["status"])
        dn = float(np.mean([(i - a) / fps for a, i, b in wins]))
        up = float(np.mean([(b - i) / fps for a, i, b in wins]))
        info_row("tempo", "Raise / lower time (average)", "زمن الرفع / الإنزال (المتوسط)", f"{dn:.1f} s / {up:.1f} s")
        plot = dict(title="Leg raise over time", ylabel="hip flexion (deg)", hud="Leg raise",
                    lines=[(("Left" if sd == "L" else "Right") + " leg", f, "#3F6200")])
        if view == "front":
            notes.append(("Straight leg raise must be filmed from the side.", "يجب تصوير رفع الساق المستقيمة من الجانب."))

    elif movement == "gait":
        sig = P[:, L_AN, 0] - P[:, R_AN, 0]
        rngv = float(np.nanmax(sig) - np.nanmin(sig)) + 1e-9
        dist_ = max(1, int(fps * 0.25))
        pp, _ = find_peaks(sig, prominence=0.15 * rngv, distance=dist_)
        pn, _ = find_peaks(-sig, prominence=0.15 * rngv, distance=dist_)
        ev = sorted([(int(i), "L") for i in pp] + [(int(i), "R") for i in pn])
        f = sig
        if len(ev) < 4:
            notes.append(("Fewer than 4 steps were detected. Film from the side while the person walks across the frame, 5+ steps.",
                          "تم رصد أقل من ٤ خطوات. صوّر من الجانب والشخص يعبر الإطار، ٥ خطوات أو أكثر."))
            ev = ev or [(int(np.argmax(np.abs(sig))), "L")]
        steps = len(ev)
        reps = steps
        key_idx = ev[len(ev) // 2][0]
        wins = [(max(0, e - int(fps * 0.3)), e, min(len(f) - 1, e + int(fps * 0.3))) for e, _ in ev[:12]]
        if steps >= 4:
            tt = (ev[-1][0] - ev[0][0]) / fps
            cad = (steps - 1) / max(tt, 0.1) * 60
            st_c = "ok" if 90 <= cad <= 135 else ("watch" if 75 <= cad <= 150 else "flag")
            add("cadence", "Cadence", "إيقاع المشي (عدد الخطوات)", st_c, value=round(cad, 0), unit="steps/min",
                ref_en="Comfortable adult walking ~100-120 steps/min", ref_ar="المشي المريح للبالغين ~100-120 خطوة بالدقيقة")
            chip("Cadence", "الإيقاع", f"{cad:.0f}/min", st_c)
            gaps = {"L": [], "R": []}
            for (e0, s0), (e1, _) in zip(ev[:-1], ev[1:]):
                gaps[s0].append((e1 - e0) / fps)
            if gaps["L"] and gaps["R"]:
                a_ = asym(np.mean(gaps["L"]), np.mean(gaps["R"]))
                st_a = asym_status(a_)
                add("step_sym", "Left/right step-time difference", "فرق زمن الخطوة بين الجانبين", st_a, value=round(a_, 1), unit="%",
                    ref_en="Under ~10% suggests an even rhythm", ref_ar="أقل من ~10% يدل على إيقاع متساوٍ")
                chip("Step symmetry", "تماثل الخطوات", f"{a_:.0f}%", st_a)
        info_row("steps", "Steps counted", "عدد الخطوات", f"{steps}")
        kL, kR = np.nanpercentile(180 - S["knee_L"], 98), np.nanpercentile(180 - S["knee_R"], 98)
        if not (np.isnan(kL) or np.isnan(kR)):
            if view == "side":
                info_row("gait_knee", "Peak knee bend during swing", "أقصى ثني للركبة أثناء المرجحة", f"left {kL:.0f}  right {kR:.0f} deg",
                         "Typical ~60-70 deg; the far leg is less reliable in side view")
            else:
                info_row("gait_knee", "Peak knee bend during swing", "أقصى ثني للركبة أثناء المرجحة", f"left {kL:.0f}  right {kR:.0f} deg")
        tr = float(np.nanmean(S["trunk"]))
        info_row("trunk", "Average trunk lean", "متوسط ميلان الجذع", f"{tr:.0f} deg")
        trunk_sig = S["trunk"]
        plot = dict(title="Left-right ankle separation over time (peaks = steps)", ylabel="separation (px)", hud="Ankle gap",
                    lines=[("Ankle separation", sig, "#3F6200")])
        if view == "front":
            notes.append(("Gait is analysed best from the side.", "يُحلَّل المشي بشكل أفضل من الجانب."))

    elif movement == "tug":
        hy = np.nanmean(np.stack([P[:, L_HIP, 1], P[:, R_HIP, 1]]), axis=0)
        hx = np.nanmean(np.stack([P[:, L_HIP, 0], P[:, R_HIP, 0]]), axis=0)
        sh_y = np.nanmean(np.stack([P[:, L_SH, 1], P[:, R_SH, 1]]), axis=0)
        torso = float(np.nanmedian(np.abs(sh_y - hy))) + 1e-9
        y0 = float(np.nanmedian(hy[:max(5, int(fps))]))
        gain = y0 - hy
        f = gain
        up_idx = np.where(gain > 0.15 * torso)[0]
        if len(up_idx) < int(fps):
            notes.append(("No stand-up was detected. Start seated, stand, walk about 3 m, turn, return and sit down.",
                          "لم يتم رصد وقوف. ابدأ جالساً ثم قف وامشِ نحو ٣ أمتار ثم استدر وعد واجلس."))
            up_idx = np.array([0, len(hy) - 1])
        a, b = int(up_idx[0]), int(up_idx[-1])
        total = (b - a) / fps
        st_t = "ok" if total < 10 else ("watch" if total < 13.5 else "flag")
        add("tug_time", "Total test time (rise to sit)", "زمن الاختبار الكلي (من الوقوف حتى الجلوس)", st_t, value=round(total, 1), unit="s",
            ref_en="Under ~10 s: normal mobility; 13.5 s or more is the usual fall-risk cut-point in older adults. Timed from leaving the seat, so a stopwatch started at 'go' reads slightly longer",
            ref_ar="أقل من ~10 ثوانٍ: حركة طبيعية؛ 13.5 ثانية فأكثر هي الحد المعتمد لخطر السقوط لدى كبار السن. يبدأ التوقيت من مغادرة المقعد، لذا قراءة ساعة الإيقاف من كلمة «ابدأ» أطول قليلاً")
        chip("Test time", "زمن الاختبار", f"{total:.1f} s", st_t)
        top = float(np.nanpercentile(gain[a:b + 1], 90))
        rs = np.where(gain[a:b + 1] >= 0.9 * top)[0]
        rise = (rs[0]) / fps if len(rs) else 0.0
        info_row("tug_rise", "Time to stand up", "زمن النهوض", f"{rise:.1f} s")
        span = float(np.nanmax(hx[a:b + 1]) - np.nanmin(hx[a:b + 1])) / torso
        info_row("tug_walk", "Distance covered (in torso lengths)", "المسافة المقطوعة (بأطوال الجذع)", f"{span:.1f}")
        key_idx = int((a + b) // 2)
        wins = [(a, key_idx, b)]
        reps = 1
        trunk_sig = S["trunk"]
        plot = dict(title="Hip height gained over time (stand, walk, sit)", ylabel="height (px)", hud="Hip rise",
                    lines=[("Hip height", gain, "#3F6200")])

    elif movement == "posture":
        n_ = len(P)
        key_idx = n_ // 2
        wins = [(0, key_idx, n_ - 1)]
        reps = 1
        if side:
            ear = 7 if side == "L" else 8
            sh_, hp_ = (L_SH, L_HIP) if side == "L" else (R_SH, R_HIP)
            direction = np.sign(np.nanmedian(P[:, 0, 0] - P[:, ear, 0])) or 1.0
            torso = float(np.nanmedian(np.abs(P[:, sh_, 1] - P[:, hp_, 1]))) + 1e-9
            head = (P[:, ear, 0] - P[:, sh_, 0]) * direction / torso * 100
            shf = (P[:, sh_, 0] - P[:, hp_, 0]) * direction / torso * 100
            lean = S[f"trunk_{side}"]
            hv = float(np.nanmedian(head))
            single_row("head_fwd", "Head forward of the shoulder", "تقدّم الرأس أمام الكتف", max(hv, 0.0), "% torso", 15, 25,
                       "Ear should sit roughly over the shoulder", "يجب أن تكون الأذن فوق الكتف تقريباً")
            chip("Head forward", "تقدّم الرأس", f"{max(hv, 0):.0f}%", rows[-1]["status"])
            sv = float(np.nanmedian(shf))
            single_row("sh_fwd", "Shoulder forward of the hip", "تقدّم الكتف أمام الحوض", abs(sv), "% torso", 10, 20,
                       "Shoulder should stack over the hip", "يجب أن يكون الكتف فوق الحوض")
            lv = float(np.nanmedian(lean))
            single_row("trunk", "Trunk lean from vertical", "ميلان الجذع عن العمودي", lv, "deg", 5, 10,
                       "Should stay near vertical", "يجب أن يبقى قريباً من العمودي")
            chip("Trunk lean", "ميلان الجذع", f"{lv:.0f}°", rows[-1]["status"])
            f = head
            plot = dict(title="Head position over time (% of torso length)", ylabel="head forward (%)", hud="Head fwd",
                        lines=[("Head forward", head, "#3F6200")])
        else:
            def tilt(a_, b_):
                dd = P[:, b_] - P[:, a_]
                return np.degrees(np.arctan2(dd[:, 1], np.abs(dd[:, 0]) + 1e-9))
            shw = float(np.nanmedian(np.abs(P[:, L_SH, 0] - P[:, R_SH, 0]))) + 1e-9
            sh_t, hp_t, hd_t = tilt(L_SH, R_SH), tilt(L_HIP, R_HIP), tilt(7, 8)
            shift = ((P[:, L_SH, 0] + P[:, R_SH, 0]) / 2 - (P[:, L_HIP, 0] + P[:, R_HIP, 0]) / 2) / shw * 100
            v1 = abs(float(np.nanmedian(sh_t)))
            single_row("sh_tilt", "Shoulder level (tilt)", "مستوى الكتفين (الميلان)", v1, "deg", 3, 5,
                       "Shoulders should be level", "يجب أن يكون الكتفان بمستوى واحد")
            chip("Shoulder tilt", "ميلان الكتفين", f"{v1:.1f}°", rows[-1]["status"])
            v2 = abs(float(np.nanmedian(hp_t)))
            single_row("hip_tilt", "Pelvis level (tilt)", "مستوى الحوض (الميلان)", v2, "deg", 3, 5,
                       "Hips should be level", "يجب أن يكون الحوض مستوياً")
            chip("Pelvis tilt", "ميلان الحوض", f"{v2:.1f}°", rows[-1]["status"])
            v3 = abs(float(np.nanmedian(hd_t)))
            single_row("head_tilt", "Head tilt", "ميلان الرأس", v3, "deg", 3, 5,
                       "Head should sit level", "يجب أن يكون الرأس مستوياً")
            v4 = abs(float(np.nanmedian(shift)))
            single_row("shift", "Sideways shift of the trunk over the pelvis", "إزاحة الجذع جانبياً فوق الحوض", v4, "% shoulder width", 5, 10,
                       "Shoulders should sit centred over the hips", "يجب أن يتمركز الكتفان فوق الحوض")
            chip("Trunk shift", "إزاحة الجذع", f"{v4:.0f}%", rows[-1]["status"])
            f = sh_t
            plot = dict(title="Shoulder line tilt over time", ylabel="tilt (deg)", hud="Shoulder tilt",
                        lines=[("Shoulders", sh_t, "#3F6200"), ("Pelvis", hp_t, "#f2994a")])
        notes.append(("Stand relaxed and still for 3-5 seconds, whole body in view, fitted clothing. A photo also works.",
                      "قف مسترخياً وثابتاً لمدة ٣-٥ ثوانٍ، الجسم كاملاً في الإطار وبملابس ضيقة. تصلح الصورة أيضاً."))

    elif movement == "neck":
        n_ = len(P)
        def range_row(key, en, ar, v, good, watch, ref_en, ref_ar):
            stt = "ok" if v >= good else ("watch" if v >= watch else "flag")
            return add(key, en, ar, stt, value=round(float(v), 1), unit="deg", ref_en=ref_en, ref_ar=ref_ar)
        if side:
            ear = 7 if side == "L" else 8
            dx = np.abs(P[:, 0, 0] - P[:, ear, 0]) + 1e-9
            ang_ = np.degrees(np.arctan2(P[:, 0, 1] - P[:, ear, 1], dx))       # + = nose lower than ear (looking down)
            ang_ = ang_ - np.median(ang_[:max(5, int(fps))])
            flex, ext = float(np.percentile(ang_, 98)), float(-np.percentile(ang_, 2))
            r1 = range_row("neck_flex", "Neck flexion (chin down)", "ثني الرقبة (الذقن للأسفل)", flex, 35, 20,
                           "Approximate normal ~45-50 deg", "الطبيعي التقريبي ~45-50 درجة")
            r2 = range_row("neck_ext", "Neck extension (looking up)", "مد الرقبة (النظر للأعلى)", ext, 40, 25,
                           "Approximate normal ~55-70 deg", "الطبيعي التقريبي ~55-70 درجة")
            chip("Flexion", "الثني", f"{flex:.0f}°", r1["status"])
            chip("Extension", "المد", f"{ext:.0f}°", r2["status"])
            f = ang_
            plot = dict(title="Head tilt up/down over time", ylabel="angle (deg)", hud="Head tilt", lines=[("Head", ang_, "#3F6200")])
            notes.append(("Side-view neck angles are approximate (head tilt from the ear-nose line).",
                          "زوايا الرقبة من الجانب تقريبية (ميلان الرأس من خط الأذن-الأنف)."))
        else:
            tilt = np.degrees(np.arctan2(P[:, 8, 1] - P[:, 7, 1], np.abs(P[:, 8, 0] - P[:, 7, 0]) + 1e-9))
            tilt = tilt - np.median(tilt[:max(5, int(fps))])
            ew = np.abs(P[:, 8, 0] - P[:, 7, 0]) + 1e-9
            rot = np.degrees(np.arcsin(np.clip((P[:, 0, 0] - (P[:, 7, 0] + P[:, 8, 0]) / 2) / (ew / 2 + 1e-9), -1, 1)))
            rot = rot - np.median(rot[:max(5, int(fps))])
            tl, tr_ = float(-np.percentile(tilt, 2)), float(np.percentile(tilt, 98))
            rl, rr = float(np.percentile(rot, 98)), float(-np.percentile(rot, 2))   # image right = the person's left
            if max(tl, tr_) < 10:      # tilt was not performed in this clip: report, do not grade
                r1 = info_row("neck_tilt", "Sideways neck tilt", "ميلان الرقبة جانبياً", "not performed in this video")
                r1["asym"] = "-"
            else:
                r1 = pair_row("neck_tilt", "Sideways neck tilt", "ميلان الرقبة جانبياً", tl, tr_, "deg",
                              "Approximate normal ~40-45 deg each side", "الطبيعي التقريبي ~40-45 درجة لكل جانب", 30, 20)
            r2 = pair_row("neck_rot", "Neck rotation", "دوران الرقبة", rl, rr, "deg",
                          "Approximate normal ~70-80 deg each side (underestimated from a single camera)",
                          "الطبيعي التقريبي ~70-80 درجة لكل جانب (مقدّر بأقل من الحقيقة بكاميرا واحدة)")
            chip("Tilt L/R", "الميلان", f"{r1['asym']}%" if r1["status"] != "info" else "n/a", r1["status"])
            if r1["status"] == "info":
                r1["asym"] = None
            chip("Rotation L/R", "الدوران", f"{r2['asym']}%", r2["status"])
            f = tilt
            plot = dict(title="Sideways head tilt over time", ylabel="tilt (deg)", hud="Head tilt", lines=[("Tilt", tilt, "#3F6200"), ("Rotation", rot, "#f2994a")])
            notes.append(("Left/right are the person's own sides (do not use a mirrored selfie camera). Neck angles are rough "
                          "estimates from face points; rotation is the least reliable.",
                          "اليمين/اليسار هما جانبا الشخص نفسه (لا تستخدم كاميرا السيلفي المعكوسة). زوايا الرقبة تقديرات تقريبية من نقاط الوجه، والدوران هو الأقل موثوقية."))
        key_idx = int(np.argmax(np.abs(f)))
        wins = [(0, key_idx, n_ - 1)]
        reps = 1

    elif movement == "wrist":
        sides_ = [side] if side else ["L", "R"]
        dev = {}
        for sd in sides_:
            el, wr, ix = (L_EL, L_WR, 19) if sd == "L" else (R_EL, R_WR, 20)
            dev[sd] = 180 - ang(P[:, el], P[:, wr], P[:, ix])
        f = dev[side] if side else np.maximum(dev["L"], dev["R"])
        pk, _ = find_peaks(f, prominence=15, distance=max(1, int(fps * 0.6)))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(f))])
            notes.append(("No clear repetitions were detected; the largest bend in the video was used.",
                          "لم يتم رصد تكرارات واضحة؛ تم اعتماد أكبر ثني في الفيديو."))
        wins = rep_windows(f, pk, thr=6)
        reps = len(pk)
        key_idx = int(pk[np.argmax(f[pk])])
        ref_en, ref_ar = "Normal ~70-80 deg flexion or ~60-70 deg extension", "الطبيعي ~70-80 درجة ثني أو ~60-70 درجة مد"
        if side:
            m = float(np.mean(dev[side][pk]))
            r1 = side_row("wrist_bend", "Maximum wrist bend from neutral", "أقصى ثني للرسغ عن الوضع المحايد", m, side, 50, 30, ref_en, ref_ar)
            chip("Wrist bend", "ثني الرسغ", f"{m:.0f}°", r1["status"])
        else:
            mL, mR = float(np.percentile(dev["L"], 98)), float(np.percentile(dev["R"], 98))
            r1 = pair_row("wrist_bend", "Maximum wrist bend from neutral", "أقصى ثني للرسغ عن الوضع المحايد", mL, mR, "deg", ref_en, ref_ar)
            chip("Left/right wrist", "تماثل الرسغ", f"{r1['asym']}%", r1["status"])
        trunk_sig = S["trunk"]
        plot = dict(title="Wrist bend from neutral over time", ylabel="bend (deg)", hud="Wrist",
                    lines=[((("Left" if sd == "L" else "Right")), dev[sd], "#3F6200" if side else ("#2f80ed" if sd == "L" else "#f2994a")) for sd in sides_])
        notes.append(("Wrist tracking is the least reliable: film close up, forearm still, hand fully in view. "
                      "Flexion and extension cannot be told apart here, so film them as two separate movements.",
                      "تتبع الرسغ هو الأقل موثوقية: صوّر عن قرب مع ثبات الساعد وظهور اليد كاملة. لا يمكن التمييز بين الثني والمد هنا، لذا صوّرهما كحركتين منفصلتين."))
    else:
        sys.exit("unknown movement")

    # per-repetition table
    rep_table = []
    for n, (a, i, b) in enumerate(wins, 1):
        rep_table.append(dict(n=n, start=round(a / fps, 1), down=round((i - a) / fps, 1), up=round((b - i) / fps, 1),
                              peak=round(float(f[i]), 1), trunk=round(float(trunk_sig[i]), 1)))
    if wins:
        k = [w for w in wins if w[1] == key_idx] or wins[:1]
        a, i, b = k[0]
    else:
        a, b = max(0, key_idx - int(fps)), min(len(f) - 1, key_idx + int(fps))
    frames_idx = dict(start=int(a), mid=int((a + key_idx) // 2), key=int(key_idx), end=int(b))

    if side:
        notes.append(("Side view: only the limb facing the camera (" + SIDE_EN[side] + ") can be tracked, so a left/right "
                      "comparison is not possible. Film the other side separately to compare.",
                      "التصوير الجانبي: يمكن تتبع الطرف المواجه للكاميرا فقط (" + SIDE_AR[side] + ")، لذلك لا تمكن المقارنة بين "
                      "الجانبين. صوّر الجانب الآخر على حدة للمقارنة."))
    R = {r["key"]: r for r in rows}
    checks = clinician_checks(movement, side, R)
    return dict(rows=rows, notes=notes, key_idx=key_idx, reps=reps, series=S, side=side, rep_table=rep_table,
                chips=chips, frames_idx=frames_idx, checks=checks, wins=wins, plot=plot)


def clinician_checks(movement, side, R):
    c = []
    bad = lambda k: k in R and R[k]["status"] in ("watch", "flag")
    if movement == "squat":
        if bad("squat_flex") and R["squat_flex"]["asym"] is None:
            c.append(("Limited squat depth: consider assessing ankle dorsiflexion, hip mobility, pain at end range, and "
                      "quadriceps/gluteal strength.",
                      "عمق قرفصاء محدود: يُنصح بتقييم حركة الكاحل (ثني ظهر القدم) ومرونة الورك والألم عند نهاية المدى وقوة العضلة الرباعية والألوية."))
        if bad("squat_flex") and R["squat_flex"]["asym"] is not None:
            c.append(("Left/right difference: consider assessing strength, pain, weight shifting and joint range on the "
                      "side with less motion.",
                      "فرق بين الجانبين: يُنصح بتقييم القوة والألم وانتقال الوزن ومدى حركة المفصل في الجانب الأقل حركة."))
        if bad("trunk"):
            c.append(("Marked trunk lean: consider assessing ankle mobility, hip mobility, trunk control, and the load used.",
                      "ميلان واضح للجذع: يُنصح بتقييم حركة الكاحل ومرونة الورك والتحكم بالجذع والحمل المستخدم."))
        if bad("valgus_L") or bad("valgus_R"):
            c.append(("Knee drifting inward: consider assessing hip abductor and external rotator strength, and foot posture.",
                      "ميل الركبة للداخل: يُنصح بتقييم قوة العضلات المبعِّدة والمدوِّرة للخارج للورك ووضعية القدم."))
        if bad("rep_range"):
            c.append(("Depth varies between repetitions: may reflect fatigue, pain, or limited movement control.",
                      "يختلف العمق بين التكرارات: قد يعكس التعب أو الألم أو ضعف التحكم بالحركة."))
    elif movement == "shoulder_raise":
        if bad("sh_elev"):
            c.append(("Reduced arm elevation on one side: consider assessing glenohumeral and scapular motion, pain, "
                      "rotator cuff strength and cervical contribution.",
                      "انخفاض رفع الذراع في جانب واحد: يُنصح بتقييم حركة مفصل الكتف وعظم اللوح والألم وقوة الكفة المدوّرة ومساهمة الرقبة."))
        if bad("hike") or bad("trunk"):
            c.append(("Compensation (shoulder hiking or trunk lean) suggests the arm may not be elevating through shoulder "
                      "motion alone.",
                      "التعويض (رفع الكتف أو ميلان الجذع) يشير إلى أن رفع الذراع قد لا يتم بحركة الكتف وحدها."))
    elif movement == "knee_bend":
        if bad("knee_flex"):
            c.append(("Reduced knee flexion: consider assessing swelling, pain, and quadriceps/hamstring tightness.",
                      "انخفاض ثني الركبة: يُنصح بتقييم التورم والألم وشدّ العضلة الرباعية والأوتار الخلفية."))
    elif movement == "elbow_flexion":
        if bad("elbow_flex"):
            c.append(("Reduced elbow flexion: consider assessing pain, swelling, joint stiffness, and biceps/brachialis strength.",
                      "انخفاض ثني المرفق: يُنصح بتقييم الألم والتورم وتيبس المفصل وقوة العضلة ذات الرأسين والعضدية."))
        if bad("elbow_ext"):
            c.append(("Elbow cannot fully straighten: consider assessing joint effusion, capsular tightness, pain at end range "
                      "and triceps strength.",
                      "المرفق لا يستقيم بالكامل: يُنصح بتقييم الانصباب المفصلي وتيبس المحفظة والألم عند نهاية المدى وقوة العضلة ثلاثية الرؤوس."))
        if bad("trunk") or bad("arm_swing"):
            c.append(("Compensation (trunk swing or shoulder movement) suggests momentum is helping the curl; consider lighter "
                      "load or assessing trunk and shoulder control.",
                      "التعويض (تأرجح الجذع أو حركة الكتف) يشير إلى أن الزخم يساعد الحركة؛ فكّر بحمل أخف أو قيّم تحكم الجذع والكتف."))
    elif movement == "hip_flexion":
        if bad("hip_flex"):
            c.append(("Reduced hip flexion: consider assessing hip joint mobility, hip flexor strength, pain or pinching at "
                      "end range, and lumbar-pelvic compensation.",
                      "انخفاض ثني الورك: يُنصح بتقييم حركة مفصل الورك وقوة العضلات القابضة للورك والألم أو الانحشار عند نهاية المدى والتعويض من أسفل الظهر والحوض."))
        if bad("trunk"):
            c.append(("Trunk leaning to lift the knee suggests compensation; consider assessing core and hip control.",
                      "ميلان الجذع لرفع الركبة يشير إلى تعويض؛ يُنصح بتقييم التحكم بعضلات الجذع والورك."))
    elif movement == "sit_to_stand":
        if bad("sts_time"):
            c.append(("Slow sit-to-stand: consider assessing quadriceps and gluteal strength, balance, pain, and fear of falling; "
                      "compare with age-based norms.",
                      "بطء في الجلوس والوقوف: يُنصح بتقييم قوة العضلة الرباعية والألوية والتوازن والألم والخوف من السقوط ومقارنتها بالمعايير حسب العمر."))
        if bad("sts_ext"):
            c.append(("Not standing fully upright: consider assessing hip/knee extension strength, pain, and balance.",
                      "عدم الوقوف باعتدال كامل: يُنصح بتقييم قوة مد الورك والركبة والألم والتوازن."))
        if bad("trunk"):
            c.append(("Marked forward lean while rising: consider assessing ankle mobility and quadriceps strength.",
                      "ميلان واضح للأمام أثناء النهوض: يُنصح بتقييم حركة الكاحل وقوة العضلة الرباعية."))
    elif movement == "lunge":
        if bad("lunge_front"):
            c.append(("Limited lunge depth: consider assessing pain, ankle and hip mobility, and quadriceps/gluteal strength.",
                      "عمق محدود في الطعنة: يُنصح بتقييم الألم وحركة الكاحل والورك وقوة العضلة الرباعية والألوية."))
        if bad("valgus"):
            c.append(("Front knee drifting inward: consider assessing hip abductor/external rotator strength and foot posture.",
                      "ميل الركبة الأمامية للداخل: يُنصح بتقييم قوة العضلات المبعِّدة والمدوِّرة للخارج للورك ووضعية القدم."))
        if bad("trunk"):
            c.append(("Marked trunk lean: consider assessing balance, hip mobility and trunk control.",
                      "ميلان واضح للجذع: يُنصح بتقييم التوازن ومرونة الورك والتحكم بالجذع."))
    elif movement == "single_leg_balance":
        if bad("hold"):
            c.append(("Short single-leg hold: consider assessing balance/proprioception, ankle stability, hip abductor strength and pain.",
                      "قِصر مدة الثبات على ساق واحدة: يُنصح بتقييم التوازن والحس العميق وثبات الكاحل وقوة العضلات المبعِّدة للورك والألم."))
        if bad("pelvic_drop"):
            c.append(("Pelvis drops on the lifted side (Trendelenburg pattern): consider assessing gluteus medius strength of the standing leg.",
                      "هبوط الحوض في جهة الساق المرفوعة (نمط ترندلنبرغ): يُنصح بتقييم قوة العضلة الألوية الوسطى في الساق الثابتة."))
        if bad("sway") or bad("trunk"):
            c.append(("Large sway or trunk lean: may be compensating for hip weakness or a balance deficit.",
                      "تمايل كبير أو ميلان الجذع: قد يكون تعويضاً عن ضعف الورك أو خلل في التوازن."))
    elif movement == "shoulder_flexion":
        if bad("sh_flex"):
            c.append(("Reduced forward arm elevation: consider assessing glenohumeral and scapular motion, pain, rotator cuff strength and thoracic extension.",
                      "انخفاض رفع الذراع للأمام: يُنصح بتقييم حركة مفصل الكتف وعظم اللوح والألم وقوة الكفة المدوّرة ومرونة الفقرات الصدرية."))
        if bad("trunk"):
            c.append(("Leaning back to lift suggests compensation for limited shoulder motion.",
                      "الميل للخلف أثناء الرفع يشير إلى تعويض عن محدودية حركة الكتف."))
    elif movement == "hip_abduction":
        if bad("hip_abd"):
            c.append(("Reduced hip abduction: consider assessing hip joint mobility, gluteus medius strength, adductor tightness and pain.",
                      "انخفاض تبعيد الورك: يُنصح بتقييم حركة مفصل الورك وقوة الألوية الوسطى وشدّ العضلات المقرِّبة والألم."))
        if bad("trunk"):
            c.append(("Sideways trunk lean may be substituting for hip motion.", "الميل الجانبي للجذع قد يحل محل حركة الورك."))
    elif movement == "straight_leg_raise":
        if bad("slr"):
            c.append(("Limited straight leg raise: consider assessing hamstring length, neural tension, lumbar-pelvic control and hip flexor strength.",
                      "محدودية رفع الساق المستقيمة: يُنصح بتقييم طول الأوتار الخلفية والشدّ العصبي والتحكم القطني-الحوضي وقوة القابضات."))
        if bad("knee_lag"):
            c.append(("Knee bends during the raise: consider assessing quadriceps strength (extensor lag) and hamstring tightness.",
                      "ينثني الركبة أثناء الرفع: يُنصح بتقييم قوة العضلة الرباعية والأوتار الخلفية."))
    elif movement == "gait":
        if bad("cadence"):
            c.append(("Cadence outside the usual range: consider assessing pain, strength, balance and walking aids.",
                      "إيقاع المشي خارج المعتاد: يُنصح بتقييم الألم والقوة والتوازن ووسائل المساعدة على المشي."))
        if bad("step_sym"):
            c.append(("Uneven step timing between sides: consider assessing pain, weight-bearing tolerance and strength on the quicker-stepping side.",
                      "اختلاف زمن الخطوة بين الجانبين: يُنصح بتقييم الألم وتحمل الوزن والقوة في الجانب الأسرع خطوة."))
    elif movement == "tug":
        if bad("tug_time"):
            c.append(("Slow Timed Up and Go: consider assessing lower-limb strength, balance, gait speed, pain and fall risk; compare with age norms.",
                      "بطء في اختبار الوقوف والمشي: يُنصح بتقييم قوة الطرفين السفليين والتوازن وسرعة المشي والألم وخطر السقوط ومقارنتها بالمعايير حسب العمر."))
    elif movement == "posture":
        if bad("head_fwd") or bad("sh_fwd"):
            c.append(("Forward head or shoulder position: consider assessing neck and upper-back posture, deep neck flexor and scapular control.",
                      "تقدّم الرأس أو الكتف: يُنصح بتقييم وضعية الرقبة والظهر العلوي وقوة العضلات القابضة العميقة للرقبة والتحكم باللوح."))
        if bad("sh_tilt") or bad("hip_tilt") or bad("shift"):
            c.append(("Uneven shoulder or pelvis level, or a trunk shift: consider assessing leg length, scoliosis screening and muscle imbalance.",
                      "عدم تساوي مستوى الكتفين أو الحوض أو إزاحة الجذع: يُنصح بتقييم طول الساقين وفحص الجنف والخلل العضلي."))
    elif movement == "neck":
        if any(bad(k) for k in ("neck_flex", "neck_ext", "neck_tilt", "neck_rot")):
            c.append(("Reduced neck motion in a direction: consider assessing pain, muscle guarding, cervical joint mobility and neurological screening as indicated.",
                      "انخفاض حركة الرقبة في اتجاه ما: يُنصح بتقييم الألم والتشنج العضلي وحركة المفاصل العنقية والفحص العصبي عند الحاجة."))
    elif movement == "wrist":
        if bad("wrist_bend"):
            c.append(("Reduced wrist bend: consider assessing pain, swelling, joint stiffness and forearm muscle tightness; confirm with a goniometer.",
                      "انخفاض ثني الرسغ: يُنصح بتقييم الألم والتورم وتيبس المفصل وشدّ عضلات الساعد والتأكد بمقياس الزوايا."))
    if movement == "knee_bend" and bad("knee_ext"):
        c.append(("Knee does not fully straighten: consider assessing swelling/effusion, hamstring tightness, pain at end range "
                  "and quadriceps strength (extensor lag).",
                  "الركبة لا تستقيم بالكامل: يُنصح بتقييم التورم والانصباب وشدّ الأوتار الخلفية والألم عند نهاية المدى وقوة العضلة الرباعية."))
    if not c:
        c.append(("No notable deviations in the measured items. Re-test with the same set-up to track change over time.",
                  "لا توجد انحرافات ملحوظة في المقاييس المقيسة. أعد الاختبار بنفس الإعداد لمتابعة التغير مع الوقت."))
    return c


def findings(movement, view, rows):
    out = []
    for r in rows:
        if r["status"] == "info":
            continue
        if r.get("side"):
            if movement != "squat":
                out.append((r["status"],
                            f"{r['en']} ({SIDE_EN[r['side']]}): {r['value']} deg - {STATUS_EN[r['status']].lower()}.",
                            f"{r['ar']} (الجانب {SIDE_AR[r['side']]}): {r['value']}° - {STATUS_AR[r['status']]}."))
        elif r["asym"] is not None:
            weaker = "L" if r["left"] < r["right"] else "R"
            if r.get("low"):
                out.append((r["status"],
                            f"{r['en']}: both sides are below the expected range (left {r['left']} deg, right {r['right']} deg).",
                            f"{r['ar']}: الجانبان أقل من المدى المتوقع (الأيسر {r['left']}° والأيمن {r['right']}°)."))
            elif r["status"] == "ok":
                out.append(("ok", f"{r['en']}: left {r['left']} deg vs right {r['right']} deg - balanced ({r['asym']}% difference).",
                            f"{r['ar']}: الأيسر {r['left']}° مقابل الأيمن {r['right']}° - متوازن (فرق {r['asym']}%)."))
            else:
                out.append((r["status"],
                            f"{r['en']}: the {SIDE_EN[weaker]} side shows less range ({r['left']} deg left vs {r['right']} deg right, {r['asym']}% difference).",
                            f"{r['ar']}: الجانب {SIDE_AR[weaker]} أقل مدى (الأيسر {r['left']}° مقابل الأيمن {r['right']}°، فرق {r['asym']}%)."))
        elif r["status"] != "ok":
            out.append((r["status"], f"{r['en']}: {r['value']} {r['unit']} - {STATUS_EN[r['status']].lower()}.",
                        f"{r['ar']}: {r['value']} - {STATUS_AR[r['status']]}."))
    if movement == "squat" and rows and rows[0]["key"] == "squat_flex":
        r0 = rows[0]
        d = float(np.mean([r0[k] for k in ("left", "right") if r0.get(k) is not None]))
        sd = r0.get("side")
        sfx_en = f" ({SIDE_EN[sd]} side)" if sd else ""
        sfx_ar = f" (الجانب {SIDE_AR[sd]})" if sd else ""
        if d >= 85:
            cat = ("reached parallel or deeper", "وصل إلى مستوى الموازاة أو أعمق", "ok")
        elif d >= 60:
            cat = ("partial depth", "عمق جزئي", "watch")
        else:
            cat = ("shallow depth", "عمق محدود", "flag")
        out.insert(0, (cat[2], f"Squat depth{sfx_en}: average knee flexion {d:.1f} deg - {cat[0]}.",
                       f"عمق القرفصاء{sfx_ar}: متوسط ثني الركبة {d:.1f}° - {cat[1]}."))
    return out


# ---------------------------------------------------------------- rendering
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_angle(img, A, B, C, label, color):
    A, B, C = (np.asarray(x, float) for x in (A, B, C))
    if np.isnan([A, B, C]).any():
        return
    a1 = math.degrees(math.atan2(A[1] - B[1], A[0] - B[0]))
    a2 = math.degrees(math.atan2(C[1] - B[1], C[0] - B[0]))
    d = (a2 - a1 + 540) % 360 - 180
    s, e = (a1, a1 + d) if d >= 0 else (a1 + d, a1)
    r = int(max(14, 0.07 * img.shape[0]))
    cv2.ellipse(img, tuple(B.astype(int)), (r, r), 0, s, e, color, 2, cv2.LINE_AA)
    mid = math.radians(a1 + d / 2)
    tx, ty = int(B[0] + (r + 5) * math.cos(mid)), int(B[1] + (r + 5) * math.sin(mid))
    cv2.putText(img, label, (tx, ty), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(video_path, P, fps, size, movement, comp, out_path, crop_paths, side=None):
    W, H = size
    S = comp["series"]
    cap = cv2.VideoCapture(video_path)
    tmp = out_path + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    blue, orange, white = (255, 160, 40), (40, 140, 255), (255, 255, 255)
    want = {}
    for k in ("start", "mid", "key", "end"):
        want.setdefault(comp["frames_idx"][k], []).append(k)
    base_keys = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP] if movement == "elbow_flexion" else KEYS
    used = [j for j in base_keys if not side or (j % 2 == 1) == (side == "L")]
    pts = P[:, used, :]
    x0, y0 = np.nanmin(pts[:, :, 0]), np.nanmin(pts[:, :, 1])
    x1, y1 = np.nanmax(pts[:, :, 0]), np.nanmax(pts[:, :, 1])
    mx, my = 0.75 * (x1 - x0) + 30, 0.18 * (y1 - y0) + 20
    cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    cx1, cy1 = int(min(W, x1 + mx)), int(min(H, y1 + my))

    def text(img, s, x, y, scale=0.6):
        cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), FONT, scale, white, 1, cv2.LINE_AA)

    i = 0
    last_crop = None
    while True:
        ok, fr = cap.read()
        if not ok or i >= len(P):
            break
        fr = cv2.resize(fr, (W, H))
        for a, b in BONES:
            if np.isnan(P[i, a]).any() or np.isnan(P[i, b]).any():
                continue
            if side and ((a % 2 == 1) != (side == "L") or (b % 2 == 1) != (side == "L")):
                continue
            col = blue if (a % 2 == 1 and b % 2 == 1) else (orange if (a % 2 == 0 and b % 2 == 0) else white)
            cv2.line(fr, tuple(P[i, a].astype(int)), tuple(P[i, b].astype(int)), col, 3, cv2.LINE_AA)
        for j in used:
            if not np.isnan(P[i, j]).any():
                cv2.circle(fr, tuple(P[i, j].astype(int)), 5, white, -1, cv2.LINE_AA)
        if movement in ("shoulder_raise", "shoulder_flexion"):
            sides = [side] if side else ["L", "R"]
            for sd in sides:
                h_, s_, e_ = (L_HIP, L_SH, L_EL) if sd == "L" else (R_HIP, R_SH, R_EL)
                draw_angle(fr, P[i, h_], P[i, s_], P[i, e_], f"{S['sh_' + sd][i]:.0f}", blue if sd == "L" else orange)
            hud = "Arm  " + "  ".join(f"{s_}:{S['sh_' + s_][i]:.0f}" for s_ in sides)
        elif movement in ("hip_abduction", "straight_leg_raise", "gait", "tug", "posture", "neck", "wrist"):
            vals = [ln[1][i] for ln in comp["plot"]["lines"]]
            hud = comp["plot"]["hud"] + " " + "  ".join(f"{v:.0f}" for v in vals if not np.isnan(v))
        elif movement == "hip_flexion":
            sides = [side] if side else ["L", "R"]
            for sd in sides:
                sh_, hp_, kn_ = (L_SH, L_HIP, L_KN) if sd == "L" else (R_SH, R_HIP, R_KN)
                draw_angle(fr, P[i, sh_], P[i, hp_], P[i, kn_], f"{180 - S['hip_' + sd][i]:.0f}", blue if sd == "L" else orange)
            hud = "Hip flexion " + "  ".join(f"{s_}:{180 - S['hip_' + s_][i]:.0f}" for s_ in sides)
        elif movement == "single_leg_balance":
            if not (np.isnan(P[i, L_HIP]).any() or np.isnan(P[i, R_HIP]).any()):
                cv2.line(fr, tuple(P[i, L_HIP].astype(int)), tuple(P[i, R_HIP].astype(int)), (60, 200, 60), 3, cv2.LINE_AA)
            hud = f"Foot lift {comp['plot']['lines'][0][1][i]:.0f}% of leg"
        elif movement == "elbow_flexion":
            sides = [side] if side else ["L", "R"]
            for sd in sides:
                sh_, el_, wr_ = (L_SH, L_EL, L_WR) if sd == "L" else (R_SH, R_EL, R_WR)
                draw_angle(fr, P[i, sh_], P[i, el_], P[i, wr_], f"{180 - S['elb_' + sd][i]:.0f}", blue if sd == "L" else orange)
            hud = "Elbow flexion " + "  ".join(f"{s_}:{180 - S['elb_' + s_][i]:.0f}" for s_ in sides)
        else:
            sides = [side] if side else ["L", "R"]
            for sd in sides:
                h, k, a_ = (L_HIP, L_KN, L_AN) if sd == "L" else (R_HIP, R_KN, R_AN)
                draw_angle(fr, P[i, h], P[i, k], P[i, a_], f"{180 - S['knee_' + sd][i]:.0f}", blue if sd == "L" else orange)
            hud = "Knee flexion " + "  ".join(f"{s_}:{180 - S['knee_' + s_][i]:.0f}" for s_ in sides)
        if i in want:
            crop = fr[cy0:cy1, cx0:cx1]
            sc = 360 / max(1, crop.shape[0])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            for k_ in want[i]:
                cv2.imwrite(crop_paths[k_], crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            last_crop = crop
        text(fr, hud, 8, 22, 0.5)
        text(fr, "Taafi Movement Lab", 8, H - 10, 0.4)
        wr.write(fr)
        i += 1
    cap.release()
    wr.release()
    # safety net: a requested frame that was never reached (shorter video than tracked) reuses the last crop
    for k_, p_ in crop_paths.items():
        if not os.path.exists(p_) and last_crop is not None:
            cv2.imwrite(p_, last_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", out_path], check=False)
        if os.path.exists(out_path):
            os.remove(tmp)
            return
    os.replace(tmp, out_path)


def chart(comp, fps, movement, path, side=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = comp["series"]
    two = movement != "knee_bend"
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(7.2, 3.9 if two else 2.5), dpi=150, sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(len(S["trunk"])) / fps
    ax = axes[0]
    if comp.get("plot"):
        for lab, arr, col in comp["plot"]["lines"]:
            ax.plot(t, arr, color=col, lw=2, label=lab)
        ax.set_ylabel(comp["plot"]["ylabel"])
        ax.set_title(comp["plot"]["title"], fontsize=10)
    elif movement == "shoulder_raise":
        ax.plot(t, S["sh_L"], color="#2f80ed", lw=2, label="Left arm")
        ax.plot(t, S["sh_R"], color="#f2994a", lw=2, label="Right arm")
        ax.set_ylabel("elevation (deg)")
        ax.set_title("Arm elevation over time", fontsize=10)
    elif movement == "elbow_flexion":
        if side:
            ax.plot(t, 180 - S["elb_" + side], color="#3F6200", lw=2.2, label=("Left" if side == "L" else "Right") + " elbow flexion")
        else:
            ax.plot(t, 180 - S["elb_L"], color="#2f80ed", lw=2, label="Left")
            ax.plot(t, 180 - S["elb_R"], color="#f2994a", lw=2, label="Right")
        ax.set_ylabel("flexion (deg)")
        ax.set_title("Elbow flexion / extension over time", fontsize=10)
    elif side:
        ax.plot(t, 180 - S["knee_" + side], color="#3F6200", lw=2.2, label=("Left" if side == "L" else "Right") + " knee flexion")
        ax.set_ylabel("flexion (deg)")
        ax.set_title("Knee flexion / extension over time", fontsize=10)
    else:
        ax.plot(t, 180 - S["knee_L"], color="#2f80ed", lw=2, label="Left")
        ax.plot(t, 180 - S["knee_R"], color="#f2994a", lw=2, label="Right")
        ax.set_ylabel("flexion (deg)")
        ax.set_title("Knee flexion over time", fontsize=10)
    for k, (a, i, b) in enumerate(comp["wins"], 1):
        ax.axvspan(a / fps, b / fps, color="#70B000", alpha=0.10)
        yv = ax.lines[0].get_ydata()[i]
        ax.plot([i / fps], [yv], "o", color="#3F6200", ms=5)
        if side:
            ax.annotate(f"#{k}  {yv:.0f}°", (i / fps, yv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    if two:
        ax2 = axes[1]
        key = "trunk" if (not side) else f"trunk_{side}"
        ax2.plot(t, S[key], color="#70B000", lw=2)
        ax2.set_ylabel("trunk lean (deg)")
        ax2.set_title((comp.get("plot") or {}).get("trunk_title") or
                      ("Trunk lean from vertical (forward)" if side else "Trunk lean from vertical (sideways)"), fontsize=9)
    axes[-1].set_xlabel("seconds")
    for a_ in axes:
        a_.grid(alpha=.25)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_pdf(path, meta, comp, fnd, crops, chart_img, lang, compare=None):
    import arabic_reshaper
    import matplotlib
    from bidi.algorithm import get_display
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader, simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("F", os.path.join(fdir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("FB", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
    TEAL, GREY = (0.247, 0.384, 0.0), (0.4, 0.4, 0.4)   # TEAL = Taafi olive #3F6200
    COL = {"ok": (0.33, 0.56, 0.0), "watch": (0.85, 0.6, 0.1), "flag": (0.8, 0.25, 0.2), "info": (0.45, 0.5, 0.55)}
    Wp, Hp = A4
    M = 40
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("Taafi Movement Lab report")
    both = lang == "both"
    shape = lambda s: get_display(arabic_reshaper.reshape(s))

    def ar_lines(s, font, size, w):
        words, lines, cur = s.split(), [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if pdfmetrics.stringWidth(arabic_reshaper.reshape(trial), font, size) <= w:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return [shape(l) for l in lines]

    st = {"y": Hp - M, "page": 1}

    def footer():
        c.setFont("F", 7.5)
        c.setFillColorRGB(*GREY)
        c.drawString(M, 24, "Screening aid - approximate 2D measurements from a single camera. Not a diagnosis. Review by a clinician required.")
        if both:
            c.drawRightString(Wp - M, 14, shape("أداة فحص مساعدة - قياسات تقريبية ثنائية الأبعاد. ليست تشخيصاً ويلزم مراجعة الطبيب."))
        c.drawRightString(Wp - M, 34, f"Page {st['page']}")

    def newpage():
        footer()
        c.showPage()
        st["page"] += 1
        if os.path.exists(LOGO):
            c.drawImage(ImageReader(LOGO), M, Hp - 40, width=62, height=62 * 180 / 520, mask="auto")
        c.setFillColorRGB(*TEAL)
        c.setFont("FB", 9)
        c.drawRightString(Wp - M, Hp - 26, "Movement Analysis Report")
        c.setFillColorRGB(*GREY)
        c.setFont("F", 8.5)
        c.drawRightString(Wp - M, Hp - 38, f"{meta['patient']}  -  {meta['date']}")
        c.setStrokeColorRGB(*TEAL)
        c.setLineWidth(1.2)
        c.line(M, Hp - 46, Wp - M, Hp - 46)
        c.setLineWidth(1)
        st["y"] = Hp - 62

    def need(h):
        if st["y"] - h < M + 30:
            newpage()

    def en(text, size=9.5, font="F", color=(0, 0, 0), indent=0, lead=1.35):
        c.setFont(font, size)
        c.setFillColorRGB(*color)
        for ln in simpleSplit(text, font, size, Wp - 2 * M - indent):
            need(size * lead)
            c.drawString(M + indent, st["y"] - size, ln)
            st["y"] -= size * lead

    def ar(text, size=10, font="F", color=(0, 0, 0), indent=0, lead=1.5):
        c.setFont(font, size)
        c.setFillColorRGB(*color)
        for ln in ar_lines(text, font, size, Wp - 2 * M - indent):
            need(size * lead)
            c.drawRightString(Wp - M - indent, st["y"] - size, ln)
            st["y"] -= size * lead

    def heading(e, a):
        need(40)
        c.setFillColorRGB(*TEAL)
        c.rect(M, st["y"] - 18, Wp - 2 * M, 18, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("FB", 10.5)
        c.drawString(M + 8, st["y"] - 13, e)
        if both:
            c.drawRightString(Wp - M - 8, st["y"] - 13, shape(a))
        st["y"] -= 26

    # ---------------- page 1 header (Taafi document style: logo, olive tagline, bilingual title)
    if os.path.exists(LOGO):
        c.drawImage(ImageReader(LOGO), M, Hp - 36 - 44, width=127, height=127 * 180 / 520, mask="auto")
    c.setFillColorRGB(*TEAL)
    c.setFont("FB", 8.5)
    c.drawString(M, Hp - 96, "The centre's movement screening report")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("FB", 20)
    c.drawString(M, Hp - 118, "Movement Analysis Report")
    if both:
        c.setFillColorRGB(*TEAL)
        c.setFont("FB", 19)
        c.drawRightString(Wp - M, Hp - 118, shape("تقرير تحليل الحركة"))
        c.setFont("FB", 8.5)
        c.drawRightString(Wp - M, Hp - 96, shape("تقرير الفحص الحركي في المركز"))
    c.setStrokeColorRGB(*TEAL)
    c.setLineWidth(1.2)
    c.line(M, Hp - 128, Wp - M, Hp - 128)
    c.setLineWidth(1)
    st["y"] = Hp - 140

    info = [("Patient", meta["patient"]), ("Date", meta["date"]), ("Movement", MOVES[meta["movement"]][0]),
            ("Camera view", meta["view"] + (f" ({'left' if meta['side'] == 'L' else 'right'} side measured)" if meta.get("side") else "")),
            ("Repetitions", str(comp["reps"])), ("Tracking quality", f"{meta['quality']:.0f}%  |  video {meta['video']}")]
    y_info = st["y"]
    for k, v in info:
        c.setFont("FB", 9)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(M, st["y"] - 9, k + ":")
        c.setFont("F", 9)
        c.drawString(M + 105, st["y"] - 9, v)
        st["y"] -= 13.5
    if both:
        c.setFont("F", 9)
        c.drawRightString(Wp - M, y_info - 9, shape("الحركة: " + MOVES[meta["movement"]][1]))
        c.drawRightString(Wp - M, y_info - 23, shape("زاوية الكاميرا: " + ("جانبية" if meta["view"] == "side" else "أمامية")))
        c.drawRightString(Wp - M, y_info - 37, shape(f"عدد التكرارات: {comp['reps']}"))
    st["y"] -= 8

    # summary chips
    chips = comp["chips"]
    n = len(chips)
    gap = 8
    cw = (Wp - 2 * M - gap * (n - 1)) / n
    ch = 58
    for k, ch_ in enumerate(chips):
        x = M + k * (cw + gap)
        col = COL[ch_["status"]]
        c.setStrokeColorRGB(*col)
        c.setLineWidth(1.4)
        c.setFillColorRGB(col[0] * 0.08 + 0.92, col[1] * 0.08 + 0.92, col[2] * 0.08 + 0.92)
        c.roundRect(x, st["y"] - ch, cw, ch, 6, stroke=1, fill=1)
        c.setFillColorRGB(*GREY)
        c.setFont("F", 7.5)
        c.drawString(x + 8, st["y"] - 13, ch_["en"])
        if both:
            c.drawRightString(x + cw - 8, st["y"] - 13, shape(ch_["ar"]))
        c.setFillColorRGB(*col)
        c.setFont("FB", 17)
        c.drawString(x + 8, st["y"] - 36, ch_["value"])
        c.setFont("F", 7.5)
        c.drawString(x + 8, st["y"] - 50, STATUS_EN[ch_["status"]])
        if both:
            c.drawRightString(x + cw - 8, st["y"] - 50, shape(STATUS_AR[ch_["status"]]))
    c.setLineWidth(1)
    st["y"] -= ch + 12

    # key frames
    imgs = []
    for key, e, a in (("start", "Start", "البداية"), ("mid", "Midway", "منتصف الحركة"),
                      ("key", "Deepest / peak point", "أعمق نقطة"), ("end", "Finish", "النهاية")):
        try:
            imgs.append((ImageReader(crops[key]), e, a))
        except Exception:
            pass
    if imgs:
        hh = 185
        ws = [hh * im.getSize()[0] / im.getSize()[1] for im, _, _ in imgs]
        avail = Wp - 2 * M
        scale = min(1.0, (avail - 8 * (len(ws) - 1)) / sum(ws))
        ws = [w * scale for w in ws]
        gap = (avail - sum(ws)) / max(1, len(ws) - 1)
        x, h2 = M, hh * scale
        for (im, e, a), w2 in zip(imgs, ws):
            c.drawImage(im, x, st["y"] - h2, width=w2, height=h2)
            c.setFillColorRGB(*GREY)
            c.setFont("F", 7.5)
            c.drawString(x, st["y"] - h2 - 9, e)
            if both:
                c.drawRightString(x + w2, st["y"] - h2 - 9, shape(a))
            x += w2 + gap
        st["y"] -= h2 + 20
    # chart
    ci = ImageReader(chart_img)
    iw, ih = ci.getSize()
    cwid = Wp - 2 * M
    chh = cwid * ih / iw
    max_h = st["y"] - (M + 34)
    if chh > max_h:
        cwid, chh = cwid * max_h / chh, max_h
    c.drawImage(ci, M, st["y"] - chh, width=cwid, height=chh)
    st["y"] -= chh + 6

    # ---------------- page 2+
    newpage()
    heading("Results", "النتائج")
    cols = {"en": M, "L": 232, "R": 280, "A": 322, "S": 366}
    c.setFont("FB", 8)
    c.setFillColorRGB(*GREY)
    for k, t in (("en", "Measure"), ("L", "Left"), ("R", "Right"), ("A", "Diff."), ("S", "Status")):
        c.drawString(cols[k], st["y"] - 8, t)
    if both:
        c.drawRightString(Wp - M, st["y"] - 8, shape("المقياس"))
    st["y"] -= 14
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    for r in comp["rows"]:
        h = 38
        need(h)
        c.line(M, st["y"], Wp - M, st["y"])
        y0 = st["y"] - 11
        c.setFillColorRGB(0, 0, 0)
        c.setFont("F", 8.2)
        for i, ln in enumerate(simpleSplit(r["en"], "F", 8.2, 185)[:2]):
            c.drawString(cols["en"], y0 - i * 10, ln)
        c.setFont("F", 7)
        c.setFillColorRGB(*GREY)
        if r["ref_en"]:
            c.drawString(cols["en"], y0 - 23, simpleSplit("Ref: " + r["ref_en"], "F", 7, 340)[0])
        c.setFillColorRGB(0, 0, 0)
        c.setFont("F", 9)
        if r.get("text"):
            c.drawString(cols["L"], y0, r["text"])
        elif r.get("side"):
            c.drawString(cols["L"] if r["side"] == "L" else cols["R"], y0, f"{r['value']}°")
            c.setFillColorRGB(*GREY)
            c.drawString(cols["R"] if r["side"] == "L" else cols["L"], y0, "n/a")
        elif r["asym"] is not None:
            c.drawString(cols["L"], y0, f"{r['left']}°")
            c.drawString(cols["R"], y0, f"{r['right']}°")
            c.drawString(cols["A"], y0, f"{r['asym']}%")
        else:
            c.drawString(cols["L"], y0, f"{r['value']} {r['unit']}")
        if not r.get("text"):
            c.setFillColorRGB(*COL[r["status"]])
            c.circle(cols["S"] + 3, y0 + 3, 3.2, stroke=0, fill=1)
            c.setFont("F", 8.2)
            c.drawString(cols["S"] + 11, y0, STATUS_EN[r["status"]])
        if both:
            c.setFillColorRGB(0, 0, 0)
            c.setFont("F", 8.6)
            for i, ln in enumerate(ar_lines(r["ar"], "F", 8.6, 150)[:2]):
                c.drawRightString(Wp - M, y0 + 1 - i * 10, ln)
            if not r.get("text"):
                c.setFont("F", 7.5)
                c.setFillColorRGB(*COL[r["status"]])
                c.drawRightString(Wp - M, y0 - 22, shape(STATUS_AR[r["status"]]))
        st["y"] -= h
    c.line(M, st["y"], Wp - M, st["y"])
    st["y"] -= 14

    # per-rep table
    if comp["rep_table"]:
        heading("Repetition breakdown", "تفاصيل التكرارات")
        xs = [M, M + 40, M + 110, M + 185, M + 260, M + 350]
        for x, t in zip(xs, ("Rep", "Start (s)", "Down (s)", "Up (s)", "Peak angle", "Trunk lean")):
            c.setFont("FB", 8)
            c.setFillColorRGB(*GREY)
            c.drawString(x, st["y"] - 8, t)
        st["y"] -= 13
        for rt in comp["rep_table"]:
            need(16)
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.line(M, st["y"], Wp - M, st["y"])
            c.setFont("F", 9)
            c.setFillColorRGB(0, 0, 0)
            for x, t in zip(xs, (f"#{rt['n']}", f"{rt['start']}", f"{rt['down']}", f"{rt['up']}", f"{rt['peak']}°", f"{rt['trunk']}°")):
                c.drawString(x, st["y"] - 11, t)
            st["y"] -= 16
        st["y"] -= 8

    # compare
    if compare:
        heading(f"Change since previous session ({compare['date']})", "التغير منذ الجلسة السابقة")
        prev = {r["key"]: r for r in compare["rows"]}
        for x, t in ((M, "Measure"), (250, "Previous"), (340, "Now"), (430, "Change")):
            c.setFont("FB", 8)
            c.setFillColorRGB(*GREY)
            c.drawString(x, st["y"] - 8, t)
        st["y"] -= 13
        fmt = lambda r: (f"L {r['left']} / R {r['right']}" if r.get("asym") is not None else f"{r['value']}")
        for r in comp["rows"]:
            p = prev.get(r["key"])
            if not p or r.get("text") or r["status"] == "info":
                continue
            need(16)
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.line(M, st["y"], Wp - M, st["y"])
            if r.get("asym") is not None and p.get("asym") is not None:
                ch_t = f"L/R diff {r['asym'] - p['asym']:+.1f} pts"
            elif r.get("value") is not None and p.get("value") is not None:
                ch_t = f"{r['value'] - p['value']:+.1f} {r['unit']}"
            else:
                continue
            c.setFont("F", 8.2)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(M, st["y"] - 11, simpleSplit(r["en"], "F", 8.2, 200)[0])
            c.drawString(250, st["y"] - 11, fmt(p))
            c.drawString(340, st["y"] - 11, fmt(r))
            c.setFont("FB", 8.2)
            c.drawString(430, st["y"] - 11, ch_t)
            st["y"] -= 16
        st["y"] -= 8

    # findings
    heading("Findings", "النتائج التفصيلية")
    for sev, e, _a in fnd:
        c.setFillColorRGB(*COL[sev])
        need(14)
        c.circle(M + 3, st["y"] - 7, 2.6, stroke=0, fill=1)
        en(e, 9.2, indent=12)
        st["y"] -= 2
    if both:
        st["y"] -= 4
        for sev, _e, a in fnd:
            y_before = st["y"]
            ar(a, 9.6, indent=12)
            c.setFillColorRGB(*COL[sev])
            c.circle(Wp - M - 3, y_before - 8, 2.6, stroke=0, fill=1)
            st["y"] -= 2
    st["y"] -= 8
    heading("Suggested checks for the clinician", "نقاط مقترحة للفحص السريري")
    for e, a in comp["checks"]:
        c.setFillColorRGB(*TEAL)
        need(14)
        c.circle(M + 3, st["y"] - 7, 2.6, stroke=0, fill=1)
        en(e, 9.2, indent=12)
        st["y"] -= 2
    if both:
        st["y"] -= 4
        for e, a in comp["checks"]:
            y_before = st["y"]
            ar(a, 9.6, indent=12)
            c.setFillColorRGB(*TEAL)
            c.circle(Wp - M - 3, y_before - 8, 2.6, stroke=0, fill=1)
            st["y"] -= 2
    for ne, na in comp["notes"] + meta.get("extra_notes", []):
        st["y"] -= 4
        en("Note: " + ne, 8.5, color=GREY)
        if both:
            ar("ملاحظة: " + na, 9, color=GREY)

    st["y"] -= 14
    need(60)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("F", 9)
    c.drawString(M, st["y"] - 9, "Reviewed by: ______________________      Signature: ______________________")
    if both:
        c.drawRightString(Wp - M, st["y"] - 25, shape("اسم الطبيب المراجع والتوقيع: ______________________"))
    footer()
    c.save()


# --------------------------------------------------------------------- main
def run(video, movement, view, patient, outdir, lang="both", landmarks=None, compare=None):
    os.makedirs(outdir, exist_ok=True)
    lm, fps, size = track(video) if landmarks is None else landmarks
    if len(lm) < 10:
        sys.exit("Video too short.")
    side = visible_side(lm) if (view == "side" and movement in ("squat", "knee_bend", "hip_flexion", "sit_to_stand",
                                                     "straight_leg_raise", "posture", "neck")) else None
    if view == "side" and movement in ("elbow_flexion", "shoulder_flexion", "wrist"):
        side = visible_arm_side(lm)
    if view == "side" and movement == "neck":
        vL, vR = np.nanmean(lm[:, 7, 2]), np.nanmean(lm[:, 8, 2])
        side = "L" if (np.isnan(vR) or vL >= vR) else "R"
    keys = needed(movement, side)
    P, rate = clean(lm, fps, keys)
    if rate < 0.6:
        sys.exit(f"Person clearly detected in only {rate:.0%} of frames. Re-film: full body in frame, "
                 "steady camera, good light, fitted clothing.")
    quality = float(np.nanmean(lm[:, keys, 2])) * 100
    bad_ = ~(np.nan_to_num(lm[:, keys, 2], nan=0.0) >= 0.5).all(axis=1)
    g0, g1 = longest_run(bad_)
    gap_s = (g1 - g0) / fps
    comp = compute(P, fps, movement, view, side)
    comp["chips"].append(dict(en="Tracking quality", ar="جودة التتبع", value=f"{quality:.0f}%",
                              status="ok" if quality >= 85 else "watch"))
    fnd = findings(movement, view, comp["rows"])
    base = os.path.join(outdir, movement)
    crops = {k: f"{base}_{k}.jpg" for k in ("start", "mid", "key", "end")}
    chart_img = base + "_chart.png"
    if video:
        annotate(video, P, fps, size, movement, comp, base + "_annotated.mp4", crops, side)
    chart(comp, fps, movement, chart_img, side)
    extra = []
    if gap_s > 0.5:
        extra.append((f"Tracking was lost for up to {gap_s:.1f} s in one stretch; values there are filled in by estimation and are less reliable.",
                      f"فُقد التتبع لمدة تصل إلى {gap_s:.1f} ثانية متواصلة؛ القيم في تلك الفترة تقديرية وأقل موثوقية."))
    if min(size) < 400:
        extra.append((f"Low video resolution ({size[0]}x{size[1]}): angles are less precise. Film at 720p or higher.",
                      f"دقة الفيديو منخفضة ({size[0]}x{size[1]}): الزوايا أقل دقة. يُفضّل التصوير بدقة 720p أو أعلى."))
    meta = dict(patient=patient, movement=movement, view=view, side=side, date=datetime.date.today().isoformat(),
                quality=quality, video=f"{size[0]}x{size[1]} @ {fps:.0f} fps", extra_notes=extra)
    prev = None
    if compare:
        with open(compare, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("movement") != movement:
            prev = None
    build_pdf(base + "_report.pdf", meta, comp, fnd, crops, chart_img, lang, prev)
    dump = {k: v for k, v in comp.items() if k not in ("series", "wins", "plot")}
    dump.update(detect_rate=round(rate, 3), date=meta["date"], movement=movement, view=view, patient=patient)
    with open(base + "_results.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2, default=float)
    return comp, fnd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--movement", choices=list(MOVES), required=True)
    ap.add_argument("--view", choices=["side", "front"], default="side")
    ap.add_argument("--patient", default="-")
    ap.add_argument("--lang", choices=["both", "en"], default="both")
    ap.add_argument("--compare", help="results.json from a previous session of the same movement")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    comp, fnd = run(a.video, a.movement, a.view, a.patient, a.out, a.lang, compare=a.compare)
    print(f"Done. Reps: {comp['reps']}. Files in: {a.out}/")
    for sev, e, _ in fnd:
        print(f" [{sev}] {e}")
