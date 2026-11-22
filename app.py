"""Taafi Movement Lab - web app (Streamlit), styled in the Taafi visual identity."""
import base64
import hmac
import os
import tempfile

import streamlit as st

import movement_lab as ml

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"squat": ("Squat", "القرفصاء"),
          "shoulder_raise": ("Shoulder abduction (arm raised to the side)", "تبعيد الكتف (رفع الذراع جانباً)"),
          "shoulder_flexion": ("Shoulder flexion (arm raised forward)", "ثني الكتف (رفع الذراع للأمام)"),
          "elbow_flexion": ("Elbow flexion / extension (arm curl)", "ثني ومد المرفق (الذراع)"),
          "wrist": ("Wrist bend", "ثني الرسغ"),
          "neck": ("Neck range of motion", "مدى حركة الرقبة"),
          "posture": ("Posture screen (standing still, video or photo)", "فحص وضعية الجسم (وقوفاً، فيديو أو صورة)"),
          "hip_flexion": ("Hip flexion (standing knee raise)", "ثني الورك (رفع الركبة وقوفاً)"),
          "hip_abduction": ("Hip abduction (leg out to the side)", "تبعيد الورك (رفع الساق جانباً)"),
          "straight_leg_raise": ("Straight leg raise", "رفع الساق المستقيمة"),
          "knee_bend": ("Knee flexion / extension", "ثني ومد الركبة"),
          "lunge": ("Lunge", "الطعنة (لانج)"),
          "sit_to_stand": ("Sit-to-stand (5 repetitions)", "الجلوس والوقوف (٥ تكرارات)"),
          "single_leg_balance": ("Single-leg balance", "التوازن على ساق واحدة"),
          "gait": ("Gait (walking)", "المشي"),
          "tug": ("Timed Up and Go", "اختبار الوقوف والمشي الموقّت")}
TIPS = {"squat": ("Side view for depth; front view for knee alignment only.", "من الجانب للعمق، ومن الأمام لمحاذاة الركبة فقط."),
        "shoulder_raise": ("Front view, arms start by the sides, raise both arms out to the sides slowly.", "من الأمام، الذراعان بجانب الجسم ثم ارفعهما جانباً ببطء."),
        "shoulder_flexion": ("Side view, arm by the side, raise it straight forward and up, then lower slowly.", "من الجانب، الذراع بجانب الجسم، ارفعها للأمام وللأعلى ثم أنزلها ببطء."),
        "knee_bend": ("Side view. Bend the knee fully, then straighten fully (standing or seated).", "من الجانب. اثنِ الركبة بالكامل ثم مدّها بالكامل (وقوفاً أو جلوساً)."),
        "elbow_flexion": ("Side view, arm hanging straight, curl up, then lower fully.", "من الجانب، الذراع مستقيمة للأسفل، اثنِها للأعلى ثم أنزلها بالكامل."),
        "wrist": ("Side view, close up, forearm still, hand in view. Film bending up and bending down as separate clips. Least reliable movement.", "من الجانب وعن قرب، الساعد ثابت واليد ظاهرة. صوّر الثني للأعلى وللأسفل كمقطعين منفصلين. الأقل موثوقية."),
        "neck": ("Front view: tilt ear to shoulder and turn left/right. Side view: chin down, then look up. Start neutral for 1 second. Rough estimate.", "من الأمام: ميل الأذن نحو الكتف وأدر الرأس يميناً ويساراً. من الجانب: الذقن للأسفل ثم انظر للأعلى. ابدأ محايداً لثانية. تقدير تقريبي."),
        "posture": ("Stand relaxed and still 3-5 s, whole body in view (front or side). A photo also works.", "قف مسترخياً وثابتاً ٣-٥ ثوانٍ والجسم كاملاً في الإطار (أمامي أو جانبي). تصلح الصورة أيضاً."),
        "hip_flexion": ("Side view, raise the knee toward the chest, 3-5 times each leg.", "من الجانب، ارفع الركبة نحو الصدر من ٣ إلى ٥ مرات لكل ساق."),
        "hip_abduction": ("Front view, standing, lift one leg out to the side keeping the body upright, 3-5 times.", "من الأمام، وقوفاً، ارفع ساقاً جانباً مع إبقاء الجسم منتصباً، من ٣ إلى ٥ مرات."),
        "straight_leg_raise": ("Side view, lying on the back (standard test), raise the straight leg, 3-5 times. Standing is reported but not graded.", "من الجانب، استلقاءً على الظهر (الاختبار القياسي)، ارفع الساق المستقيمة من ٣ إلى ٥ مرات. وقوفاً تُعرض القيمة دون تقييم."),
        "sit_to_stand": ("Side view, arms crossed, stand fully then sit, 5 times. Whole body and chair in frame.", "من الجانب، الذراعان متقاطعتان، وقوف كامل ثم جلوس ٥ مرات، الجسم والكرسي في الإطار."),
        "lunge": ("Side view (depth) or front view (knee alignment), 3-5 slow lunges.", "من الجانب (العمق) أو من الأمام (محاذاة الركبة)، من ٣ إلى ٥ طعنات بطيئة."),
        "single_leg_balance": ("Front view, lift one foot and hold as long as possible (up to 30 s).", "من الأمام، ارفع قدماً واحدة وثبّت أطول وقت ممكن (حتى ٣٠ ثانية)."),
        "gait": ("Side view, walk across the frame at a normal pace, at least 5 steps, whole body in view.", "من الجانب، امشِ عبر الإطار بسرعة عادية، ٥ خطوات على الأقل والجسم كاملاً في الإطار."),
        "tug": ("Side view, 3 m path: start seated, stand, walk, turn, return, sit. Camera sees the whole path.", "من الجانب، مسار ٣ أمتار: ابدأ جالساً ثم قف وامشِ واستدر وعد واجلس. الكاميرا ترى المسار كاملاً.")}
# Only the camera views in which the movement's plane faces the camera. First entry is the default.
VIEWS = {"squat": ["side", "front"], "lunge": ["side", "front"], "neck": ["front", "side"], "posture": ["front", "side"],
         "shoulder_raise": ["front"], "hip_abduction": ["front"], "single_leg_balance": ["front"],
         "shoulder_flexion": ["side"], "elbow_flexion": ["side"], "wrist": ["side"], "hip_flexion": ["side"],
         "straight_leg_raise": ["side"], "knee_bend": ["side"], "sit_to_stand": ["side"], "gait": ["side"], "tug": ["side"]}
GREEN, OLIVE = "#70B000", "#3F6200"
CHIP_COL = {"ok": GREEN, "watch": "#D99A1A", "flag": "#C8403A", "info": "#7A8691"}


def analyze_bytes(data, suffix, movement, view, patient, prev=None):
    """Run the pipeline on uploaded bytes. Returns dict of results, or raises SystemExit(message)."""
    with tempfile.TemporaryDirectory() as tmp:
        vp = os.path.join(tmp, "in" + (suffix or ".mp4"))
        with open(vp, "wb") as f:
            f.write(data)
        if (suffix or "").lower() in (".jpg", ".jpeg", ".png"):
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit("Could not read the image.")
            h0, w0 = img.shape[:2]
            sc = min(1.0, 960 / max(h0, w0))
            img = cv2.resize(img, (int(w0 * sc) // 2 * 2, int(h0 * sc) // 2 * 2))
            vp = os.path.join(tmp, "in.mp4")
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (img.shape[1], img.shape[0]))
            for _ in range(60):
                vw.write(img)
            vw.release()
        out = os.path.join(tmp, "out")
        pp = None
        if prev:
            pp = os.path.join(tmp, "prev.json")
            with open(pp, "wb") as f:
                f.write(prev)
        comp, fnd = ml.run(vp, movement, view, patient or "-", out, "both", compare=pp)
        rd = lambda n: open(os.path.join(out, f"{movement}_{n}"), "rb").read()
        return dict(pdf=rd("report.pdf"), video=rd("annotated.mp4"), chart=rd("chart.png"), json=rd("results.json"),
                    chips=comp["chips"], frames=[rd(k + ".jpg") for k in ("start", "mid", "key", "end")],
                    rows=comp["rows"], findings=fnd, reps=comp["reps"], notes=comp["notes"], side=comp.get("side"),
                    checks=comp["checks"])


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif !important; }}
.stApp {{ background: #FFFFFF; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], header {{ display: none !important; }}
.block-container {{ max-width: 860px; padding-top: 1.2rem; padding-bottom: 3rem; }}
.hdr {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:2px solid {OLIVE};
        padding-bottom:10px; margin-bottom:6px; }}
.hdr img {{ height:58px; }}
.hdr .t {{ text-align:right; line-height:1.15; }}
.hdr .t b {{ font-size:26px; color:#111; }}
.hdr .t span {{ display:block; font-size:22px; color:{OLIVE}; font-weight:700; direction:rtl; }}
.tag {{ display:flex; justify-content:space-between; color:{OLIVE}; font-weight:700; font-size:12.5px; margin-bottom:14px; }}
.bar {{ background:{OLIVE}; color:#fff; display:flex; justify-content:space-between; padding:7px 14px;
        font-weight:600; font-size:15px; margin:22px 0 10px 0; }}
.bar span:last-child {{ direction:rtl; }}
div[data-testid="stBaseButton-primary"] button, button[data-testid="stBaseButton-primary"] {{
  background:{GREEN}; border:0; border-radius:2px; color:#fff; font-weight:700; padding:.55rem 1.6rem; }}
button[data-testid="stBaseButton-primary"]:hover {{ background:#5C9400; color:#fff; }}
button[data-testid="stBaseButton-primary"]:disabled {{ background:#C9D9AE; color:#fff; }}
button[data-testid="stBaseButton-secondary"] {{ border:1.5px solid {OLIVE}; color:{OLIVE}; border-radius:2px; font-weight:600; background:#fff; }}
button[data-testid="stBaseButton-secondary"]:hover {{ background:{OLIVE}; color:#fff; border-color:{OLIVE}; }}
[data-testid="stFileUploaderDropzone"] {{ border:1.5px dashed {OLIVE}; background:#F6FAEE; border-radius:2px; }}
[data-testid="stExpander"] {{ border:1px solid #D9E4C4; border-radius:2px; }}
.stCheckbox, .stRadio {{ accent-color:{OLIVE}; }}
input[type="checkbox"], input[type="radio"] {{ accent-color:{OLIVE}; }}
.chips {{ display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 14px; }}
.chip {{ flex:1 1 170px; border:1.5px solid var(--c); border-radius:6px; padding:10px 12px; background:#FAFDF5; }}
.chip .l, .chip .s {{ display:flex; justify-content:space-between; font-size:11.5px; color:#6B7280; }}
.chip .s {{ color:var(--c); font-weight:600; }}
.chip .l span:last-child, .chip .s span:last-child {{ direction:rtl; }}
.chip .v {{ font-size:28px; font-weight:700; color:var(--c); margin:2px 0; }}
.fnd {{ display:flex; gap:10px; align-items:flex-start; padding:6px 0; border-bottom:1px solid #EEF3E4; font-size:14.5px; }}
.fnd .d {{ width:9px; height:9px; border-radius:50%; margin-top:7px; flex:none; background:var(--c); }}
.fnd .ar {{ direction:rtl; text-align:right; color:#333; }}
.note {{ background:#F6FAEE; border-left:4px solid {GREEN}; padding:8px 12px; font-size:13px; color:#3b3b3b; margin:8px 0; }}
.foot {{ color:#6B7280; font-size:12px; margin-top:28px; border-top:1px solid #DDE6CC; padding-top:10px; }}
.foot .ar {{ direction:rtl; text-align:right; }}
</style>
"""


def bar(en, ar):
    st.markdown(f'<div class="bar"><span>{en}</span><span>{ar}</span></div>', unsafe_allow_html=True)


def gate():
    pw = os.environ.get("APP_PASSWORD", "")
    if not pw or st.session_state.get("ok"):
        return True
    bar("Sign in", "تسجيل الدخول")
    entered = st.text_input("Password / كلمة المرور", type="password")
    if entered and hmac.compare_digest(entered, pw):
        st.session_state["ok"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password - كلمة المرور غير صحيحة")
    return False


icon = os.path.join(HERE, "taafi_icon.png")
logo = os.path.join(HERE, "taafi_logo.png")
st.set_page_config(page_title="Taafi Movement Lab", page_icon=icon if os.path.exists(icon) else "🦵", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
logo_html = f'<img src="data:image/png;base64,{b64(logo)}"/>' if os.path.exists(logo) else "<b>Taafi</b>"
st.markdown(f'<div class="hdr">{logo_html}<div class="t"><b>Movement Lab</b><span>مختبر الحركة</span></div></div>'
            f'<div class="tag"><span>The centre\'s movement screening tool</span><span dir="rtl">أداة الفحص الحركي في المركز</span></div>',
            unsafe_allow_html=True)

if gate():
    bar("1 · Movement", "١ · الحركة")
    movement = st.selectbox("Movement", list(LABELS), format_func=lambda m: f"{LABELS[m][0]}   ·   {LABELS[m][1]}",
                            label_visibility="collapsed")
    view = st.radio("Camera view", VIEWS[movement], horizontal=True, key=f"view_{movement}",
                    format_func=lambda v: {"side": "Side view · جانبي", "front": "Front view · أمامي"}[v])
    st.caption(f"{TIPS[movement][0]}  ·  {TIPS[movement][1]}")
    patient = st.text_input("Name or label (optional) · الاسم أو الرمز (اختياري)")

    bar("2 · Video", "٢ · الفيديو")
    with st.expander("How to film · كيفية التصوير"):
        st.markdown(
            "- Phone on a stand, whole body in frame, 2-3 m away, good light, fitted clothes  \n"
            "- 3-5 slow repetitions, 5-30 seconds. Do **not** use a mirrored selfie camera  \n"
            "- **Side view** measures the limb (arm or leg) facing the camera only. To compare left and right, film each side separately and compare the two reports")
        st.markdown("<div dir='rtl' style='text-align:right'>"
                    "• الهاتف على حامل، الجسم كاملاً في الإطار، على بعد 2-3 أمتار، إضاءة جيدة وملابس ضيقة<br>"
                    "• من 3 إلى 5 تكرارات بطيئة، من 5 إلى 30 ثانية. لا تستخدم كاميرا السيلفي المعكوسة<br>"
                    "• التصوير الجانبي يقيس الطرف (الذراع أو الساق) المواجه للكاميرا فقط. للمقارنة بين الجانبين صوّر كل جانب على حدة وقارن التقريرين"
                    "</div>", unsafe_allow_html=True)
    up = st.file_uploader("Video", type=["mp4", "mov", "m4v", "avi"] + (["jpg", "jpeg", "png"] if movement == "posture" else []),
                           label_visibility="collapsed")
    prev_up = st.file_uploader("Previous session results (.json) - optional, shows progress · نتائج الجلسة السابقة (اختياري)",
                               type=["json"])
    consent = st.checkbox("The person in the video agreed to be filmed and analyzed · وافق الشخص في الفيديو على التصوير والتحليل")
    if st.button("Analyze · تحليل", type="primary", disabled=not (up and consent)):
        with st.spinner("Tracking joints and building the report... · جارٍ تتبع المفاصل وإعداد التقرير"):
            try:
                st.session_state["res"] = analyze_bytes(up.getvalue(), os.path.splitext(up.name)[1], movement, view,
                                                        patient, prev_up.getvalue() if prev_up else None)
                st.session_state["err"] = None
            except SystemExit as e:
                st.session_state["res"], st.session_state["err"] = None, str(e)
    if st.session_state.get("err"):
        st.error(st.session_state["err"])

    r = st.session_state.get("res")
    if r:
        bar("Results", "النتائج")
        chips = "".join(
            f'<div class="chip" style="--c:{CHIP_COL[c["status"]]}"><div class="l"><span>{c["en"]}</span><span>{c["ar"]}</span></div>'
            f'<div class="v">{c["value"]}</div><div class="s"><span>{ml.STATUS_EN[c["status"]]}</span>'
            f'<span>{ml.STATUS_AR[c["status"]]}</span></div></div>' for c in r["chips"])
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        _vc = st.columns([1, 1.4, 1])
        _vc[1].video(r["video"])
        cols = st.columns(4)
        for col, im, cap in zip(cols, r["frames"], ["Start", "Midway", "Deepest / peak", "Finish"]):
            col.image(im, caption=cap, use_container_width=True)
        st.image(r["chart"], use_container_width=True)

        bar("Findings", "النتائج التفصيلية")
        for sev, en, ar in r["findings"]:
            st.markdown(f'<div class="fnd" style="--c:{CHIP_COL[sev]}"><div class="d"></div><div style="flex:1">{en}'
                        f'<div class="ar">{ar}</div></div></div>', unsafe_allow_html=True)
        bar("Suggested checks for the clinician", "نقاط مقترحة للفحص السريري")
        for en, ar in r["checks"]:
            st.markdown(f'<div class="fnd" style="--c:{GREEN}"><div class="d"></div><div style="flex:1">{en}'
                        f'<div class="ar">{ar}</div></div></div>', unsafe_allow_html=True)
        for en, ar in r["notes"]:
            st.markdown(f'<div class="note">{en}<div dir="rtl" style="text-align:right">{ar}</div></div>', unsafe_allow_html=True)

        bar("Download", "التحميل")
        c1, c2, c3 = st.columns(3)
        c1.download_button("PDF report · التقرير", r["pdf"], "taafi_movement_report.pdf", "application/pdf", use_container_width=True)
        c2.download_button("Annotated video · الفيديو", r["video"], "taafi_movement_annotated.mp4", "video/mp4", use_container_width=True)
        c3.download_button("Results .json · النتائج", r["json"], "taafi_movement_results.json", "application/json",
                           use_container_width=True, help="Keep this file and upload it next session to see progress")

    st.markdown('<div class="foot">Taafi - Rehabilitation and Healing Center &nbsp;|&nbsp; Screening aid with approximate 2D angles '
                'from one camera. Not a diagnosis; a clinician must review. Videos are uploaded to the hosting server for analysis and deleted afterwards.'
                '<div class="ar">تعافي - مركز الشفاء والعلاج الطبيعي &nbsp;|&nbsp; أداة فحص مساعدة بزوايا تقريبية من كاميرا واحدة. '
                'ليست تشخيصاً ويلزم مراجعة الطبيب. يُرفع الفيديو إلى خادم الاستضافة للتحليل ثم يُحذف بعد ذلك.</div></div>',
                unsafe_allow_html=True)
