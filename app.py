"""Taafi Movement Lab - web app (Streamlit), styled in the Taafi visual identity."""
import base64
import hmac
import os
import tempfile

import streamlit as st

import movement_lab as ml

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"squat": ("Squat", "القرفصاء"),
          "shoulder_raise": ("Shoulder raise (arm elevation)", "رفع الذراع"),
          "knee_bend": ("Knee flexion / extension", "ثني ومد الركبة"),
          "elbow_flexion": ("Elbow flexion / extension (arm curl)", "ثني ومد المرفق (الذراع)")}
GREEN, OLIVE = "#70B000", "#3F6200"
CHIP_COL = {"ok": GREEN, "watch": "#D99A1A", "flag": "#C8403A", "info": "#7A8691"}


def analyze_bytes(data, suffix, movement, view, patient, prev=None):
    """Run the pipeline on uploaded bytes. Returns dict of results, or raises SystemExit(message)."""
    with tempfile.TemporaryDirectory() as tmp:
        vp = os.path.join(tmp, "in" + (suffix or ".mp4"))
        with open(vp, "wb") as f:
            f.write(data)
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
    view = st.radio("Camera view", ["side", "front"], horizontal=True, index=1 if movement == "shoulder_raise" else 0,
                    format_func=lambda v: {"side": "Side view · جانبي", "front": "Front view · أمامي"}[v])
    patient = st.text_input("Name or label (optional) · الاسم أو الرمز (اختياري)")

    bar("2 · Video", "٢ · الفيديو")
    with st.expander("How to film · كيفية التصوير"):
        st.markdown(
            "- Phone on a stand, whole body in frame, 2-3 m away, good light, fitted clothes  \n"
            "- 3-5 slow repetitions, 5-30 seconds. Do **not** use a mirrored selfie camera  \n"
            "- **Side view** measures the limb (arm or leg) facing the camera only. Use **front view** to compare left and right")
        st.markdown("<div dir='rtl' style='text-align:right'>"
                    "• الهاتف على حامل، الجسم كاملاً في الإطار، على بعد 2-3 أمتار، إضاءة جيدة وملابس ضيقة<br>"
                    "• من 3 إلى 5 تكرارات بطيئة، من 5 إلى 30 ثانية. لا تستخدم كاميرا السيلفي المعكوسة<br>"
                    "• التصوير الجانبي يقيس الطرف (الذراع أو الساق) المواجه للكاميرا فقط. استخدم التصوير الأمامي للمقارنة بين الجانبين"
                    "</div>", unsafe_allow_html=True)
    up = st.file_uploader("Video", type=["mp4", "mov", "m4v", "avi"], label_visibility="collapsed")
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
                'from one camera. Not a diagnosis; a clinician must review. Videos are processed in memory and not stored.'
                '<div class="ar">تعافي - مركز الشفاء والعلاج الطبيعي &nbsp;|&nbsp; أداة فحص مساعدة بزوايا تقريبية من كاميرا واحدة. '
                'ليست تشخيصاً ويلزم مراجعة الطبيب. تتم معالجة الفيديو في الذاكرة ولا يتم حفظه.</div></div>',
                unsafe_allow_html=True)
