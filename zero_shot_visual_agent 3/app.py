"""Zero-Shot Visual Agent - Streamlit frontend.   Run:  streamlit run app.py"""
import io
import json
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zero-Shot Visual Agent", page_icon="👁️", layout="wide")

from agent import LANGGRAPH_AVAILABLE, VisualAgent          # noqa: E402
from agent import config as C                                # noqa: E402
from agent.graph import TOOL_TITLES                          # noqa: E402
from agent.utils import annotate, load_runs, prepare_image, save_run, to_jsonable   # noqa: E402

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #4f46e5 0%, #0ea5e9 100%); color: white;}
.hero h1 {margin: 0; font-size: 1.9rem; color: white;}
.hero p {margin: .25rem 0 0 0; opacity: .92;}
.badge {display:inline-block; padding: .2rem .7rem; border-radius: 999px; font-weight: 600; font-size: .85rem;}
.b-green {background:#dcfce7; color:#166534;} .b-orange {background:#ffedd5; color:#9a3412;}
.b-grey {background:#e5e7eb; color:#374151;}
</style>
<div class="hero"><h1>👁️ Zero-Shot Visual Agent</h1>
<p>Ask anything about an image. A LangGraph agent chooses the right vision tools (BLIP · OWL-ViT · CLIP · ResNet-50),
cross-checks its own answer, and reports what it is unsure about.</p></div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Starting up (importing PyTorch) ...")
def get_hub():
    from agent.models import ModelHub
    return ModelHub()


@st.cache_resource(show_spinner=False)
def get_agent():
    return VisualAgent(get_hub())


try:
    hub = get_hub()
    agent = get_agent()
except Exception as e:  # torch / transformers missing
    st.error(f"Could not start the model layer: {e}")
    st.info("Install the dependencies with `pip install -r requirements.txt` and restart.")
    st.stop()

EXAMPLES = {
    "🧰 Objects": "What objects are visible in this image?",
    "👥 Count people": "How many people are there?",
    "🦺 Safety": "Are there any potential safety hazards?",
    "🏙️ Scene": "What kind of scene is this?",
}
if "question" not in st.session_state:
    st.session_state["question"] = EXAMPLES["🧰 Objects"]


def _set_question(q):
    st.session_state["question"] = q


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio("Mode", ["General image Q&A", "Safety analysis"],
                          help="Safety analysis always adds person / vehicle / PPE screening.")
    custom_text = st.text_area("Custom zero-shot labels (optional)", placeholder="zebra, bicycle lane, hard hat",
                               help="Comma-separated. Used for CLIP classification and added to the detector vocabulary.")
    threshold = st.slider("Detection threshold", 0.05, 0.50, 0.12, 0.01,
                          help="Lower = more (noisier) boxes, higher = fewer but surer boxes.")
    verify = st.checkbox("Cross-check answer with detector", value=True,
                         help="The agent verifies objects mentioned in the VQA answer with OWL-ViT.")
    with st.expander("Advanced: choose tools manually"):
        manual = st.multiselect("Tools (leave empty = agent decides)", list(TOOL_TITLES),
                                format_func=TOOL_TITLES.get)
    st.divider()
    st.subheader("Models")
    icons = {"loaded": "🟢", "not loaded": "⚪", "failed": "🔴"}
    from agent.models import MODEL_LABELS
    for name, state in hub.status().items():
        st.write(f"{icons[state]} {MODEL_LABELS[name]} — {state}")
        if hub.error(name):
            st.caption(hub.error(name))
    st.caption(f"Device: **{hub.device}** · Orchestration: **{agent.backend}**")
    if st.button("⬇️ Preload all models"):
        with st.spinner("Downloading / loading models (first run ≈ 3 GB)..."):
            hub.preload()
        st.rerun()
    st.caption("Models load lazily — only the ones a question needs.")

custom_labels = [x.strip() for x in custom_text.split(",") if x.strip()]
tab_run, tab_hist, tab_about = st.tabs(["🔍 Analyze", "🕘 History", "ℹ️ About"])


# ---------------------------------------------------------------- result view
def render_result(last):
    res, img = last["result"], last["image"]
    tr = res["tool_results"]
    dets = (tr.get("detection") or {}).get("detections", []) if isinstance(tr.get("detection"), dict) else []
    safety = tr.get("safety") if isinstance(tr.get("safety"), dict) and not tr["safety"].get("error") else None

    badge = {"consistent": ("b-green", "🟢 Tools agree"), "mixed": ("b-orange", "🟠 Tools disagree – review"),
             "unverified": ("b-grey", "⚪ Not cross-checked")}.get(res["agreement"], ("b-grey", "—"))
    st.markdown(f'<span class="badge {badge[0]}">{badge[1]}</span>', unsafe_allow_html=True)
    st.markdown(res["summary"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tools used", len(res["plan"]))
    m2.metric("Objects detected", len(dets))
    m3.metric("Safety flags", sum(f["severity"] in ("low", "medium") for f in safety["findings"]) if safety else "–")
    m4.metric("Time", f"{res['elapsed_seconds']} s")

    t_ev, t_det, t_cls, t_saf, t_trace, t_raw = st.tabs(
        ["Evidence", "Detections", "Classification", "Safety", "Agent trace", "Raw / export"])

    with t_ev:
        for e in res["evidence"]:
            st.markdown(f"**{e['source']}** — {e['finding']}")
        if res["claims"]:
            st.markdown("**Claim check** (VQA answer vs. detector)")
            st.dataframe(pd.DataFrame(res["claims"]), hide_index=True)
        if res["uncertainty"]:
            st.markdown("**Uncertainty / review notes**")
            for u in res["uncertainty"]:
                st.warning(u, icon="⚠️")

    with t_det:
        if dets:
            highlight = [p["person_box"] for p in safety["proximity"]] if safety else []
            left, right = st.columns([3, 2])
            left.image(annotate(img, dets, highlight), caption="Candidate detections (yellow = person close to vehicle)")
            counts = (tr["detection"] or {}).get("counts", {})
            right.markdown("**Counts**")
            right.bar_chart(pd.Series(counts, name="count"))
            df = pd.DataFrame([{"label": d["label"], "score": d["score"], "x1": d["box"][0], "y1": d["box"][1],
                                "x2": d["box"][2], "y2": d["box"][3]} for d in dets])
            right.dataframe(df, hide_index=True)
            st.caption("OWL-ViT is open-vocabulary; inspect false positives and raise the threshold in the sidebar if needed.")
        elif "detection" in tr:
            st.info("The detector found nothing above the threshold. Try lowering it in the sidebar.")
        else:
            st.info("Detection was not needed for this question. Ask about objects / counts or pick it under *Advanced*.")

    with t_cls:
        shown = False
        clip = tr.get("clip")
        if isinstance(clip, dict) and clip.get("rows"):
            shown = True
            st.markdown(f"**CLIP zero-shot** ({clip['label_set']}) — relative scores")
            st.bar_chart(pd.DataFrame(clip["rows"]).set_index("label")["score"])
        cnn = tr.get("cnn")
        if isinstance(cnn, dict) and cnn.get("rows"):
            shown = True
            st.markdown("**ResNet-50** (supervised, 1000 ImageNet classes)")
            st.dataframe(pd.DataFrame(cnn["rows"]), hide_index=True)
        if not shown:
            st.info("Classification was not needed. Ask e.g. *“What kind of scene is this?”*, add custom labels, or pick the tools manually.")

    with t_saf:
        if safety:
            icon = {"medium": "🟥", "low": "🟧", "info": "ℹ️"}
            st.markdown(f"**Overall:** {safety['overall']}  ·  work-site probability {safety['worksite_probability']:.2f}"
                        f"  ·  {safety['persons_considered']} person(s), {safety['vehicles_considered']} vehicle(s) considered")
            for f in safety["findings"] or []:
                st.markdown(f"{icon[f['severity']]} **{f['severity'].upper()}** — {f['detail']}")
            if not safety["findings"]:
                st.success("No indicators flagged by the screening rules.")
            c1, c2 = st.columns(2)
            c1.markdown("**Scene checks** (hazard vs. safe prompt)")
            c1.dataframe(pd.DataFrame(safety["scene_checks"]), hide_index=True)
            if safety["ppe_checks"]:
                c2.markdown("**Per-person PPE** (probability PPE is worn)")
                c2.dataframe(pd.DataFrame(safety["ppe_checks"]).drop(columns=["box"]), hide_index=True)
            st.warning(safety["warning"], icon="🛑")
        else:
            st.info("Safety screening was not run. Choose **Safety analysis** mode or ask about hazards / PPE.")

    with t_trace:
        st.markdown(f"**Orchestration:** {res['backend']}")
        st.markdown("**Why these tools?**")
        for tool, why in res["plan_reasons"].items():
            st.markdown(f"- **{TOOL_TITLES.get(tool, tool)}** — {why}")
        st.dataframe(pd.DataFrame(res["trace"]), hide_index=True)

    with t_raw:
        payload = json.dumps(to_jsonable(res), indent=2)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download JSON report", payload, "agent_report.json", "application/json")
        if dets:
            buf = io.BytesIO()
            annotate(img, dets).save(buf, format="PNG")
            d2.download_button("⬇️ Download annotated image", buf.getvalue(), "annotated.png", "image/png")
        with st.expander("Full structured output"):
            st.json(to_jsonable(res))


# ---------------------------------------------------------------- analyze tab
with tab_run:
    left, right = st.columns([5, 6], gap="large")
    with left:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        with st.expander("…or use your camera"):
            camera = st.camera_input("Take a photo")
        image_file = uploaded or camera
        if image_file is not None:
            try:
                preview = prepare_image(image_file)
                st.image(preview, caption=f"{getattr(image_file, 'name', 'camera photo')} · {preview.width}×{preview.height}")
            except Exception as e:
                st.error(f"Could not read this image: {e}")
                image_file = None
    with right:
        st.text_area("Your question", key="question", height=90)
        cols = st.columns(len(EXAMPLES))
        for col, (label, q) in zip(cols, EXAMPLES.items()):
            col.button(label, on_click=_set_question, args=(q,), key=f"ex_{label}", help=q)
        run = st.button("🚀 Analyze image", type="primary", disabled=image_file is None)
        if image_file is None:
            st.caption("Upload an image to enable analysis.")

    if run and image_file is not None:
        image = prepare_image(image_file)
        options = {"mode": "safety" if mode_label.startswith("Safety") else "general",
                   "custom_labels": custom_labels, "threshold": threshold, "verify": verify,
                   "forced_tools": manual or None}
        status = st.status("Agent running…", expanded=True)
        status.write("🧭 Planning. The first run downloads the models it needs, which can take a few minutes.")

        def on_step(node, update, state):
            t = (update.get("trace") or [{}])[0]
            status.write(f"✅ **{node}** — {t.get('detail', '')} ({t.get('seconds', 0)} s)")

        try:
            result = agent.run(image, st.session_state["question"], options, on_step)
        except Exception as e:
            status.update(label="Agent failed", state="error")
            st.exception(e)
            st.stop()
        status.update(label=f"Done in {result['elapsed_seconds']} s", state="complete", expanded=False)
        name = getattr(image_file, "name", "camera_photo")
        save_run(name, st.session_state["question"], result, options)
        st.session_state["last"] = {"result": result, "image": image, "name": name,
                                    "question": st.session_state["question"]}

    if "last" in st.session_state:
        st.divider()
        st.subheader(f"Result · “{st.session_state['last']['question']}”")
        render_result(st.session_state["last"])


# ----------------------------------------------------------------- history tab
with tab_hist:
    runs = load_runs()
    if not runs:
        st.info("No runs yet. Every analysis is logged to `outputs/agent_runs.jsonl`.")
    else:
        df = pd.DataFrame([{"time (UTC)": r.get("timestamp_utc"), "image": r.get("image_name"),
                            "question": r.get("question"), "tools": ", ".join(r.get("plan") or []),
                            "agreement": r.get("agreement"), "seconds": r.get("elapsed_seconds")} for r in runs])
        st.dataframe(df, hide_index=True)
        pick = st.selectbox("Inspect a run", range(len(runs)),
                            format_func=lambda i: f"{runs[i].get('timestamp_utc')} · {runs[i].get('question')}")
        st.markdown(runs[pick].get("summary") or "")
        with st.expander("Raw record"):
            st.json(runs[pick])
        if st.button("🗑️ Clear history"):
            C.RUNS_FILE.unlink(missing_ok=True)
            st.rerun()


# ------------------------------------------------------------------- about tab
with tab_about:
    st.markdown("### How the agent works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      plan -> tools -> reconcile; reconcile -> verify [label="unchecked claims"]; verify -> reconcile;
      reconcile -> compose [label="done"]; }
    """)
    st.markdown("""
| Tool | Model | Used for |
|---|---|---|
| VQA | BLIP (`Salesforce/blip-vqa-base`) | answering the natural-language question |
| Detection | OWL-ViT (`google/owlvit-base-patch32`) | open-vocabulary boxes, counting, claim verification |
| Zero-shot classification | CLIP (`openai/clip-vit-base-patch32`) | scene tags / custom labels |
| CNN baseline | ResNet-50 (ImageNet) | supervised comparison |
| Safety | CLIP paired prompts + OWL-ViT geometry | PPE, proximity, obstruction & fall hazards |

**Verification loop.** The VQA answer is turned into claims (e.g. *“cars and trees”* → `car`, `tree`) and each is
checked against the detector. Unchecked claims trigger one extra, bounded detector pass; disagreements are reported
instead of being silently resolved.

**Limitations.** All models run zero-shot and can be wrong. CLIP probabilities compare two prompts and are not calibrated.
Proximity uses 2-D boxes (no depth). Safety output is a *preliminary screening*, never a compliance decision.
""")
