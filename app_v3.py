import os, shutil, subprocess, tempfile, time
import cv2, pandas as pd, plotly.express as px, streamlit as st, torch
from ultralytics import YOLO

st.set_page_config(page_title="AETHER", page_icon="🚦", layout="wide")
WEIGHTS = "/content/drive/MyDrive/AETHER/runs/emergency_v1/weights/best.pt"
EMERGENCY = {"Ambulance", "Fire-Truck", "Police-car"}
DEVICE = 0 if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("RoblabWhGe/ARGUS-YOLO", "argus_yolo11l_1280.pt")
    return YOLO(p), YOLO(WEIGHTS), YOLO("yolo11s.pt")


def to_h264(src, dst):
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def analyse(path, model, conf, imgsz, skip, bar, txt, extra=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(7)) or 1
    raw = os.path.join(tempfile.gettempdir(), "aether_raw.mp4")
    out = os.path.join(tempfile.gettempdir(), "aether_out.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model.predictor = None
    if extra is not None:
        extra.predictor = None
    rows, last, n, t0 = [], None, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if last is None or (n - 1) % skip == 0:
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf,
                            imgsz=imgsz, device=DEVICE, verbose=False)[0]
            base = frame
            if extra is not None:
                g = extra.track(frame, persist=True, tracker="bytetrack.yaml",
                                conf=0.3, classes=[2, 3, 5, 7], imgsz=imgsz,
                                device=DEVICE, verbose=False)[0]
                base = g.plot()
                gids = g.boxes.id.int().tolist() if g.boxes.id is not None else [-1] * len(g.boxes)
                for t, c, cf in zip(gids, g.boxes.cls.int().tolist(), g.boxes.conf.tolist()):
                    rows.append({"frame": n, "track_id": 100000 + t,
                                 "class": extra.names[c], "confidence": round(cf, 3)})
            last = r.plot(img=base)
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
            for t, c, cf in zip(ids, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                rows.append({"frame": n, "track_id": t, "class": model.names[c],
                             "confidence": round(cf, 3)})
        vw.write(last)
        if n % 5 == 0:
            bar.progress(min(n / total, 1.0))
            txt.text(f"Frame {n}/{total} | {n / (time.time() - t0):.1f} fps")
    cap.release()
    vw.release()
    secs = time.time() - t0
    ok = to_h264(raw, out)
    data = open(out if ok else raw, "rb").read()
    cols = ["frame", "track_id", "class", "confidence"]
    return {"df": pd.DataFrame(rows, columns=cols), "video": data, "frames": n, "secs": secs}


with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Camera type", ["Road-level / CCTV / Dashcam (emergency recognition)",
                                        "Aerial / Drone (vehicle detection)"])
    aerial = mode.startswith("Aerial")
    conf = st.slider("Confidence", 0.05, 0.90, 0.20 if aerial else 0.50, 0.05)
    skip = st.slider("Analyse every Nth frame", 1, 5, 1 if DEVICE != "cpu" else 3)
    imgsz = st.select_slider("Resolution", [640, 960, 1280], value=640)
    st.caption("Device: " + ("GPU" if DEVICE != "cpu" else "CPU (slow)"))
    st.caption("Simulation only. Does not control real traffic signals.")

st.title("🚦 AETHER")
st.caption("AI traffic intelligence and emergency response (simulation)")
argus, emg, gen = load_models()
model = argus if aerial else emg

up = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
if up is not None:
    c1, c2 = st.columns(2)
    c1.video(up)
    go = c2.button("Analyse traffic", type="primary", use_container_width=True)
    bar, txt = c2.empty(), c2.empty()
    if go:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
            tf.write(up.getbuffer())
        pb = bar.progress(0.0)
        st.session_state["res"] = analyse(tf.name, model, conf, imgsz, skip, pb, txt,
                                          None if aerial else gen)
        txt.success("Done")

res = st.session_state.get("res")
if res:
    df = res["df"]
    a, b, c, d = st.columns(4)
    a.metric("Frames", res["frames"])
    b.metric("Unique tracked objects", df["track_id"].nunique())
    c.metric("Detections", len(df))
    d.metric("Speed", f"{res['frames'] / max(res['secs'], 1e-6):.1f} fps")
    ed = df[df["class"].isin(EMERGENCY)]
    if not ed.empty:
        kinds = ", ".join(sorted(ed["class"].unique()))
        st.error(f"🚨 Emergency vehicle detected: {kinds} in {ed['frame'].nunique()} frames. "
                 "A real deployment would send a priority request. SIMULATION ONLY.")
    st.video(res["video"])
    if df.empty:
        st.warning("Nothing detected. Try a lower confidence.")
    else:
        by = df.groupby("class")["track_id"].nunique().reset_index(name="unique objects")
        st.plotly_chart(px.bar(by, x="class", y="unique objects", template="plotly_dark"),
                        use_container_width=True)
        per = df.groupby("frame").size().reset_index(name="objects")
        st.plotly_chart(px.line(per, x="frame", y="objects", template="plotly_dark"),
                        use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "AETHER_report.csv", "text/csv")
    st.download_button("Download video", res["video"], "AETHER_analysis.mp4", "video/mp4")
