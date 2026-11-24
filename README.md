# AETHER: AI Emergency Traffic & Highway Emergency Response

A Digital Literacy project at JECRC University. AETHER is a **prototype and simulation**. It does **not** connect to or control real traffic signals.

## What it does

Upload a traffic video and AETHER will:

1. Detect and track vehicles (ByteTrack tracking on YOLO detections)
2. Recognise three emergency vehicle types: **Ambulance, Fire-Truck, Police-car**
3. Show an annotated video, per-class counts, charts, and a downloadable CSV report
4. Raise a **simulated** priority alert when an emergency vehicle is detected

Two modes, chosen by camera type:

| Mode | Models | Purpose |
|---|---|---|
| Aerial / Drone | ARGUS-YOLO | vehicle detection and tracking from above |
| Road-level / CCTV / Dashcam | custom emergency model + YOLO11s | emergency recognition plus ordinary-vehicle tracking |

## Emergency model results

Fine-tuned from `yolo11s.pt` for 40 epochs (about 24 minutes on a T4 GPU).

**Held-out test set (223 images, never used for training or model selection):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.82 | 0.83 | 0.845 | 0.625 |
| Ambulance | 0.88 | 0.87 | 0.905 | 0.683 |
| Fire-Truck | 0.77 | 0.82 | 0.808 | 0.508 |
| Police-car | 0.81 | 0.78 | 0.822 | 0.683 |

Validation mAP50 was 0.925; test numbers are the honest ones to quote.

## Limitations (please read)

- Trained on mostly **close-up photos** of emergency vehicles. Performance on distant, blurry CCTV or dashcam footage is expected to be lower and has not been measured.
- It **will not recognise vehicles seen from straight above**. Aerial mode only detects generic vehicles and people.
- Police-car has only 37 test boxes, so its metrics are rough.
- The emergency model has never seen ordinary cars. Ordinary traffic is handled by a separate general model, and false alarms on ordinary vehicles have not been systematically measured.
- Tracking IDs may switch between frames, so "unique tracked objects" can overcount.
- The synthetic congestion dataset (10,000 rows) is **simulated**, not real sensor data.
- Live-camera input, direction analysis and the full priority-control simulation are not built yet.

## Data and credits

- Emergency vehicle dataset: **"Emergency_Vehicle Dataset" by Shikha**, Roboflow Universe, licence **CC BY 4.0**: https://universe.roboflow.com/shikha-eavtx/emergency_vehicle (version 2, three classes). The dataset is not included in this repository.
- Aerial detection: ARGUS-YOLO (`RoblabWhGe/ARGUS-YOLO` on Hugging Face), downloaded at runtime and not redistributed here. Check its model card for its licence.
- Detection and tracking: Ultralytics YOLO (AGPL-3.0) with ByteTrack.

## Repository contents

```
app_v3.py            Streamlit web app
best.pt              trained emergency-vehicle weights
notebooks/           Colab notebooks (data generation, CV experiments, web app, emergency training)
results/             training curves, confusion matrices, screenshots
README.md
```

## How to run (Google Colab)

1. Open a Colab notebook (a T4 GPU runtime is recommended; CPU works but is slow).
2. Install the packages: `pip install streamlit ultralytics plotly huggingface_hub`.
3. Put `app_v3.py` and `best.pt` in your Google Drive and update the `WEIGHTS` path at the top of `app_v3.py`.
4. Start it with `streamlit run app_v3.py`, then expose port 8501 with a Cloudflare quick tunnel (`cloudflared tunnel --url http://127.0.0.1:8501`) and open the printed link.

The tunnel link changes on every restart and only works while the Colab session is running.

## Safety note

AETHER is an academic simulation. A real deployment would need traffic-authority approval, validated hardware, and extensive safety testing.
