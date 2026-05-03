from datetime import datetime
from io import BytesIO
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.stats import kurtosis, skew

BASE_DIR = Path(__file__).resolve().parent


def artifact_path(filename):
    path = BASE_DIR / "artifacts" / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"{filename} not found in artifacts/")

EEG_COLS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4"]
REGIONS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central": ["C3", "C4", "Cz"],
    "Temporal": ["T3", "T4", "T5"],
    "Parietal": ["P3", "P4", "Pz"],
}
RISK_BANDS = [
    (0.25, "No Risk", "#16a34a"),
    (0.50, "Moderate Risk", "#f59e0b"),
    (0.75, "High Risk", "#ef4444"),
    (1.01, "Extreme Risk", "#b91c1c"),
]
RECS = {
    "No Risk": [
        "Maintain regular sleep, exercise, and hydration habits.",
        "Keep the brain active through reading, puzzles, and learning.",
        "Continue a balanced diet rich in antioxidants and omega-3 sources.",
        "Schedule routine preventive neurological checkups as advised.",
    ],
    "Moderate Risk": [
        "Consult a neurologist for a detailed clinical assessment.",
        "Track memory lapses, mood shifts, sleep quality, and daily functioning.",
        "Reduce stress with mindfulness, breathing exercises, or yoga.",
        "Discuss follow-up EEG or cognitive screening with a specialist.",
    ],
    "High Risk": [
        "Arrange an urgent neurological consultation.",
        "Ask about MRI, PET, or detailed cognitive testing as follow-up.",
        "Create a structured daily routine with family or caregiver support.",
        "Review early intervention options with a qualified clinician.",
    ],
    "Extreme Risk": [
        "Seek immediate medical consultation and avoid delaying care.",
        "Request advanced neurological diagnostics and cognitive testing.",
        "Set up continuous monitoring and a supervised care plan.",
        "Discuss treatment planning and clinical trial eligibility with a neurologist.",
    ],
}

app = Flask(__name__)
model = joblib.load(artifact_path("model.joblib"))
scaler = joblib.load(artifact_path("scaler.joblib"))


def risk_for(prob):
    return next((label, color) for limit, label, color in RISK_BANDS if prob < limit)


def engineer(df):
    df = df.copy()
    d = df[EEG_COLS].astype(float).values
    df["mean"] = df[EEG_COLS].mean(1)
    df["std"] = df[EEG_COLS].std(1)
    df["var"] = df[EEG_COLS].var(1)
    df["skew"] = df[EEG_COLS].apply(lambda x: skew(x), 1)
    df["kurt"] = df[EEG_COLS].apply(lambda x: kurtosis(x), 1)
    df["max"] = df[EEG_COLS].max(1)
    df["min"] = df[EEG_COLS].min(1)
    df["range"] = df["max"] - df["min"]
    df["rms"] = np.sqrt((d**2).mean(1))
    df["iqr"] = np.percentile(d, 75, 1) - np.percentile(d, 25, 1)
    mean = d.mean(1, keepdims=True)
    df["mad"] = np.mean(np.abs(d - mean), 1)
    df["coeff_var"] = df["std"] / (df["mean"] + 1e-6)
    energy = np.sum(d**2, 1)
    df["energy"] = energy
    df["log_energy"] = np.log(energy + 1e-6)
    for prefix, left, right in [("Fp", "Fp1", "Fp2"), ("C", "C3", "C4"), ("P", "P3", "P4")]:
        df[f"{prefix}_diff"] = df[left] - df[right]
        df[f"{prefix}_ratio"] = df[left] / (df[right] + 1e-6)
    for region, cols in REGIONS.items():
        key = region.lower()
        df[f"{key}_mean"] = df[cols].mean(1)
        df[f"{key}_std"] = df[cols].std(1)
    return df


def predict_rows(df, source="Manual input"):
    features = engineer(df)
    scaled = scaler.transform(features[model.feature_names_in_])
    probs = model.predict_proba(scaled)[:, 1]
    results = []
    for i, prob in enumerate(probs, 1):
        prob = float(prob)
        level, color = risk_for(prob)
        results.append({
            "source": source,
            "row": i,
            "probability": round(prob, 4),
            "percent": round(prob * 100, 2),
            "risk": level,
            "color": color,
            "recommendations": RECS[level],
        })
    return results


@app.get("/")
def index():
    return render_template("index.html", channels=EEG_COLS, regions=REGIONS)


@app.post("/predict/manual")
def predict_manual():
    payload = request.get_json(force=True)
    try:
        df = pd.DataFrame([{ch: float(payload[ch]) for ch in EEG_COLS}])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Enter valid numeric values for all 16 EEG channels. {exc}"}), 400
    return jsonify({"results": predict_rows(df)})


@app.post("/predict/csv")
def predict_csv():
    if not request.files:
        return jsonify({"error": "Upload at least one CSV file."}), 400
    all_results = []
    for file in request.files.getlist("files"):
        try:
            df = pd.read_csv(file)
            missing = [col for col in EEG_COLS if col not in df.columns]
            if missing:
                return jsonify({"error": f"{file.filename}: missing columns {', '.join(missing)}"}), 400
            rows = df[EEG_COLS].dropna()
            for item in predict_rows(rows, file.filename):
                all_results.append(item)
        except Exception as exc:
            return jsonify({"error": f"{file.filename}: {exc}"}), 400
    return jsonify({"results": all_results})


@app.post("/report")
def report():
    data = request.get_json(force=True)
    prob = float(data["probability"])
    level, color = risk_for(prob)
    name = data.get("name") or "Patient"
    age = data.get("age") or "Not provided"
    gender = data.get("gender") or "Not provided"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, textColor=colors.HexColor("#0f4c81"), alignment=TA_CENTER, spaceAfter=12)
    sec = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#0f4c81"), spaceBefore=16, spaceAfter=8)
    body = ParagraphStyle("body", fontSize=10, textColor=colors.HexColor("#1f2937"), leading=15, spaceAfter=5)
    table_head = ParagraphStyle("table_head", fontName="Helvetica-Bold", fontSize=9.5, textColor=colors.white, leading=12)
    table_label = ParagraphStyle("table_label", fontName="Helvetica-Bold", fontSize=9.5, textColor=colors.HexColor("#0f172a"), leading=12)
    table_value = ParagraphStyle("table_value", fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#111827"), leading=12)

    story = [
        Paragraph("EEG Based Alzheimer's Detection", title),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f4c81")),
        Spacer(1, 8),
        Paragraph("Patient Information", sec),
    ]
    patient_rows = [
        ["Field", "Details"],
        ["Full Name", name],
        ["Age", str(age)],
        ["Gender", gender],
        ["Report Date", datetime.now().strftime("%d %B %Y, %I:%M %p")],
        ["Analysis Method", "EEG feature engineering with XGBoost classifier"],
    ]
    patient = Table([
        [Paragraph("Field", table_head), Paragraph("Details", table_head)],
        *[[Paragraph(str(a), table_label), Paragraph(str(b), table_value)] for a, b in patient_rows[1:]],
    ], colWidths=[1.9 * inch, 4.2 * inch], rowHeights=[0.36 * inch] + [0.42 * inch] * 5)
    patient.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fbff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbeafe")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    result = Table([["Confidence Probability", "Risk Level"], [f"{prob:.4f}", level]], colWidths=[3.05 * inch, 3.05 * inch], rowHeights=[0.38 * inch, 0.58 * inch])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(color)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbeafe")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    rec_rows = [[Paragraph(str(i), body), Paragraph(text, body)] for i, text in enumerate(RECS[level], 1)]
    rec_table = Table(rec_rows, colWidths=[0.35 * inch, 5.75 * inch])
    rec_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f4c81")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [patient, Spacer(1, 18), Paragraph("Prediction Result", sec), result, Paragraph("Recommendations", sec), rec_table]
    story += [
        Spacer(1, 18),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dbeafe")),
        Spacer(1, 8),
        Paragraph("Disclaimer: This AI screening report is for educational and informational use only. It is not a medical diagnosis. Consult a qualified neurologist for clinical assessment.", body),
    ]
    doc.build(story)
    buf.seek(0)
    filename = f"EEG_AD_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
