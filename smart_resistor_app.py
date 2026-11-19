import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Smart Resistor Manager", page_icon="🔎", layout="wide")
st.title("🔎 Smart Resistor Identification & Inventory")
st.caption("Camera-based resistor colour-band reading using Python, OpenCV and NumPy")

if "inventory" not in st.session_state:
    st.session_state.inventory = []

DIGITS = {
    "Black": 0, "Brown": 1, "Red": 2, "Orange": 3, "Yellow": 4,
    "Green": 5, "Blue": 6, "Violet": 7, "Grey": 8, "White": 9
}
MULTIPLIERS = {
    "Black": 1, "Brown": 10, "Red": 100, "Orange": 1_000,
    "Yellow": 10_000, "Green": 100_000, "Blue": 1_000_000,
    "Gold": 0.1, "Silver": 0.01
}
TOLERANCES = {
    "Brown": 1, "Red": 2, "Green": 0.5, "Blue": 0.25,
    "Violet": 0.1, "Grey": 0.05, "Gold": 5, "Silver": 10
}

def classify_hsv(h, s, v):
    # Approximate HSV ranges; camera white balance and lighting affect results.
    if v < 55 and s < 160: return "Black"
    if s < 55 and v > 175: return "White"
    if s < 55 and 65 <= v <= 185: return "Grey"
    if h <= 10 or h >= 170:
        return "Red" if v > 70 else "Brown"
    if 5 < h <= 22:
        return "Orange" if v > 115 and s > 100 else "Brown"
    if 22 < h <= 36: return "Yellow"
    if 36 < h <= 85: return "Green"
    if 85 < h <= 130: return "Blue"
    if 130 < h <= 169: return "Violet"
    return None

def detect_bands(image_rgb):
    """Basic horizontal-resistor detector. Returns approximate colour bands.
    Best results: resistor horizontal, close-up, plain background, diffuse light.
    """
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Find likely resistor body pixels: moderately bright, low-to-medium saturation.
    body = cv2.inRange(hsv, np.array([0, 15, 90]), np.array([179, 180, 255]))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.18 and ch > h * 0.035 and cw / max(ch, 1) > 1.7:
            candidates.append((cw * ch, x, y, cw, ch))
    if not candidates:
        # Fallback: sample the middle horizontal strip of the whole image.
        x, y, cw, ch = int(w*.1), int(h*.35), int(w*.8), int(h*.3)
    else:
        _, x, y, cw, ch = max(candidates)
        x += int(cw * .12); cw = int(cw * .76)
        y += int(ch * .2); ch = int(ch * .6)

    # Average colour in narrow vertical slices, ignoring edges.
    y0, y1 = max(0, y), min(h, y + ch)
    x0, x1 = max(0, x), min(w, x + cw)
    if x1 <= x0 or y1 <= y0:
        return [], None
    strip = hsv[y0:y1, x0:x1]
    n = strip.shape[1]
    labels = []
    for xx in range(n):
        col = strip[:, max(0, xx-2):min(n, xx+3), :].reshape(-1, 3)
        # Prefer saturated pixels, often the coloured band rather than beige body.
        sat = col[:, 1]
        chosen = col[sat > 65]
        if len(chosen) < 3:
            labels.append(None)
        else:
            med = np.median(chosen, axis=0)
            labels.append(classify_hsv(*med))
    # Smooth labels and group consecutive colour runs.
    for i in range(2, len(labels)-2):
        vals = [v for v in labels[i-2:i+3] if v is not None]
        if vals:
            labels[i] = max(set(vals), key=vals.count)
    runs = []
    start = 0
    while start < n:
        label = labels[start]
        end = start + 1
        while end < n and labels[end] == label:
            end += 1
        if label is not None and end-start >= max(3, int(n*.012)):
            runs.append((label, x0+start, x0+end))
        start = end
    # Merge adjacent runs of same colour and retain plausible band widths.
    merged = []
    for item in runs:
        if merged and merged[-1][0] == item[0] and item[1] - merged[-1][2] < 5:
            merged[-1] = (item[0], merged[-1][1], item[2])
        else:
            merged.append(item)
    # Keep up to the 5 most central plausible bands, ordered left-to-right.
    merged = [r for r in merged if (r[2]-r[1]) < n*.22]
    return merged[:5], (x0, y0, x1, y1)

def format_ohms(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:g} MΩ"
    if value >= 1_000:
        return f"{value/1_000:g} kΩ"
    return f"{value:g} Ω"

def calculate(bands):
    if len(bands) < 4:
        raise ValueError("At least four bands are needed for a standard 4-band resistor.")
    b = bands[:4]
    if b[0] not in DIGITS or b[1] not in DIGITS:
        raise ValueError("The first two detected bands must be digit colours.")
    if b[2] not in MULTIPLIERS or b[3] not in TOLERANCES:
        raise ValueError("The third band must be a multiplier and the fourth a tolerance colour.")
    value = (DIGITS[b[0]] * 10 + DIGITS[b[1]]) * MULTIPLIERS[b[2]]
    tol = TOLERANCES[b[3]]
    return value, tol

st.info("For best results, place one resistor horizontally on a plain, light background. Keep all bands visible and use even lighting.")
source = st.radio("Choose image source", ["Take a picture", "Upload a picture"], horizontal=True)
image_file = st.camera_input("Take a clear resistor photo") if source == "Take a picture" else st.file_uploader("Upload resistor photo", type=["jpg", "jpeg", "png"])

if image_file:
    pil = Image.open(image_file).convert("RGB")
    rgb = np.array(pil)
    st.image(pil, caption="Captured resistor image", use_container_width=True)
    with st.spinner("Analysing colour bands..."):
        detected, roi = detect_bands(rgb)

    if roi:
        preview = rgb.copy()
        x0, y0, x1, y1 = roi
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for idx, (name, a, b) in enumerate(detected):
            cv2.rectangle(preview, (a, y0), (b, y1), (255, 0, 0), 2)
            cv2.putText(preview, name, (a, max(20, y0-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1)
        st.image(preview, caption="Approximate detected band regions", use_container_width=True)

    st.subheader("Detected colours — please verify")
    if detected:
        detected_names = [x[0] for x in detected]
        st.write(" → ".join(detected_names))
    else:
        st.warning("No reliable colour bands were detected. Try a closer, brighter, sharper photo.")
        detected_names = []

    st.caption("Automatic colour recognition is an experimental estimate. Confirm the band order and colours before using the result.")
    default_bands = (detected_names + ["Brown", "Black", "Orange", "Gold"])[:4]
    options = list(dict.fromkeys(list(DIGITS) + ["Gold", "Silver"]))
    c1, c2, c3, c4 = st.columns(4)
    with c1: band1 = st.selectbox("Band 1", options, index=options.index(default_bands[0]) if default_bands[0] in options else 1)
    with c2: band2 = st.selectbox("Band 2", options, index=options.index(default_bands[1]) if default_bands[1] in options else 0)
    with c3: band3 = st.selectbox("Multiplier", list(MULTIPLIERS), index=list(MULTIPLIERS).index(default_bands[2]) if default_bands[2] in MULTIPLIERS else 3)
    with c4: band4 = st.selectbox("Tolerance", list(TOLERANCES), index=list(TOLERANCES).index(default_bands[3]) if default_bands[3] in TOLERANCES else 6)

    try:
        value, tol = calculate([band1, band2, band3, band4])
        low, high = value*(1-tol/100), value*(1+tol/100)
        st.success(f"Estimated resistance: {format_ohms(value)}  |  Tolerance: ±{tol}%")
        st.write(f"Expected range: **{format_ohms(low)} to {format_ohms(high)}**")

        st.markdown("### 📋 Resistor specifications")
        st.write("The colour bands provide the nominal resistance and tolerance. Other specifications depend on the resistor's physical type or datasheet.")
        spec1, spec2 = st.columns(2)
        with spec1:
            rated_power = st.selectbox("Rated power (select from resistor marking/datasheet)", ["Not specified", "1/8 W", "1/4 W", "1/2 W", "1 W", "2 W", "5 W"])
            tcr = st.selectbox("Temperature coefficient (TCR)", ["Not specified", "50 ppm/°C", "100 ppm/°C", "200 ppm/°C", "Other / datasheet"])
        with spec2:
            max_voltage = st.text_input("Maximum working voltage (V)", placeholder="Enter datasheet value")
            resistor_type = st.selectbox("Resistor type", ["Not specified", "Carbon film", "Metal film", "Metal oxide", "Wirewound", "Thick film / SMD"])
        st.caption("Do not assume power rating, maximum voltage or TCR from the colour code alone. Confirm these from the component marking or manufacturer datasheet.")

        spec_record = {
            "value": value, "tolerance": tol, "quantity": 1,
            "power_rating": rated_power, "tcr": tcr,
            "max_voltage": max_voltage or "Not specified",
            "resistor_type": resistor_type
        }
        if st.button("➕ Add verified resistor to inventory"):
            match = next((x for x in st.session_state.inventory
                          if x["value"] == value and x["tolerance"] == tol
                          and x.get("power_rating", "Not specified") == rated_power
                          and x.get("tcr", "Not specified") == tcr
                          and x.get("max_voltage", "Not specified") == (max_voltage or "Not specified")
                          and x.get("resistor_type", "Not specified") == resistor_type), None)
            if match:
                match["quantity"] += 1
            else:
                st.session_state.inventory.append(spec_record)
            st.success("Added resistor and its specifications to inventory.")
    except ValueError as e:
        st.error(str(e))

st.divider()
st.subheader("📦 Digital inventory")
if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)
    df["resistance"] = df["value"].apply(format_ohms)
    for col in ["power_rating", "tcr", "max_voltage", "resistor_type"]:
        if col not in df.columns:
            df[col] = "Not specified"
    st.dataframe(
        df[["resistance", "tolerance", "power_rating", "max_voltage", "tcr", "resistor_type", "quantity"]],
        use_container_width=True, hide_index=True
    )
    if st.button("Download inventory as CSV"):
        st.download_button("Save CSV", df.to_csv(index=False).encode("utf-8"), "resistor_inventory.csv", "text/csv")
else:
    st.write("No resistor records saved yet. Scan and add a resistor to begin.")
st.caption("Prototype limitation: the colour detector is a simple image-processing baseline, not a calibrated industrial vision system. It currently targets horizontal 4-band resistors; 5-band support, orientation detection, lighting calibration and persistent database storage are future improvements.")
