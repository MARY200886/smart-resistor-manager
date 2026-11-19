import io
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Smart Resistor Manager", page_icon="🔎", layout="wide")

DB_PATH = Path("resistor_inventory.db")

DIGITS = {
    "Black": 0, "Brown": 1, "Red": 2, "Orange": 3, "Yellow": 4,
    "Green": 5, "Blue": 6, "Violet": 7, "Grey": 8, "White": 9,
}
MULTIPLIERS = {
    "Pink": 0.001, "Silver": 0.01, "Gold": 0.1, "Black": 1,
    "Brown": 10, "Red": 100, "Orange": 1_000, "Yellow": 10_000,
    "Green": 100_000, "Blue": 1_000_000, "Violet": 10_000_000,
}
TOLERANCES = {
    "Brown": 1, "Red": 2, "Green": 0.5, "Blue": 0.25,
    "Violet": 0.1, "Grey": 0.05, "Gold": 5, "Silver": 10,
}
TEMP_COEFFICIENTS = {
    "Brown": 100, "Red": 50, "Orange": 15, "Yellow": 25,
    "Blue": 10, "Violet": 5,
}
COLOUR_NAMES = list(DIGITS) + ["Gold", "Silver", "Pink"]
COLOUR_HEX = {
    "Black": "#222222", "Brown": "#8B4513", "Red": "#E53935",
    "Orange": "#F28C28", "Yellow": "#FDD835", "Green": "#43A047",
    "Blue": "#1E88E5", "Violet": "#8E44AD", "Grey": "#9E9E9E",
    "White": "#F5F5F5", "Gold": "#C9A227", "Silver": "#BFC5CC",
    "Pink": "#E8A0B8",
}

def db_connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            resistance_ohms REAL NOT NULL,
            tolerance REAL NOT NULL,
            bands TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            power_rating TEXT,
            resistor_type TEXT,
            max_voltage TEXT,
            tcr TEXT,
            notes TEXT
        )
    """)
    con.commit()
    return con

def format_ohms(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g} MΩ"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g} kΩ"
    return f"{value:g} Ω"

def calculate_resistance(bands, band_count):
    if band_count == 4:
        digits = 2
    elif band_count in (5, 6):
        digits = 3
    else:
        raise ValueError("Choose a 4, 5 or 6-band resistor.")
    if len(bands) < band_count:
        raise ValueError("Not enough colour bands selected.")
    digit_value = 0
    for colour in bands[:digits]:
        if colour not in DIGITS:
            raise ValueError("The significant-digit bands must be digit colours.")
        digit_value = digit_value * 10 + DIGITS[colour]
    multiplier_colour = bands[digits]
    tolerance_colour = bands[digits + 1]
    if multiplier_colour not in MULTIPLIERS:
        raise ValueError("Select a valid multiplier colour.")
    if tolerance_colour not in TOLERANCES:
        raise ValueError("Select a valid tolerance colour.")
    value = digit_value * MULTIPLIERS[multiplier_colour]
    tolerance = TOLERANCES[tolerance_colour]
    tcr = None
    if band_count == 6:
        tcr_colour = bands[5]
        tcr = TEMP_COEFFICIENTS.get(tcr_colour)
    return value, tolerance, tcr

def classify_hsv(h, s, v):
    """Approximate colour classifier. Values are OpenCV HSV (H: 0..179)."""
    h, s, v = float(h), float(s), float(v)
    if v < 48:
        return "Black"
    if s < 38:
        if v > 205:
            return "White"
        if v < 125:
            return "Grey"
        return None
    if h <= 7 or h >= 174:
        if v < 145:
            return "Brown"
        return "Red"
    if 8 <= h <= 18:
        return "Brown" if v < 175 else "Orange"
    if 19 <= h <= 34:
        return "Yellow" if v > 145 else "Brown"
    if 35 <= h <= 82:
        return "Green"
    if 83 <= h <= 130:
        return "Blue"
    if 131 <= h <= 169:
        return "Violet"
    return None

def _long_axis_crop(rgb, orientation):
    """Find a likely elongated resistor body; return crop and full-image box."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Beige/tan axial resistor bodies are usually bright and relatively unsaturated.
    mask = cv2.inRange(hsv, np.array([5, 8, 75]), np.array([38, 185, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if area > (h * w) * 0.00015 and aspect >= 1.35 and min(cw, ch) >= 5:
            candidates.append((area * min(aspect, 8), x, y, cw, ch))
    if candidates:
        _, x, y, cw, ch = max(candidates)
        # Pad the body slightly, but not the long wire leads.
        pad_x, pad_y = int(cw * .12), int(ch * .12)
        x0, y0 = max(0, x-pad_x), max(0, y-pad_y)
        x1, y1 = min(w, x+cw+pad_x), min(h, y+ch+pad_y)
        crop = rgb[y0:y1, x0:x1]
        if crop.size:
            return crop, (x0, y0, x1, y1), True
    # Fallback to a central crop. User can adjust orientation and crop in UI.
    side = "vertical" if orientation == "Vertical" else "horizontal"
    if orientation == "Auto":
        side = "vertical" if h > w else "horizontal"
    if side == "vertical":
        x0, x1 = int(w*.30), int(w*.70)
        y0, y1 = int(h*.20), int(h*.80)
    else:
        x0, x1 = int(w*.20), int(w*.80)
        y0, y1 = int(h*.30), int(h*.70)
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1), False

def detect_bands(rgb, orientation="Auto", band_count=4):
    """
    Experimental axial resistor detector.
    Locates a likely beige body, rotates it so its long axis is horizontal,
    samples the centre of the body, then groups colour runs.
    """
    crop, box, auto_found = _long_axis_crop(rgb, orientation)
    if crop.size == 0:
        return [], None, 0.0, "Could not isolate a useful image region."
    ch, cw = crop.shape[:2]
    if orientation == "Vertical" or (orientation == "Auto" and ch > cw):
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == "Horizontal" and ch > cw:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    # Upscale small resistor images before sampling.
    scale = max(1.0, 500 / max(crop.shape[:2]))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    # Sample the central 35% of body height; avoid body edges and background.
    y0, y1 = int(h*.32), max(int(h*.68), int(h*.32)+1)
    strip = hsv[y0:y1, :]
    labels = []
    window = max(3, int(w*.012))
    for x in range(w):
        xa, xb = max(0, x-window), min(w, x+window+1)
        pixels = strip[:, xa:xb].reshape(-1, 3)
        # Median reduces glare and isolated pixel noise; retain low-saturation
        # samples too because black, brown and metallic bands may be muted.
        med = np.median(pixels, axis=0)
        labels.append(classify_hsv(*med))
    # Smooth short gaps/noise.
    smoothed = labels[:]
    for i in range(2, len(labels)-2):
        neighbours = [v for v in labels[i-2:i+3] if v is not None]
        if neighbours:
            smoothed[i] = max(set(neighbours), key=neighbours.count)
    runs = []
    i = 0
    min_width = max(3, int(w*.012))
    while i < w:
        colour = smoothed[i]
        j = i + 1
        while j < w and smoothed[j] == colour:
            j += 1
        if colour and j-i >= min_width:
            runs.append((colour, i, j))
        i = j
    # Remove very broad body/background classifications and merge tiny same-colour gaps.
    runs = [r for r in runs if (r[2]-r[1]) < w*.20]
    merged = []
    for run in runs:
        if merged and merged[-1][0] == run[0] and run[1]-merged[-1][2] <= max(4, int(w*.015)):
            merged[-1] = (run[0], merged[-1][1], run[2])
        else:
            merged.append(run)
    # Keep a plausible sequence of 4/5/6 bands, selecting central narrow runs.
    if len(merged) > band_count:
        merged = sorted(merged, key=lambda r: (r[2]-r[1], abs((r[1]+r[2])/2-w/2)))[:band_count]
        merged.sort(key=lambda r: r[1])
    confidence = min(0.95, len(merged) / float(band_count)) if merged else 0.0
    message = "Likely resistor body found." if auto_found else "Using a central crop; automatic body location was not confident."
    return merged, box, confidence, message

def init_state():
    if "db_ready" not in st.session_state:
        st.session_state.db = db_connect()
        st.session_state.db_ready = True

def inventory_frame():
    rows = st.session_state.db.execute(
        """SELECT id, created_at, resistance_ohms, tolerance, bands, quantity,
                  power_rating, resistor_type, max_voltage, tcr, notes
           FROM inventory ORDER BY id DESC"""
    ).fetchall()
    cols = ["ID", "Added", "Resistance (Ω)", "Tolerance (%)", "Bands", "Quantity",
            "Power rating", "Type", "Max voltage", "TCR", "Notes"]
    return pd.DataFrame(rows, columns=cols)

init_state()

st.title("🔎 Smart Resistor Identification & Inventory")
st.caption("Camera scanner • 4/5/6-band calculator • specifications • inventory • CSV export")
st.info("For best results, photograph one resistor close-up on a plain, matte background. Keep the body and every band in focus. Automatic colour recognition is experimental: verify each band before saving.")

tab_scan, tab_inventory, tab_reference = st.tabs(["📷 Scan & calculate", "📦 Inventory", "📚 Colour-code guide"])

with tab_scan:
    source = st.radio("Image source", ["Take a picture", "Upload a picture"], horizontal=True)
    uploaded = st.camera_input("Take a close-up resistor photo") if source == "Take a picture" else st.file_uploader("Upload resistor photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            pil = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            rgb = np.array(pil)
            st.image(pil, caption="Input image", use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                orientation = st.selectbox("Resistor orientation", ["Auto", "Horizontal", "Vertical"])
            with c2:
                band_count = st.selectbox("Number of colour bands", [4, 5, 6], index=0)
            with st.spinner("Analysing image..."):
                detected, roi, confidence, detection_message = detect_bands(rgb, orientation, band_count)
            st.caption(detection_message)
            if roi:
                preview = rgb.copy()
                x0, y0, x1, y1 = roi
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 220, 0), 3)
                st.image(preview, caption="Approximate detected resistor region", use_container_width=True)
            if detected:
                st.write("Detected colour sequence (left-to-right after orientation correction):")
                st.write(" → ".join(x[0] for x in detected))
                st.progress(min(1.0, confidence), text=f"Band-count coverage indicator: {confidence:.0%} (not a calibrated probability)")
            else:
                st.warning("No reliable bands were detected. Use a closer crop, adjust orientation, or enter the bands manually.")
            st.caption("Please verify the detected order and colours. Lighting, camera white balance, resistor coating and image scale can change the estimate.")

            # Manual confirmation is always available and is the authoritative input to calculation.
            default_names = [r[0] for r in detected]
            while len(default_names) < band_count:
                default_names.append(["Brown", "Black", "Red", "Gold", "Brown", "Brown"][len(default_names)])
            st.subheader("Confirm or correct the bands")
            band_options = COLOUR_NAMES
            band_cols = st.columns(band_count)
            selected = []
            for i in range(band_count):
                if band_count == 4:
                    label = ["Digit 1", "Digit 2", "Multiplier", "Tolerance"][i]
                elif band_count == 5:
                    label = ["Digit 1", "Digit 2", "Digit 3", "Multiplier", "Tolerance"][i]
                else:
                    label = ["Digit 1", "Digit 2", "Digit 3", "Multiplier", "Tolerance", "TCR"][i]
                opts = band_options
                default = default_names[i] if default_names[i] in opts else ("Gold" if i == band_count-1 else "Brown")
                with band_cols[i]:
                    selected.append(st.selectbox(label, opts, index=opts.index(default), key=f"band_{i}_{band_count}"))
            try:
                value, tolerance, tcr_ppm = calculate_resistance(selected, band_count)
                low = value * (1-tolerance/100)
                high = value * (1+tolerance/100)
                st.success(f"Calculated resistance: {format_ohms(value)}  |  Tolerance: ±{tolerance:g}%")
                st.write(f"Permitted nominal range: **{format_ohms(low)} to {format_ohms(high)}**")
                if band_count == 6:
                    st.write(f"Temperature coefficient: {tcr_ppm} ppm/°C" if tcr_ppm else "TCR colour not in the common reference table; check datasheet.")
                st.subheader("Resistor specifications")
                st.write("Colour bands determine nominal resistance and (where present) tolerance. They do not identify power rating or maximum working voltage.")
                a, b = st.columns(2)
                with a:
                    power = st.selectbox("Rated power", ["Not specified", "1/8 W", "1/4 W", "1/2 W", "1 W", "2 W", "3 W", "5 W", "10 W"])
                    resistor_type = st.selectbox("Resistor type", ["Not specified", "Carbon film", "Metal film", "Metal oxide", "Wirewound", "Thick film / SMD", "Thin film / SMD"])
                with b:
                    max_voltage = st.text_input("Maximum working voltage (from datasheet)", placeholder="e.g. 250 V")
                    tcr_choice = st.selectbox("TCR (from datasheet)", ["Not specified", "5 ppm/°C", "10 ppm/°C", "15 ppm/°C", "25 ppm/°C", "50 ppm/°C", "100 ppm/°C", "200 ppm/°C", "Other / datasheet"])
                notes = st.text_area("Notes (optional)", placeholder="Manufacturer, part number, package, source, etc.")
                if st.button("➕ Save verified resistor to inventory", type="primary"):
                    bands_text = " - ".join(selected)
                    existing = st.session_state.db.execute(
                        """SELECT id, quantity FROM inventory
                           WHERE resistance_ohms=? AND tolerance=? AND bands=?
                           AND power_rating=? AND resistor_type=? AND max_voltage=? AND tcr=? AND notes=?""",
                        (float(value), float(tolerance), bands_text, power, resistor_type,
                         max_voltage or "Not specified", tcr_choice, notes)
                    ).fetchone()
                    if existing:
                        st.session_state.db.execute("UPDATE inventory SET quantity=? WHERE id=?", (existing[1]+1, existing[0]))
                    else:
                        st.session_state.db.execute(
                            """INSERT INTO inventory
                            (created_at, resistance_ohms, tolerance, bands, quantity, power_rating,
                             resistor_type, max_voltage, tcr, notes)
                             VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
                            (datetime.now().isoformat(timespec="seconds"), float(value), float(tolerance),
                             bands_text, power, resistor_type, max_voltage or "Not specified", tcr_choice, notes)
                        )
                    st.session_state.db.commit()
                    st.success("Saved to inventory.")
            except ValueError as exc:
                st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")

with tab_inventory:
    st.subheader("Resistor inventory")
    df = inventory_frame()
    if not df.empty:
        display = df.copy()
        display["Resistance"] = display["Resistance (Ω)"].apply(format_ohms)
        st.dataframe(display[["ID", "Added", "Resistance", "Tolerance (%)", "Bands", "Quantity",
                              "Power rating", "Type", "Max voltage", "TCR", "Notes"]],
                     use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download inventory CSV", csv_bytes, "resistor_inventory.csv", "text/csv")
        st.subheader("Update or remove a record")
        ids = df["ID"].tolist()
        chosen_id = st.selectbox("Select inventory record ID", ids)
        record = df[df["ID"] == chosen_id].iloc[0]
        new_qty = st.number_input("Set quantity", min_value=0, max_value=1_000_000, value=int(record["Quantity"]), step=1)
        u1, u2 = st.columns(2)
        with u1:
            if st.button("Update quantity"):
                st.session_state.db.execute("UPDATE inventory SET quantity=? WHERE id=?", (int(new_qty), int(chosen_id)))
                st.session_state.db.commit()
                st.success("Quantity updated. Refresh the page to see the latest table.")
        with u2:
            if st.button("Delete selected record"):
                st.session_state.db.execute("DELETE FROM inventory WHERE id=?", (int(chosen_id),))
                st.session_state.db.commit()
                st.success("Record deleted. Refresh the page to see the latest table.")
    else:
        st.info("No resistor records saved yet. Scan and save a verified resistor to begin.")
    st.subheader("Import inventory CSV")
    csv_file = st.file_uploader("Import a previously exported inventory CSV", type=["csv"], key="inventory_csv")
    if csv_file and st.button("Import CSV"):
        try:
            imported = pd.read_csv(csv_file)
            required = {"Resistance (Ω)", "Tolerance (%)", "Bands", "Quantity"}
            if not required.issubset(imported.columns):
                st.error("CSV is missing required columns.")
            else:
                for _, row in imported.iterrows():
                    st.session_state.db.execute(
                        """INSERT INTO inventory
                        (created_at, resistance_ohms, tolerance, bands, quantity, power_rating,
                         resistor_type, max_voltage, tcr, notes)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (datetime.now().isoformat(timespec="seconds"), float(row["Resistance (Ω)"]),
                         float(row["Tolerance (%)"]), str(row["Bands"]), int(row["Quantity"]),
                         str(row.get("Power rating", "Not specified")), str(row.get("Type", "Not specified")),
                         str(row.get("Max voltage", "Not specified")), str(row.get("TCR", "Not specified")),
                         str(row.get("Notes", "")))
                    )
                st.session_state.db.commit()
                st.success(f"Imported {len(imported)} inventory row(s).")
        except Exception as exc:
            st.error(f"Import failed: {exc}")

with tab_reference:
    st.subheader("Resistor colour-code reference")
    reference_rows = []
    for colour, digit in DIGITS.items():
        reference_rows.append({"Colour": colour, "Digit": digit,
                               "Multiplier": f"×{MULTIPLIERS.get(colour, '—')}",
                               "Tolerance": f"±{TOLERANCES.get(colour, '—')}%" if colour in TOLERANCES else "—",
                               "TCR (common 6th band)": f"{TEMP_COEFFICIENTS.get(colour, '—')} ppm/°C"})
    for colour in ["Gold", "Silver", "Pink"]:
        reference_rows.append({"Colour": colour, "Digit": "—",}
                               "Multiplier": f
