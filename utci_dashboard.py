"""
UTCI Live Dashboard — Version A
================================
Τραβάει 24ωρη πρόγνωση καιρού από το Open-Meteo (δωρεάν, χωρίς API key),
υπολογίζει τον δείκτη UTCI με τον επίσημο τύπο (μέσω pythermalcomfort),
και τον δείχνει σε διαδραστικό dashboard με χρωματική κατηγοριοποίηση κινδύνου.

Author: Dimitrios Tetradis
Βασισμένο στη διπλωματική: "Εφαρμογή μοντέλων AI/ML για πρόγνωση θερμικής δυσφορίας"

------------------------------------------------------------------------
ΕΓΚΑΤΑΣΤΑΣΗ (μία φορά, στο terminal):
    pip install streamlit requests pandas plotly pythermalcomfort

ΕΚΤΕΛΕΣΗ:
    streamlit run utci_dashboard.py
------------------------------------------------------------------------
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# pythermalcomfort: επίσημη βιβλιοθήκη για δείκτες θερμικής άνεσης.
# Περιέχει τον τύπο UTCI και βοηθητικές συναρτήσεις (π.χ. εκτίμηση MRT).
from pythermalcomfort.models import utci
from pythermalcomfort.utilities import v_relative  # κρατιέται για μελλοντική χρήση


# ============================================================
# 1. ΡΥΘΜΙΣΕΙΣ ΣΕΛΙΔΑΣ
# ============================================================
st.set_page_config(
    page_title="UTCI Live — Πρόγνωση Θερμικής Δυσφορίας",
    page_icon="🌡️",
    layout="wide",
)


# ============================================================
# 2. ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ UTCI
#    (ακριβώς ο Πίνακας 1 της διπλωματικής σου)
# ============================================================
# Κάθε εγγραφή: (κατώτατο όριο, ετικέτα, χρώμα hex)
# Ταξινομημένα από το ψηλότερο προς το χαμηλότερο όριο.
UTCI_CATEGORIES = [
    (46,   "Ακραίο θερμικό στρες",        "#7a0177"),
    (38,   "Πολύ ισχυρό θερμικό στρες",   "#d7301f"),
    (32,   "Ισχυρό θερμικό στρες",        "#ef6548"),
    (26,   "Μέτριο θερμικό στρες",        "#fc8d59"),
    (9,    "Χωρίς θερμικό στρες",         "#74c476"),
    (0,    "Ήπιο ψυχρό στρες",            "#a6bddb"),
    (-13,  "Μέτριο ψυχρό στρες",          "#74a9cf"),
    (-27,  "Ισχυρό ψυχρό στρες",          "#3690c0"),
    (-40,  "Πολύ ισχυρό ψυχρό στρες",     "#0570b0"),
    (-999, "Ακραίο ψυχρό στρες",          "#034e7b"),
]


def classify_utci(value):
    """Επιστρέφει (ετικέτα, χρώμα) για μια τιμή UTCI σε °C."""
    for threshold, label, color in UTCI_CATEGORIES:
        if value >= threshold:
            return label, color
    return UTCI_CATEGORIES[-1][1], UTCI_CATEGORIES[-1][2]


# ============================================================
# 3. ΛΗΨΗ ΔΕΔΟΜΕΝΩΝ ΑΠΟ OPEN-METEO
# ============================================================
# @st.cache_data: αποθηκεύει το αποτέλεσμα για 30 λεπτά ώστε να μην
# χτυπάμε το API σε κάθε αλληλεπίδραση του χρήστη.
@st.cache_data(ttl=86400)
def geocode_city(name):
    """
    Μετατρέπει όνομα πόλης σε συντεταγμένες μέσω του Geocoding API
    του Open-Meteo (δωρεάν, χωρίς key).

    Επιστρέφει λίστα από dicts: [{label, lat, lon}, ...]
    ή κενή λίστα αν δεν βρεθεί τίποτα.

    Cache 24 ώρες (86400s): οι συντεταγμένες μιας πόλης δεν αλλάζουν,
    οπότε δεν χρειάζεται συχνή ανανέωση.
    """
    if not name or len(name.strip()) < 2:
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": name.strip(),
        "count": 5,            # έως 5 αποτελέσματα (π.χ. ομώνυμες πόλεις)
        "language": "el",      # ελληνικά ονόματα όπου υπάρχουν
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return []

    options = []
    for r in results:
        # Φτιάχνουμε ευανάγνωστη ετικέτα: "Πόλη, Περιοχή, Χώρα"
        parts = [r.get("name")]
        if r.get("admin1"):
            parts.append(r["admin1"])
        if r.get("country"):
            parts.append(r["country"])
        label = ", ".join(p for p in parts if p)
        options.append({
            "label": label,
            "lat": r["latitude"],
            "lon": r["longitude"],
        })
    return options


@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    """
    Ζητάει 24ωρη ωριαία πρόγνωση από το Open-Meteo.
    Επιστρέφει pandas DataFrame με τις μεταβλητές που χρειάζεται ο UTCI.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",        # θερμοκρασία αέρα (°C)
            "relative_humidity_2m",  # σχετική υγρασία (%)
            "wind_speed_10m",        # άνεμος στα 10m
            "shortwave_radiation",   # ολική ηλιακή ακτινοβολία (W/m²)
        ],
        "wind_speed_unit": "ms",     # ΣΗΜΑΝΤΙΚΟ: ο UTCI θέλει m/s, όχι km/h
        "timezone": "auto",          # τοπική ώρα της τοποθεσίας
        "forecast_days": 1,          # επόμενο 24ωρο
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


# ============================================================
# 4. ΕΚΤΙΜΗΣΗ MRT ΚΑΙ ΥΠΟΛΟΓΙΣΜΟΣ UTCI
# ============================================================
def estimate_mrt(tdb, solar_radiation, sun_exposed=True):
    """
    Απλή εκτίμηση Mean Radiant Temperature (MRT) από την ηλιακή ακτινοβολία.

    Ο επίσημος UTCI θέλει MRT, που το Open-Meteo ΔΕΝ δίνει έτοιμο.
    Χρησιμοποιούμε μια πρακτική προσέγγιση: σε σκιά, MRT ≈ θερμοκρασία αέρα.
    Με ηλιοφάνεια, η MRT ανεβαίνει αναλογικά με την ακτινοβολία.

    Ο συντελεστής 0.025 είναι μια συντηρητική, ευρέως χρησιμοποιούμενη
    προσέγγιση για εκτεθειμένο άτομο. Είναι απλοποίηση — στο Version B
    θα μπορούσες να βάλεις πιο σύνθετο μοντέλο (π.χ. solar_gain).

    Παράμετρος sun_exposed:
        True  -> άτομο στον ήλιο: η ηλιακή ακτινοβολία ανεβάζει την MRT.
        False -> άτομο στη σκιά: σχεδόν μηδενική ηλιακή συνεισφορά,
                 οπότε MRT ≈ θερμοκρασία αέρα.
    """
    if sun_exposed:
        return tdb + 0.025 * solar_radiation
    return tdb  # στη σκιά, η MRT πέφτει στη θερμοκρασία αέρα


def compute_utci_series(df, sun_exposed=True):
    """
    Υπολογίζει UTCI για κάθε ώρα του DataFrame.
    Επιστρέφει στήλη 'utci' για το επιλεγμένο σενάριο (ήλιος ή σκιά).
    """
    utci_values = []
    for _, row in df.iterrows():
        tdb = row["temperature_2m"]
        rh = row["relative_humidity_2m"]
        # Ο UTCI ορίζεται για άνεμο 0.5–17 m/s. Κάτω από 0.5 δίνει σφάλμα,
        # οπότε κάνουμε clamp στο ελάχιστο όριο.
        wind = max(row["wind_speed_10m"], 0.5)
        tr = estimate_mrt(tdb, row["shortwave_radiation"], sun_exposed)

        result = utci(tdb=tdb, tr=tr, v=wind, rh=rh)
        # Η νεότερη έκδοση επιστρέφει αντικείμενο με .utci, η παλιά dict.
        value = result.utci if hasattr(result, "utci") else result["utci"]
        utci_values.append(value)

    df = df.copy()
    df["utci"] = utci_values
    return df


# ============================================================
# 5. ΠΡΟΚΑΘΟΡΙΣΜΕΝΕΣ ΤΟΠΟΘΕΣΙΕΣ
# ============================================================
LOCATIONS = {
    "Αθήνα (ΠΑΔΑ – Αρχαίος Ελαιώνας)": (37.9899, 23.6828),
    "Θεσσαλονίκη":                      (40.6401, 22.9444),
    "Ηράκλειο":                         (35.3387, 25.1442),
    "Πάτρα":                            (38.2466, 21.7346),
}


# ============================================================
# 6. ΔΙΕΠΑΦΗ (UI)
# ============================================================
st.title("🌡️ UTCI Live — Πρόγνωση Θερμικής Δυσφορίας")
st.markdown(
    "Πρόγνωση του δείκτη **Universal Thermal Climate Index (UTCI)** "
    "για το επόμενο 24ωρο, βασισμένη σε δεδομένα του Open-Meteo."
)

col_sel, col_info = st.columns([1, 2])
with col_sel:
    mode = st.radio(
        "Επιλογή τοποθεσίας",
        ["Γρήγορη επιλογή", "Αναζήτηση πόλης"],
        horizontal=True,
    )

    if mode == "Γρήγορη επιλογή":
        location_name = st.selectbox("Τοποθεσία", list(LOCATIONS.keys()))
        lat, lon = LOCATIONS[location_name]
        display_name = location_name
    else:
        query = st.text_input(
            "Όνομα πόλης",
            placeholder="π.χ. Λάρισα, Βόλος, Ρόδος...",
        )
        matches = geocode_city(query)
        if query and not matches:
            st.warning("Δεν βρέθηκε πόλη με αυτό το όνομα. Δοκίμασε αλλιώς.")
            st.stop()
        if not matches:
            st.info("Πληκτρολόγησε μια πόλη για να ξεκινήσεις.")
            st.stop()
        # Αν πολλά αποτελέσματα (ομώνυμες πόλεις), αφήνουμε τον χρήστη να διαλέξει
        labels = [m["label"] for m in matches]
        chosen = st.selectbox("Αποτελέσματα", labels)
        selected = next(m for m in matches if m["label"] == chosen)
        lat, lon = selected["lat"], selected["lon"]
        display_name = selected["label"]

    exposure = st.radio(
        "Έκθεση",
        ["Στον ήλιο", "Στη σκιά"],
        horizontal=True,
        help="Η Mean Radiant Temperature διαφέρει δραστικά με την έκθεση. "
             "Στον ήλιο, η ηλιακή ακτινοβολία ανεβάζει αισθητά τον UTCI.",
    )
sun_exposed = (exposure == "Στον ήλιο")

# --- Λήψη & υπολογισμός ---
# Υπολογίζουμε ΚΑΙ τα δύο σενάρια: το επιλεγμένο για τα metrics,
# και τα δύο μαζί για το διάγραμμα (ώστε να φαίνεται το εύρος αβεβαιότητας).
try:
    with st.spinner("Λήψη δεδομένων από Open-Meteo..."):
        weather_raw = fetch_weather(lat, lon)
        weather_sun = compute_utci_series(weather_raw, sun_exposed=True)
        weather_shade = compute_utci_series(weather_raw, sun_exposed=False)
except Exception as e:
    st.error(f"Σφάλμα κατά τη λήψη/υπολογισμό: {e}")
    st.stop()

# Το ενεργό σενάριο για metrics/κατηγοριοποίηση
weather = weather_sun if sun_exposed else weather_shade

# --- Τρέχουσα κατάσταση (πρώτη διαθέσιμη ώρα) ---
current = weather.iloc[0]
cur_label, cur_color = classify_utci(current["utci"])

with col_info:
    c1, c2, c3 = st.columns(3)
    c1.metric("UTCI τώρα", f"{current['utci']:.1f} °C")
    c2.metric("Θερμοκρασία αέρα", f"{current['temperature_2m']:.1f} °C")
    c3.metric("Υγρασία", f"{current['relative_humidity_2m']:.0f} %")
    st.markdown(
        f"<div style='padding:8px 12px;border-radius:8px;"
        f"background:{cur_color};color:white;font-weight:600;"
        f"display:inline-block'>{cur_label}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# --- Διάγραμμα 24ώρου ---
st.subheader(f"Πρόγνωση UTCI — {display_name}")
st.caption(
    "Οι δύο καμπύλες δείχνουν το εύρος ανάλογα με την έκθεση. Η διαφορά "
    "τους τις μεσημεριανές ώρες αποτυπώνει την αβεβαιότητα της εκτίμησης MRT."
)

# Χρώμα ανά σημείο για το ενεργό σενάριο
point_colors = [classify_utci(v)[1] for v in weather["utci"]]

fig = go.Figure()

# Καμπύλη σκιάς (πάντα η χαμηλότερη) — λεπτή, διακριτική
fig.add_trace(go.Scatter(
    x=weather_shade["time"],
    y=weather_shade["utci"],
    mode="lines",
    line=dict(color="#74a9cf", width=1.5, dash="dot"),
    name="Στη σκιά",
    hovertemplate="%{x|%H:%M}<br>Σκιά: %{y:.1f} °C<extra></extra>",
))

# Καμπύλη ήλιου (πάντα η υψηλότερη)
fig.add_trace(go.Scatter(
    x=weather_sun["time"],
    y=weather_sun["utci"],
    mode="lines",
    line=dict(color="#fc8d59", width=1.5, dash="dot"),
    name="Στον ήλιο",
    hovertemplate="%{x|%H:%M}<br>Ήλιος: %{y:.1f} °C<extra></extra>",
))

# Ενεργό σενάριο — έντονο, με χρωματιστά σημεία κατηγορίας
fig.add_trace(go.Scatter(
    x=weather["time"],
    y=weather["utci"],
    mode="lines+markers",
    line=dict(color="#888", width=2),
    marker=dict(size=9, color=point_colors),
    name=f"Επιλογή: {exposure.lower()}",
    hovertemplate="%{x|%H:%M}<br>UTCI: %{y:.1f} °C<extra></extra>",
))

# Σκιασμένες ζώνες θερμικού στρες για οπτικό context
for threshold, label, color in UTCI_CATEGORIES:
    if 26 <= threshold <= 46:
        fig.add_hline(y=threshold, line_dash="dot",
                      line_color=color, opacity=0.4)

fig.update_layout(
    height=420,
    xaxis_title="Ώρα",
    yaxis_title="UTCI (°C)",
    margin=dict(l=40, r=20, t=20, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# --- Πίνακας δεδομένων ---
with st.expander("Δες αναλυτικά τα ωριαία δεδομένα"):
    display = weather[[
        "time", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "shortwave_radiation", "utci",
    ]].copy()
    display["Κατηγορία"] = [classify_utci(v)[0] for v in weather["utci"]]
    display.columns = [
        "Ώρα", "Θερμ. (°C)", "Υγρασία (%)", "Άνεμος (m/s)",
        "Ακτινοβολία (W/m²)", "UTCI (°C)", "Κατηγορία",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(
    "Δεδομένα: Open-Meteo.com (CC-BY 4.0) · "
    "UTCI: pythermalcomfort · MRT: απλοποιημένη εκτίμηση από ηλιακή ακτινοβολία"
)
