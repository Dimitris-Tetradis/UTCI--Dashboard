# 🌡️ UTCI Live — Thermal Comfort Forecast Dashboard

A live web dashboard that forecasts the **Universal Thermal Climate Index (UTCI)**
for the next 24 hours at any location, using real-time weather data.

**🔗 Live app:** [utci--dashboard.streamlit.app](https://utci--dashboard.streamlit.app/)

---

## What it does

UTCI is a biometeorological index that expresses human thermal stress as an
equivalent temperature (°C), combining air temperature, humidity, wind, and
radiation into a single physiologically-grounded value. This app:

- Fetches a 24-hour hourly weather forecast from the **Open-Meteo API** (free, no key required)
- Computes UTCI for each hour using the official formula (via `pythermalcomfort`)
- Classifies each value into the standard 10-level thermal stress scale
- Lets the user search **any city worldwide** (Open-Meteo Geocoding API)
- Visualises the forecast with colour-coded risk zones

## The Mean Radiant Temperature problem

The official UTCI requires four inputs: air temperature, humidity, wind speed,
and **Mean Radiant Temperature (MRT)**. Open-Meteo provides the first three but
not MRT, which depends on solar exposure.

Rather than hide this assumption, the app makes it an explicit, user-controlled
parameter through a **sun / shade toggle**:

- **In the sun:** MRT is estimated from solar radiation, raising the UTCI
- **In the shade:** MRT ≈ air temperature

The chart shows both curves simultaneously. Their divergence at midday makes the
**uncertainty of the MRT estimation visible**, instead of presenting a single
potentially misleading number.

## Background

This project builds on my diploma thesis at the **University of West Attica**,
*"Application of artificial intelligence and machine learning models to predict
thermal discomfort/comfort levels due to adverse weather conditions"*, which
developed and compared ML models (Artificial Neural Networks, Gaussian Process
Regression) for UTCI prediction from urban microclimate data.

This dashboard is the deployment-focused continuation of that work, turning a
static analysis into a live, interactive tool.

## Tech stack

- **Python** — core language
- **Streamlit** — web app framework
- **Open-Meteo API** — weather forecast + geocoding
- **pythermalcomfort** — official UTCI computation
- **Plotly** — interactive charts
- **pandas** — data handling

## Running locally

```bash
# Clone the repository
git clone https://github.com/Dimitris-Tetradis/UTCI--Dashboard.git
cd UTCI--Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run utci_dashboard.py
```

The app will open in your browser at `localhost:8501`.

## Data sources & attribution

- Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC-BY 4.0)
- UTCI computation: [pythermalcomfort](https://pythermalcomfort.readthedocs.io/)

## Roadmap

- [ ] Integrate the trained GPR model from the thesis as a live predictor
- [ ] Compare model predictions against the official UTCI in real time
- [ ] Add a more rigorous MRT estimation (e.g. `solar_gain`)

---

*Built by Dimitrios Tetradis.*
