# HEMS API

Home Energy Management System (HEMS) FastAPI backend.

The API provides AC mode prediction, occupancy estimation, natural-language
command parsing, fuzzy AC control, and a unified recommendation endpoint.

## Project setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r HEMS/requirements.txt
```

Start the development server:

```bash
python -m uvicorn HEMS.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Running tests

Install the project dependencies, including the test dependency, with:

```bash
python -m pip install -r HEMS/requirements.txt
```

Run the Phase 5 integration test suite from the repository root:

```bash
python -m pytest tests/test_api.py -q
```

The suite covers:

- Health checks
- Random Forest prediction and model failures
- Input range and data-type validation
- Bayesian occupancy estimation
- Valid, unknown, and conflicting NLP commands
- Fuzzy control and import failures
- Unified recommendation outputs and component disagreement
- Missing required fields

Latest verified result:

```text
34 passed, 32 warnings
```

The warnings are from dependency compatibility and deprecation notices:
Pydantic V1-style validators, the Starlette/httpx test client integration,
scikit-learn model-version metadata, and a scikit-fuzzy/NumPy deprecation.

## API endpoints

### 1. Health check

**Endpoint:** `GET /health`

Checks whether the backend is running.

#### Request

```bash
curl http://127.0.0.1:8000/health
```

#### Output

```json
{
  "status": "healthy",
  "service": "HEMS Phase 4 Backend",
  "version": "0.1.0",
  "message": "HEMS backend is running successfully."
}
```

HTTP status: `200 OK`

---

### 2. AC mode prediction

**Endpoint:** `POST /predict/ac`

Uses the pretrained Random Forest model to predict one of the following modes:

- `Eco`
- `Max Cool`
- `Dry`

The model expects these six inputs:

| Field | Type | Description | Valid range |
|---|---|---|---|
| `ambient_temp` | number | Ambient temperature in Celsius | `15` to `45` |
| `relative_humidity` | number | Relative humidity percentage | `0` to `100` |
| `motion_sensor` | integer | Motion sensor state | `0` or `1` |
| `grid_power` | number | Normalized grid power | `0` to `1` |
| `hour` | integer | Hour of day | `0` to `23` |
| `day_of_week` | integer | Day of week, Monday = `0` | `0` to `6` |

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/predict/ac" \
  -H "Content-Type: application/json" \
  -d '{
    "ambient_temp": 32,
    "relative_humidity": 70,
    "motion_sensor": 1,
    "grid_power": 0.8,
    "hour": 14,
    "day_of_week": 3
  }'
```

#### Output

```json
{
  "ambient_temp": 32.0,
  "relative_humidity": 70.0,
  "motion_sensor": 1,
  "grid_power": 0.8,
  "hour": 14,
  "day_of_week": 3,
  "predicted_mode": "Eco",
  "predicted_mode_code": 0,
  "probability": 0.6839,
  "model_input_feature_order": [
    "ambient_temp",
    "relative_humidity",
    "motion_sensor",
    "grid_power",
    "hour",
    "day_of_week"
  ],
  "model_limitations": "Model predicts among 3 modes: Eco, Max Cool, Dry. Does not predict 'Off' mode. Class mapping derived from dataset analysis."
}
```

HTTP status: `200 OK`

---

### 3. Occupancy estimation

**Endpoint:** `POST /occupancy/estimate`

Estimates occupancy using the Bayesian occupancy component.

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | ISO timestamp or time-period identifier |
| `motion_detected` | boolean | Whether motion was detected |

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/occupancy/estimate" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "morning",
    "motion_detected": true
  }'
```

#### Output

```json
{
  "time_period": "morning",
  "motion_detected": true,
  "occupancy_probability": 0.9474,
  "occupancy_percentage": 94.74,
  "likely_occupied": true,
  "provisional_note": "Illustrative probabilities; not trained or validated against ground-truth occupancy labels."
}
```

HTTP status: `200 OK`

---

### 4. Natural-language command parsing

**Endpoint:** `POST /commands/parse`

Parses supported natural-language appliance commands.

| Field | Type | Description |
|---|---|---|
| `command` | string | Natural-language appliance command |

Example supported commands include:

```text
turn on the AC
turn off the fan
set the AC to eco mode
save energy while I am away
```

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/commands/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "turn on the AC"
  }'
```

#### Output

```json
{
  "command": "turn on the ac",
  "intent": "power_control",
  "appliance": "ac",
  "action": "set_power",
  "value": "on",
  "confidence": 0.95,
  "needs_clarification": false,
  "message": "ac power command identified: on."
}
```

HTTP status: `200 OK`

---

### 5. Fuzzy AC control

**Endpoint:** `POST /control/fuzzy`

Computes cooling demand and suggests an AC mode using the fuzzy logic
controller.

| Field | Type | Description | Valid range |
|---|---|---|---|
| `temperature_c` | number | Temperature in Celsius | `15` to `45` |
| `humidity_pct` | number | Relative humidity percentage | `0` to `100` |
| `occupancy_pct` | number | Occupancy percentage | `0` to `100` |

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/control/fuzzy" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_c": 32,
    "humidity_pct": 70,
    "occupancy_pct": 90
  }'
```

#### Output

```json
{
  "temperature_c": 32.0,
  "humidity_pct": 70.0,
  "occupancy_pct": 90.0,
  "cooling_demand": 72.56,
  "suggested_mode": "Dry",
  "model_note": "Membership functions, fuzzy rules, and AC mode thresholds are provisional engineering choices. The controller has not been validated against measured energy savings, real-world comfort measurements, or a physical AC unit. See PRD for full limitations."
}
```

HTTP status: `200 OK`

---

### 6. Unified recommendation

**Endpoint:** `POST /recommendation`

Combines the temperature-based AC recommendation, Bayesian occupancy estimate,
and fuzzy controller output.

| Field | Type | Description | Valid range |
|---|---|---|---|
| `temperature_c` | number | Temperature in Celsius | `15` to `45` |
| `humidity_pct` | number | Relative humidity percentage | `0` to `100` |
| `occupancy_pct` | number | Occupancy percentage | `0` to `100` |
| `command` | string or null | Optional natural-language command | Optional |

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/recommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_c": 32,
    "humidity_pct": 70,
    "occupancy_pct": 90,
    "command": "turn on the AC"
  }'
```

#### Output

```json
{
  "component_outputs": [
    {
      "component": "ac_prediction",
      "function": "temperature_based_ac_mode",
      "input_summary": {
        "temperature_c": 32.0,
        "humidity_pct": 70.0,
        "occupancy_pct": 90.0
      },
      "output": {
        "status": "computed",
        "mode": "Dry",
        "temperature_c": 32.0,
        "humidity_pct": 70.0,
        "note": "Simple temperature-based AC mode; full Random Forest model requires ambient_temp, relative_humidity, motion_sensor, grid_power, hour, day_of_week"
      }
    },
    {
      "component": "occupancy_estimation",
      "function": "bayesian_occupancy_estimator",
      "input_summary": {
        "occupancy_pct": 90.0
      },
      "output": {
        "status": "computed",
        "time_period": "morning",
        "occupancy_percentage": 94.74,
        "likely_occupied": true
      }
    },
    {
      "component": "fuzzy_logic_controller",
      "function": "predict_ac_mode",
      "input_summary": {
        "temperature_c": 32.0,
        "humidity_pct": 70.0,
        "occupancy_pct": 90.0
      },
      "output": {
        "status": "computed",
        "cooling_demand": 72.56,
        "suggested_mode": "Dry"
      }
    }
  ],
  "overall_recommendation": {
    "ac_mode": "Dry",
    "fuzzy_suggested_mode": "Dry",
    "occupancy_percentage": 90.0,
    "note": "AC mode, occupancy estimate, and fuzzy recommendation are distinguishable outputs with different purposes. Do not equate them. See component_outputs for detailed individual results.",
    "ac_disclaimer": "Simple temperature-based AC mode; full Random Forest model requires additional inputs (motion_sensor, grid_power, hour, day_of_week)"
  },
  "disclaimer": "AC mode is temperature-threshold based; full Random Forest model requires additional inputs (motion_sensor, grid_power, hour, day_of_week).; Bayesian occupancy probabilities are provisional/illustrative assumptions, not trained or validated against ground-truth occupancy labels (per PRD limitation).; Membership functions, fuzzy rules, and AC mode thresholds are provisional engineering choices. The controller has not been validated against measured energy savings, real-world comfort measurements, or a physical AC unit."
}
```

HTTP status: `200 OK`

---

### 7. Echo test endpoint

**Endpoint:** `GET /test/echo`

This endpoint is excluded from the OpenAPI schema and is intended for simple
server connectivity tests.

#### Request

```bash
curl "http://127.0.0.1:8000/test/echo?message=hello"
```

#### Output

```json
{
  "echo": "hello"
}
```

HTTP status: `200 OK`

## Notes and limitations

- The Random Forest model predicts `Eco`, `Max Cool`, or `Dry`; it does not
  predict `Off`.
- Occupancy probabilities are illustrative and have not been validated against
  ground-truth occupancy labels.
- The fuzzy controller uses provisional membership functions, rules, and mode
  thresholds. It has not been validated against measured energy savings,
  comfort measurements, or a physical AC unit.
- The unified recommendation keeps the outputs from each component separate;
  the component outputs should not be treated as equivalent.
- Input validation errors return HTTP `422`.

## Phase 6 dashboard (React + weather)

The `phase-6-green-weather-dashboard` branch includes a simple, responsive
smart-home dashboard in
[frontend/](/home/dakshhinde/Code/DSL-2/dsl-mini-project/frontend). It uses
React, TypeScript, and Vite. The existing FastAPI backend remains separate.

Start the backend first:

```bash
source .venv/bin/activate
python -m uvicorn HEMS.main:app --reload
```

In a second terminal, start the Vite frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5500
```

For a production build, run `npm run build` from `frontend/`.

Dashboard behavior:

- Shows the actual backend connection state from `GET /health`.
- Loads recommendation, occupancy, cooling-demand, and suggested-mode data
  from `POST /recommendation`.
- Loads estimated outdoor conditions near Shree L. R. Tiwari College of
  Engineering from Open-Meteo. Weather values are not indoor sensor readings.
- Sends commands to `POST /commands/parse` and displays the returned intent,
  confidence, and clarification state.
- Supports explicit Live API mode and deterministic Demo mode with four
  scenarios: Hot Afternoon, Humid Evening, Home Unoccupied, and Comfortable
  Night.
- Labels appliance states as unavailable because no physical device-control API
  exists.
