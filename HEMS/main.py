from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
import joblib
import numpy as np
import os
import json
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HEMS.schemas import (
    AcPredictionRequest,
    AcPredictionResponse,
    HealthResponse,
    OccupancyEstimateRequest,
    OccupancyEstimateResponse,
    NlpParseRequest,
    NlpParseResponse,
    FuzzyControlRequest,
    FuzzyControlResponse,
    RecommendationRequest,
    RecommendationResponse,
    ComponentOutput,
)


# Initialize FastAPI app
app = FastAPI(
    title="HEMS Phase 4 Backend",
    version="0.1.0",
    description="""
    HEMS (Home Energy Management System) Phase 4 FastAPI Backend.
    
    API endpoints for AC mode prediction, occupancy estimation, NLP command parsing,
    and fuzzy logic control for smart home energy management.
    
    Work one step at a time and verify outputs before continuing.
    » Start with GET /health to verify the server works.
    » Then proceed to /predict/ac for AC mode prediction.
    » Then /occupancy/estimate for occupancy estimation.
    » Then /commands/parse for NLP command parsing.
    » Then /control/fuzzy for fuzzy logic control.
    » Then /recommendation for unified component outputs.
    """,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load ML model at startup ---

_model: Any = None
_model_loaded: bool = False


def get_model() -> Any:
    """Lazy-load the Random Forest model, creating a global singleton."""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hems_random_forest.joblib",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Ensure hems_random_forest.joblib is in the project root."
        )

    try:
        _model = joblib.load(model_path)
        _model_loaded = True
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        _model_loaded = False
        raise

    return _model


# --- Phase 3A: Bayesian Occupancy Estimator ---

_occupancy_config: Any = None


def get_occupancy_config() -> Any:
    """Load the Phase 3A Bayesian occupancy config."""
    global _occupancy_config

    if _occupancy_config is not None:
        return _occupancy_config

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "HEMS_Phase3A_Artifacts",
        "occupancy_config.json",
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Occupancy config not found at: {config_path}"
        )

    try:
        with open(config_path, "r") as f:
            _occupancy_config = json.load(f)
        print(f"✅ Occupancy config loaded from: {config_path}")
    except Exception as e:
        print(f"❌ Failed to load occupancy config: {e}")
        _occupancy_config = None
        raise

    return _occupancy_config


def estimate_occupancy_bayesian(timestamp: str, motion_detected: bool) -> dict:
    """Estimate occupancy using the Bayesian approach from Phase 3A.
    
    Uses time-period priors and motion sensor likelihood to compute
    occupancy probability. Probabilities are illustrative/provisional,
    not trained on ground-truth labels (per PRD limitation).
    """
    config = get_occupancy_config()
    
    ts_lower = timestamp.lower().strip()
    
    time_period = "unknown"
    if any(kw in ts_lower for kw in ["morning", "am", "0:00-12:00", "6am", "8am"]):
        time_period = "morning"
    elif any(kw in ts_lower for kw in ["afternoon", "pm", "12:00-18:00", "2pm", "4pm"]):
        time_period = "afternoon"
    elif any(kw in ts_lower for kw in ["evening", "p.m.", "18:00-21:00", "6pm", "9pm"]):
        time_period = "evening"
    elif any(kw in ts_lower for kw in ["night", "a.m.", "21:00-6:00", "10pm", "12am"]):
        time_period = "night"
    
    time_periods = config.get("time_periods", {})
    p_occupied = time_periods.get(time_period, 0.5)  # default to 0.5 if unknown
    p_empty = 1 - p_occupied
    
    p_motion_given_occupied = config.get("p_motion_given_occupied", 0.9)
    p_motion_given_empty = config.get("p_motion_given_empty", 0.05)
    
    if motion_detected:
        p_motion = (p_motion_given_occupied * p_occupied) + (p_motion_given_empty * p_empty)
        if p_motion > 0:
            p_occupied_given_motion = (p_motion_given_occupied * p_occupied) / p_motion
        else:
            p_occupied_given_motion = p_occupied
    else:
        p_no_motion_given_occupied = 1 - p_motion_given_occupied
        p_no_motion_given_empty = 1 - p_motion_given_empty
        p_no_motion = (p_no_motion_given_occupied * p_occupied) + (p_no_motion_given_empty * p_empty)
        if p_no_motion > 0:
            p_occupied_given_no_motion = (p_no_motion_given_occupied * p_occupied) / p_no_motion
        else:
            p_occupied_given_no_motion = p_occupied
    
    occupancy_percentage = round(
        p_occupied_given_motion * 100 if motion_detected else p_occupied_given_no_motion * 100, 2
    )
    likely_occupied = occupancy_percentage > 50
    
    return {
        "time_period": time_period,
        "motion_detected": motion_detected,
        "occupancy_probability": round(
            p_occupied_given_motion if motion_detected else p_occupied_given_no_motion, 4
        ),
        "occupancy_percentage": occupancy_percentage,
        "likely_occupied": likely_occupied,
        "provisional_note": config.get("assumption", "Illustrative probabilities; not trained or validated against ground-truth occupancy labels.")
    }


# --- Phase 3B: NLP Command Parser ---

def parse_nlp_command(command: str) -> dict:
    """Parse a natural language command using the Phase 3B parser."""
    phase_3b_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output phase 3b"
    )
    if phase_3b_path not in sys.path:
        sys.path.insert(0, phase_3b_path)
    
    from nlp_parser import parse_command as parse_nlp
    
    if not isinstance(command, str) or not command.strip():
        return {
            "command": command, "intent": "unknown",
            "appliance": None, "action": None, "value": None,
            "confidence": 0.0, "needs_clarification": True,
            "message": "Please enter a command."
        }
    
    result = parse_nlp(command.lower().strip())
    return {
        "command": result["command"],
        "intent": result["intent"],
        "appliance": result["appliance"],
        "action": result["action"],
        "value": result["value"],
        "confidence": result["confidence"],
        "needs_clarification": result["needs_clarification"],
        "message": result["message"],
        "tokens": result.get("tokens", [])
    }


# --- Phase 3C: Fuzzy Logic Controller ---

_fuzzy_controller_imported: bool = False


def get_fuzzy_controller() -> Any:
    """Ensure fuzzy controller module is importable."""
    global _fuzzy_controller_imported

    if _fuzzy_controller_imported:
        return

    try:
        import skfuzzy  # noqa: F401  # verify import works
        _fuzzy_controller_imported = True
        print("✅ Fuzzy controller dependencies available")
    except ImportError:
        print("⚠️ scikit-fuzzy not available, fuzzy endpoint will fail")
        _fuzzy_controller_imported = True


def predict_fuzzy_ac_mode(temperature_c: float, humidity_pct: float, occupancy_pct: float) -> dict:
    """Predict AC mode using the Phase 3C fuzzy logic controller."""
    import importlib
    import os
    
    output_3c_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Output 3C"
    )
    if output_3c_path not in sys.path:
        sys.path.insert(0, output_3c_path)
    
    try:
        from fuzzy_controller import predict_ac_mode as predict_fuzzy
        result = predict_fuzzy(temperature_c, humidity_pct, occupancy_pct)
        return {
            "temperature_c": result["temperature_c"],
            "humidity_pct": result["humidity_pct"],
            "occupancy_pct": result["occupancy_pct"],
            "cooling_demand": result["cooling_demand"],
            "suggested_mode": result["suggested_mode"],
            "model_note": (
                "Membership functions, fuzzy rules, and AC mode thresholds are "
                "provisional engineering choices. The controller has not been validated "
                "against measured energy savings, real-world comfort measurements, "
                "or a physical AC unit. See PRD for full limitations."
            ),
        }
    except Exception as e:
        raise ImportError(f"Fuzzy prediction failed: {e}")


# --- FastAPI App Routes ---

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["basic"],
)
def health_check() -> HealthResponse:
    """Root health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="HEMS Phase 4 Backend",
        version="0.1.0",
        message="HEMS backend is running successfully.",
    )


@app.post(
    "/predict/ac",
    response_model=AcPredictionResponse,
    summary="Predict AC mode using Random Forest model",
    tags=["ml-prediction"],
    responses={422: {"description": "Validation error"}, 500: {"description": "Internal error"}},
)
def predict_ac_endpoint(request: AcPredictionRequest) -> AcPredictionResponse:
    """POST /predict/ac - Predict AC mode using the pretrained Random Forest model.
    
    The model expects 6 features: ['ambient_temp', 'relative_humidity', 'motion_sensor', 
    'grid_power', 'hour', 'day_of_week']
    """
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail={"error": "Model file not found"})

    features = np.array([
        [request.ambient_temp, request.relative_humidity, request.motion_sensor,
         request.grid_power, request.hour, request.day_of_week]
    ])

    try:
        predicted_code = int(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Prediction failed: {str(e)}"})

    label_map = {0: "Eco", 1: "Max Cool", 2: "Dry"}
    predicted_mode = label_map.get(predicted_code, f"Unknown_{predicted_code}")

    probability: Optional[float] = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            probability = round(float(probs[predicted_code]), 4)
    except Exception:
        pass

    return AcPredictionResponse(
        ambient_temp=request.ambient_temp,
        relative_humidity=request.relative_humidity,
        motion_sensor=request.motion_sensor,
        grid_power=request.grid_power,
        hour=request.hour,
        day_of_week=request.day_of_week,
        predicted_mode=predicted_mode,
        predicted_mode_code=predicted_code,
        probability=probability,
        model_limitations=(
            "Model predicts among 3 modes: Eco, Max Cool, Dry. Does not predict 'Off' mode. "
            "Class mapping derived from dataset analysis."
        ),
    )


@app.post(
    "/occupancy/estimate",
    response_model=OccupancyEstimateResponse,
    summary="Estimate occupancy using Bayesian model",
    tags=["occupancy"],
    responses={422: {"description": "Validation error"}},
)
def estimate_occupancy_endpoint(request: OccupancyEstimateRequest) -> OccupancyEstimateResponse:
    """POST /occupancy/estimate - Estimate occupancy using the Bayesian model from Phase 3A.
    
    Probabilities are provisional/illustrative assumptions, not ground-truth validated.
    """
    try:
        result = estimate_occupancy_bayesian(request.timestamp, request.motion_detected)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

    return OccupancyEstimateResponse(
        time_period=result["time_period"],
        motion_detected=result["motion_detected"],
        occupancy_probability=result["occupancy_probability"],
        occupancy_percentage=result["occupancy_percentage"],
        likely_occupied=result["likely_occupied"],
        provisional_note=result["provisional_note"],
    )


@app.post(
    "/commands/parse",
    response_model=NlpParseResponse,
    summary="Parse natural language command using NLP parser",
    tags=["nlp"],
    responses={422: {"description": "Validation error"}},
)
def parse_command_endpoint(request: NlpParseRequest) -> NlpParseResponse:
    """POST /commands/parse - Parse a natural language command using the Phase 3B rule-based parser.
    
    The parser is rule-based and must not be described as a fully trained language model.
    Unknown and ambiguous commands should request clarification rather than execute
    an assumed action.
    """
    try:
        result = parse_nlp_command(request.command)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Parsing failed: {str(e)}"})

    return NlpParseResponse(
        command=result["command"],
        intent=result["intent"],
        appliance=result["appliance"],
        action=result["action"],
        value=result["value"],
        confidence=result["confidence"],
        needs_clarification=result["needs_clarification"],
        message=result["message"],
    )


@app.post(
    "/control/fuzzy",
    response_model=FuzzyControlResponse,
    summary="Compute fuzzy logic control output",
    tags=["fuzzy-control"],
    responses={422: {"description": "Validation error"}, 500: {"description": "Internal error"}},
)
def fuzzy_control_endpoint(request: FuzzyControlRequest) -> FuzzyControlResponse:
    """POST /control/fuzzy - Compute AC mode and cooling demand using the Phase 3C fuzzy logic controller.
    
    Important limitations per PRD:
    - Membership functions and fuzzy rules are provisional engineering choices
    - The controller has not been validated against measured energy savings,
      real-world comfort measurements, or a physical AC unit
    - Current policy chooses Max Cool at temperatures of 38°C or above, even at low occupancy
    """
    try:
        get_fuzzy_controller()
        result = predict_fuzzy_ac_mode(
            request.temperature_c, request.humidity_pct, request.occupancy_pct
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Fuzzy computation failed: {str(e)}"})

    return FuzzyControlResponse(
        temperature_c=request.temperature_c,
        humidity_pct=request.humidity_pct,
        occupancy_pct=request.occupancy_pct,
        cooling_demand=result["cooling_demand"],
        suggested_mode=result["suggested_mode"],
        model_note=result["model_note"],
    )


@app.post(
    "/recommendation",
    response_model=RecommendationResponse,
    summary="Unified recommendation combining all available outputs",
    tags=["unified"],
    responses={422: {"description": "Validation error"}, 500: {"description": "Internal error"}},
)
def recommendation_endpoint(request: RecommendationRequest) -> RecommendationResponse:
    """POST /recommendation - Combine outputs from all available HEMS components.
    
    Keeps component outputs distinguishable and documents the combination logic.
    Does NOT treat AC mode, occupancy estimate, and fuzzy recommendation as equivalent.
    
    Returns:
    - component_outputs: distinguishable outputs from each component (AC, occupancy, fuzzy)
    - overall_recommendation: structured summary noting differences between components
    - disclaimer: documentation of all limitations and provisional nature
    """
    component_outputs: list[ComponentOutput] = []
    disclaimer_parts: list[str] = []

    # --- AC Prediction Component ---
    ac_output: Dict[str, Any] = {"status": "not_computed", "mode": "Unknown"}
    try:
        # Use the same feature order as the Random Forest model expects
        # But the recommendation endpoint has different inputs (temp, humidity, occupancy_pct)
        # Instead, we'll derive a simple temperature-based AC mode and note the limitation
        temp = request.temperature_c
        if temp >= 38:
            ac_mode = "Max Cool"
        elif temp >= 32:
            ac_mode = "Dry"
        elif temp >= 25:
            ac_mode = "Eco"
        else:
            ac_mode = "Off"
        
        ac_output = {
            "status": "computed",
            "mode": ac_mode,
            "temperature_c": temp,
            "humidity_pct": request.humidity_pct,
            "note": "Simple temperature-based AC mode; full Random Forest model requires "
                    "ambient_temp, relative_humidity, motion_sensor, grid_power, hour, day_of_week"
        }
        component_outputs.append(
            ComponentOutput(
                component="ac_prediction",
                function="temperature_based_ac_mode",
                input_summary={
                    "temperature_c": request.temperature_c,
                    "humidity_pct": request.humidity_pct,
                    "occupancy_pct": request.occupancy_pct,
                },
                output=ac_output,
            )
        )
        disclaimer_parts.append(
            "AC mode is temperature-threshold based; full Random Forest model requires "
            "additional inputs (motion_sensor, grid_power, hour, day_of_week)."
        )
    except Exception as e:
        component_outputs.append(
            ComponentOutput(
                component="ac_prediction",
                function="temperature_based_ac_mode",
                input_summary={},
                output={"status": "error", "message": str(e)},
            )
        )
        disclaimer_parts.append(f"AC prediction failed: {str(e)[:80]}")

    # --- Occupancy Estimation Component ---
    occupancy_output: Dict[str, Any] = {"status": "not_computed"}
    try:
        from HEMS.main import estimate_occupancy_bayesian
        # Determine motion proxy from occupancy_pct
        motion_proxy = request.occupancy_pct > 50
        occ_result = estimate_occupancy_bayesian("unknown_timestamp", motion_proxy)
        occupancy_output = {
            "status": "computed",
            "time_period": occ_result.get("time_period", "unknown"),
            "occupancy_percentage": occ_result.get("occupancy_percentage", 0),
            "likely_occupied": occ_result.get("likely_occupied", False),
        }
        component_outputs.append(
            ComponentOutput(
                component="occupancy_estimation",
                function="bayesian_occupancy_estimator",
                input_summary={
                    "occupancy_pct": request.occupancy_pct,
                },
                output=occupancy_output,
            )
        )
        disclaimer_parts.append(
            "Bayesian occupancy probabilities are provisional/illustrative assumptions, "
            "not trained or validated against ground-truth occupancy labels (per PRD limitation)."
        )
    except Exception as e:
        component_outputs.append(
            ComponentOutput(
                component="occupancy_estimation",
                function="bayesian_occupancy_estimator",
                input_summary={},
                output={"status": "error", "message": str(e)},
            )
        )
        disclaimer_parts.append(f"Occupancy estimation failed: {str(e)[:80]}")

    # --- Fuzzy Logic Controller Component ---
    fuzzy_output: Dict[str, Any] = {"status": "not_computed"}
    try:
        from HEMS.main import predict_fuzzy_ac_mode
        fuzzy_result = predict_fuzzy_ac_mode(
            request.temperature_c, request.humidity_pct, request.occupancy_pct
        )
        fuzzy_output = {
            "status": "computed",
            "cooling_demand": fuzzy_result.get("cooling_demand", 0),
            "suggested_mode": fuzzy_result.get("suggested_mode", "Off"),
        }
        component_outputs.append(
            ComponentOutput(
                component="fuzzy_logic_controller",
                function="predict_ac_mode",
                input_summary={
                    "temperature_c": request.temperature_c,
                    "humidity_pct": request.humidity_pct,
                    "occupancy_pct": request.occupancy_pct,
                },
                output=fuzzy_output,
            )
        )
        disclaimer_parts.append(
            "Membership functions, fuzzy rules, and AC mode thresholds are provisional engineering "
            "choices. The controller has not been validated against measured energy savings, "
            "real-world comfort measurements, or a physical AC unit."
        )
    except Exception as e:
        component_outputs.append(
            ComponentOutput(
                component="fuzzy_logic_controller",
                function="predict_ac_mode",
                input_summary={},
                output={"status": "error", "message": str(e)},
            )
        )
        disclaimer_parts.append(f"Fuzzy control failed: {str(e)[:80]}")

    # --- Overall Recommendation Logic ---
    # Collect distinguishable results from each component
    ac_mode = "Unknown"
    fuzzy_mode = "Unknown"
    occupancy_pct_value = request.occupancy_pct

    for co in component_outputs:
        if co.component == "ac_prediction" and co.output.get("status") == "computed":
            ac_mode = co.output.get("mode", "Unknown")
        if co.component == "fuzzy_logic_controller" and co.output.get("status") == "computed":
            fuzzy_mode = co.output.get("suggested_mode", "Unknown")

    overall_recommendation: Dict[str, Any] = {
        "ac_mode": ac_mode,
        "fuzzy_suggested_mode": fuzzy_mode,
        "occupancy_percentage": occupancy_pct_value,
        "note": (
            "AC mode, occupancy estimate, and fuzzy recommendation are "
            "distinguishable outputs with different purposes. Do not equate them. "
            "See component_outputs for detailed individual results."
        ),
    }

    # Add component-specific disclaimers to the overall recommendation
    for co in component_outputs:
        if co.component == "ac_prediction" and "note" in co.output:
            overall_recommendation["ac_disclaimer"] = co.output["note"]
        if co.component == "occupancy_estimation" and "provisional_note" in co.output:
            overall_recommendation["occupancy_disclaimer"] = co.output["provisional_note"]
        if co.component == "fuzzy_logic_controller" and "model_note" in co.output:
            overall_recommendation["fuzzy_disclaimer"] = co.output["model_note"]

    return RecommendationResponse(
        component_outputs=component_outputs,
        overall_recommendation=overall_recommendation,
        disclaimer="; ".join(disclaimer_parts)
        if disclaimer_parts
        else "All components executed successfully. See component_outputs for details.",
    )


# --- Optional: test endpoint during development ---

@app.get("/test/echo", include_in_schema=False)
def test_echo(message: str = "hello") -> Dict[str, str]:
    """Simple echo endpoint for development testing (not part of official API)."""
    return {"echo": message}