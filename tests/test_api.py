import pytest
from fastapi.testclient import TestClient

from HEMS import main


client = TestClient(main.app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_random_forest_prediction() -> None:
    response = client.post(
        "/predict/ac",
        json={
            "ambient_temp": 32,
            "relative_humidity": 70,
            "motion_sensor": 1,
            "grid_power": 0.8,
            "hour": 14,
            "day_of_week": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_mode"] in {"Eco", "Max Cool", "Dry"}
    assert body["predicted_mode_code"] in {0, 1, 2}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ambient_temp", 14),
        ("ambient_temp", 46),
        ("relative_humidity", -1),
        ("relative_humidity", 101),
        ("motion_sensor", 2),
        ("grid_power", -0.1),
        ("grid_power", 1.1),
        ("hour", -1),
        ("hour", 24),
        ("day_of_week", -1),
        ("day_of_week", 7),
    ],
)
def test_random_forest_rejects_invalid_input(field: str, value: object) -> None:
    payload = {
        "ambient_temp": 32,
        "relative_humidity": 70,
        "motion_sensor": 1,
        "grid_power": 0.8,
        "hour": 14,
        "day_of_week": 3,
    }
    payload[field] = value

    response = client.post("/predict/ac", json=payload)

    assert response.status_code == 422


def test_occupancy_estimation() -> None:
    response = client.post(
        "/occupancy/estimate",
        json={"timestamp": "morning", "motion_detected": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["time_period"] == "morning"
    assert body["likely_occupied"] is True
    assert 0 <= body["occupancy_percentage"] <= 100


def test_valid_nlp_command() -> None:
    response = client.post(
        "/commands/parse",
        json={"command": "turn on the AC"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "power_control"
    assert response.json()["value"] == "on"


def test_unknown_nlp_command_requests_clarification() -> None:
    response = client.post(
        "/commands/parse",
        json={"command": "play music"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "unknown"
    assert body["needs_clarification"] is True


def test_ambiguous_nlp_command_requests_clarification() -> None:
    response = client.post(
        "/commands/parse",
        json={"command": "turn on and turn off the AC"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ambiguous"
    assert body["needs_clarification"] is True


def test_fuzzy_controller() -> None:
    response = client.post(
        "/control/fuzzy",
        json={
            "temperature_c": 32,
            "humidity_pct": 70,
            "occupancy_pct": 90,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["cooling_demand"] <= 100
    assert body["suggested_mode"] in {"Off", "Eco", "Dry", "Max Cool"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature_c", 14),
        ("temperature_c", 46),
        ("humidity_pct", -1),
        ("humidity_pct", 101),
        ("occupancy_pct", -1),
        ("occupancy_pct", 101),
    ],
)
def test_fuzzy_controller_rejects_invalid_input(
    field: str, value: object
) -> None:
    payload = {
        "temperature_c": 32,
        "humidity_pct": 70,
        "occupancy_pct": 90,
    }
    payload[field] = value

    response = client.post("/control/fuzzy", json=payload)

    assert response.status_code == 422


def test_unified_recommendation_preserves_component_outputs() -> None:
    response = client.post(
        "/recommendation",
        json={
            "temperature_c": 32,
            "humidity_pct": 70,
            "occupancy_pct": 90,
            "command": "turn on the AC",
        },
    )

    assert response.status_code == 200
    body = response.json()
    components = {item["component"] for item in body["component_outputs"]}
    assert components == {
        "ac_prediction",
        "occupancy_estimation",
        "fuzzy_logic_controller",
    }
    assert body["overall_recommendation"]["ac_mode"] == "Dry"
    assert body["overall_recommendation"]["fuzzy_suggested_mode"] == "Dry"


def test_unified_recommendation_can_show_component_disagreement() -> None:
    response = client.post(
        "/recommendation",
        json={
            "temperature_c": 27,
            "humidity_pct": 70,
            "occupancy_pct": 90,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert (
        body["overall_recommendation"]["ac_mode"]
        != body["overall_recommendation"]["fuzzy_suggested_mode"]
    )


@pytest.mark.parametrize(
    "path",
    ["/predict/ac", "/occupancy/estimate", "/commands/parse", "/control/fuzzy"],
)
def test_missing_required_fields_returns_422(path: str) -> None:
    response = client.post(path, json={})

    assert response.status_code == 422


def test_invalid_data_types_return_422() -> None:
    response = client.post(
        "/predict/ac",
        json={
            "ambient_temp": "hot",
            "relative_humidity": 70,
            "motion_sensor": 1,
            "grid_power": 0.8,
            "hour": 14,
            "day_of_week": 3,
        },
    )

    assert response.status_code == 422


def test_missing_model_artifact_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_model() -> object:
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(main, "get_model", missing_model)
    response = client.post(
        "/predict/ac",
        json={
            "ambient_temp": 32,
            "relative_humidity": 70,
            "motion_sensor": 1,
            "grid_power": 0.8,
            "hour": 14,
            "day_of_week": 3,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Model file not found"


def test_model_prediction_failure_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenModel:
        def predict(self, features: object) -> object:
            raise RuntimeError("prediction failed")

    monkeypatch.setattr(main, "_model", BrokenModel())
    monkeypatch.setattr(main, "_model_loaded", True)
    response = client.post(
        "/predict/ac",
        json={
            "ambient_temp": 32,
            "relative_humidity": 70,
            "motion_sensor": 1,
            "grid_power": 0.8,
            "hour": 14,
            "day_of_week": 3,
        },
    )

    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]["error"]


def test_fuzzy_import_failure_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_fuzzy_dependency() -> None:
        raise ImportError("fuzzy dependency missing")

    monkeypatch.setattr(main, "get_fuzzy_controller", missing_fuzzy_dependency)
    response = client.post(
        "/control/fuzzy",
        json={
            "temperature_c": 32,
            "humidity_pct": 70,
            "occupancy_pct": 90,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "fuzzy dependency missing"
