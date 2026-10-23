import type {
  AcPredictionResponse, CommandResponse, FuzzyResponse, OccupancyResponse,
  RecommendationResponse, WeatherData,
} from "./types";

const API_BASE = import.meta.env.VITE_HEMS_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

export const getHealth = () => request<{ status: string }>("/health");
export const getRecommendation = (temperature: number, humidity: number, occupancy: number) =>
  request<RecommendationResponse>("/recommendation", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ temperature_c: temperature, humidity_pct: humidity, occupancy_pct: occupancy }),
  });
export const parseCommand = (command: string) => request<CommandResponse>("/commands/parse", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }),
});
export const predictAc = (input: {
  ambient_temp: number; relative_humidity: number; motion_sensor: number;
  grid_power: number; hour: number; day_of_week: number;
}) => request<AcPredictionResponse>("/predict/ac", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input),
});
export const estimateOccupancy = (timestamp: string, motion_detected: boolean) =>
  request<OccupancyResponse>("/occupancy/estimate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ timestamp, motion_detected }),
  });
export const predictFuzzy = (temperature_c: number, humidity_pct: number, occupancy_pct: number) =>
  request<FuzzyResponse>("/control/fuzzy", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ temperature_c, humidity_pct, occupancy_pct }),
  });

const weatherDescription = (code: number | null) => {
  if (code === null) return "Condition unavailable";
  if (code === 0) return "Clear sky";
  if ([1, 2, 3].includes(code)) return "Partly cloudy";
  if ([45, 48].includes(code)) return "Foggy";
  if ([51, 53, 55, 56, 57].includes(code)) return "Drizzle";
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return "Rain";
  if ([71, 73, 75, 77, 85, 86].includes(code)) return "Snow";
  if ([95, 96, 99].includes(code)) return "Thunderstorm";
  return "Condition unavailable";
};

export async function getWeather(): Promise<WeatherData & { description: string }> {
  const params = new URLSearchParams({
    latitude: "19.3002",
    longitude: "72.8762",
    current: "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
    daily: "temperature_2m_max,temperature_2m_min",
    timezone: "Asia/Kolkata",
    forecast_days: "1",
  });
  const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
  if (!response.ok) throw new Error(`Weather service returned ${response.status}`);
  const data = await response.json() as {
    current?: { temperature_2m?: number; apparent_temperature?: number; relative_humidity_2m?: number; weather_code?: number; wind_speed_10m?: number };
    daily?: { temperature_2m_max?: number[]; temperature_2m_min?: number[] };
  };
  const code = data.current?.weather_code ?? null;
  return {
    temperature: data.current?.temperature_2m ?? null,
    apparentTemperature: data.current?.apparent_temperature ?? null,
    humidity: data.current?.relative_humidity_2m ?? null,
    windSpeed: data.current?.wind_speed_10m ?? null,
    weatherCode: code,
    high: data.daily?.temperature_2m_max?.[0] ?? null,
    low: data.daily?.temperature_2m_min?.[0] ?? null,
    updatedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    description: weatherDescription(code),
  };
}
