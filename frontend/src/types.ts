export type SelectedObject = { name: string; detail: string };
export type RecommendationResponse = {
  component_outputs: Array<{ component: string; output?: {
    suggested_mode?: string; cooling_demand?: number; occupancy_percentage?: number;
  }}>;
  overall_recommendation: {
    ac_mode: string; fuzzy_suggested_mode: string; occupancy_percentage?: number;
    ac_disclaimer?: string; occupancy_disclaimer?: string; fuzzy_disclaimer?: string;
  };
  disclaimer: string;
};
export type CommandResponse = {
  command?: string; intent: string; appliance?: string | null; action?: string | null;
  value?: string | Record<string, unknown> | null; message: string; confidence: number;
  needs_clarification: boolean;
};

export type AcPredictionResponse = {
  predicted_mode: string;
  predicted_mode_code: number;
  probability: number | null;
  model_input_feature_order: string[];
  model_limitations: string;
};

export type OccupancyResponse = {
  time_period: string;
  motion_detected: boolean;
  occupancy_probability: number;
  occupancy_percentage: number;
  likely_occupied: boolean;
  provisional_note: string;
};

export type FuzzyResponse = {
  temperature_c: number;
  humidity_pct: number;
  occupancy_pct: number;
  cooling_demand: number;
  suggested_mode: string;
  model_note: string;
};

export type ScenarioInputs = {
  ambient_temp: number;
  relative_humidity: number;
  motion_sensor: 0 | 1;
  grid_power: number;
  hour: number;
  day_of_week: number;
  occupancy_timestamp: string;
  occupancy_motion_detected: boolean;
  occupancy_pct: number;
};

export type ScenarioResult = {
  inputs: ScenarioInputs;
  randomForest: AcPredictionResponse;
  occupancy: OccupancyResponse;
  fuzzy: FuzzyResponse;
  recommendation: RecommendationResponse;
  ranAt: string;
};

export type WeatherData = {
  temperature: number | null;
  apparentTemperature: number | null;
  humidity: number | null;
  windSpeed: number | null;
  weatherCode: number | null;
  high: number | null;
  low: number | null;
  updatedAt: string;
};
