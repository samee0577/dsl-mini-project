import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { estimateOccupancy, getHealth, getRecommendation, getWeather, parseCommand, predictAc, predictFuzzy } from "./api";
import type { AcPredictionResponse, CommandResponse, FuzzyResponse, OccupancyResponse, RecommendationResponse, ScenarioInputs, ScenarioResult, WeatherData } from "./types";

type Mode = "live" | "demo";
type Scenario = { name: string; temperature: number; humidity: number; occupancy: number };
const scenarios: Scenario[] = [
  { name: "Hot Afternoon", temperature: 36, humidity: 58, occupancy: 85 },
  { name: "Humid Evening", temperature: 31, humidity: 86, occupancy: 70 },
  { name: "Home Unoccupied", temperature: 29, humidity: 65, occupancy: 10 },
  { name: "Comfortable Night", temperature: 25, humidity: 55, occupancy: 55 },
];

function Panel({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return <section className="card"><div className="card-heading"><h2>{title}</h2>{action}</div>{children}</section>;
}

function SourceBadge({ children, tone = "neutral" }: { children: ReactNode; tone?: "live" | "demo" | "neutral" }) {
  return <span className={`source-badge ${tone}`}>{children}</span>;
}

function formatValue(value: number | null, suffix = "") {
  return value === null ? "Unavailable" : `${value.toFixed(value % 1 ? 1 : 0)}${suffix}`;
}

const defaultScenario: ScenarioInputs = {
  ambient_temp: 32, relative_humidity: 70, motion_sensor: 1, grid_power: 0.7,
  hour: 15, day_of_week: 4, occupancy_timestamp: "afternoon",
  occupancy_motion_detected: true, occupancy_pct: 75,
};

function ScenarioInputsForm({ value, onChange }: { value: ScenarioInputs; onChange: (value: ScenarioInputs) => void }) {
  const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  return <div className="control-grid">
    <label>Ambient temperature <output>{value.ambient_temp}°C</output><input type="range" min="15" max="45" value={value.ambient_temp} onChange={(e) => onChange({ ...value, ambient_temp: Number(e.target.value) })} /></label>
    <label>Relative humidity <output>{value.relative_humidity}%</output><input type="range" min="0" max="100" value={value.relative_humidity} onChange={(e) => onChange({ ...value, relative_humidity: Number(e.target.value) })} /></label>
    <label>Grid power <output>{value.grid_power.toFixed(2)}</output><input type="range" min="0" max="1" step="0.05" value={value.grid_power} onChange={(e) => onChange({ ...value, grid_power: Number(e.target.value) })} /></label>
    <label>Hour of day <output>{value.hour}:00</output><input type="range" min="0" max="23" value={value.hour} onChange={(e) => onChange({ ...value, hour: Number(e.target.value) })} /></label>
    <label>Motion sensor <output>{value.motion_sensor ? "Detected" : "No motion"}</output><select value={value.motion_sensor} onChange={(e) => onChange({ ...value, motion_sensor: Number(e.target.value) as 0 | 1 })}><option value="1">Motion detected</option><option value="0">No motion</option></select></label>
    <label>Day of week <output>{days[value.day_of_week]}</output><select value={value.day_of_week} onChange={(e) => onChange({ ...value, day_of_week: Number(e.target.value) })}>{days.map((day, index) => <option value={index} key={day}>{day}</option>)}</select></label>
    <label>Bayesian time period<select value={value.occupancy_timestamp} onChange={(e) => onChange({ ...value, occupancy_timestamp: e.target.value })}><option>morning</option><option>afternoon</option><option>evening</option><option>night</option></select></label>
    <label>Bayesian motion evidence<select value={String(value.occupancy_motion_detected)} onChange={(e) => onChange({ ...value, occupancy_motion_detected: e.target.value === "true" })}><option value="true">Motion detected</option><option value="false">No motion</option></select></label>
    <label>Fuzzy occupancy input <output>{value.occupancy_pct}%</output><input type="range" min="0" max="100" value={value.occupancy_pct} onChange={(e) => onChange({ ...value, occupancy_pct: Number(e.target.value) })} /></label>
  </div>;
}

async function runScenario(inputs: ScenarioInputs): Promise<ScenarioResult> {
  const [randomForest, occupancy, fuzzy, recommendation] = await Promise.all([
    predictAc({
      ambient_temp: inputs.ambient_temp, relative_humidity: inputs.relative_humidity,
      motion_sensor: inputs.motion_sensor, grid_power: inputs.grid_power,
      hour: inputs.hour, day_of_week: inputs.day_of_week,
    }),
    estimateOccupancy(inputs.occupancy_timestamp, inputs.occupancy_motion_detected),
    predictFuzzy(inputs.ambient_temp, inputs.relative_humidity, inputs.occupancy_pct),
    getRecommendation(inputs.ambient_temp, inputs.relative_humidity, inputs.occupancy_pct),
  ]);
  return { inputs, randomForest, occupancy, fuzzy, recommendation, ranAt: new Date().toLocaleTimeString() };
}

function ScenarioResultCards({ result }: { result: ScenarioResult }) {
  const threshold = result.recommendation.overall_recommendation.ac_mode;
  const unifiedFuzzy = result.recommendation.overall_recommendation.fuzzy_suggested_mode;
  const thresholdNote = result.inputs.ambient_temp >= 38
    ? "The unified endpoint's temperature rule selects Max Cool at 38°C or above."
    : result.inputs.ambient_temp >= 32
      ? "The unified endpoint's temperature rule selects Dry from 32°C up to, but not including, 38°C."
      : result.inputs.ambient_temp >= 25
        ? "The unified endpoint's temperature rule selects Eco from 25°C up to, but not including, 32°C."
        : "The unified endpoint's temperature rule selects Off below 25°C.";
  return <div className="scenario-output">
    <div className="result-grid">
    <Panel title="Random Forest prediction" action={<SourceBadge>TRAINED ML · API</SourceBadge>}>
      <div className="model-result"><strong>{result.randomForest.predicted_mode}</strong><span>Returned probability: {result.randomForest.probability === null ? "Unavailable" : `${(result.randomForest.probability * 100).toFixed(1)}%`}</span></div>
      <p className="technical-note">The backend returned this class for the six submitted features. No additional feature explanation is claimed.</p>
    </Panel>
    <Panel title="Bayesian occupancy estimate" action={<SourceBadge>PROVISIONAL · API</SourceBadge>}>
      <div className="model-result"><strong>{result.occupancy.occupancy_percentage.toFixed(1)}%</strong><span>{result.occupancy.likely_occupied ? "Likely occupied" : "Likely empty"} · {result.occupancy.time_period}</span></div>
      <p className="technical-note">{result.occupancy.provisional_note}</p>
    </Panel>
    <Panel title="Fuzzy controller" action={<SourceBadge>RULE OUTPUT · API</SourceBadge>}>
      <div className="model-result"><strong>{result.fuzzy.suggested_mode}</strong><span>Cooling demand: {result.fuzzy.cooling_demand.toFixed(1)}/100</span></div>
      <p className="technical-note">{result.fuzzy.model_note}</p>
    </Panel>
    <Panel title="Unified recommendation" action={<SourceBadge>THRESHOLD + COMPONENTS</SourceBadge>}>
      <div className="model-result"><strong>{threshold}</strong><span>Backend fuzzy output: {unifiedFuzzy} · Occupancy input: {result.recommendation.overall_recommendation.occupancy_percentage}%</span></div>
      <p className="technical-note">{result.recommendation.overall_recommendation.ac_disclaimer ?? result.recommendation.disclaimer}</p>
    </Panel>
    </div>
    <Panel title="Why the outputs can disagree" action={<SourceBadge>BACKEND LOGIC</SourceBadge>}>
      <p className="technical-note">{thresholdNote} That threshold result uses only the submitted temperature.</p>
      <p className="technical-note">The Random Forest returned <strong>{result.randomForest.predicted_mode}</strong> from its six-feature input vector. The fuzzy controller returned <strong>{result.fuzzy.suggested_mode}</strong> after combining temperature, humidity, and the submitted occupancy input through its membership functions and rules.</p>
      <p className="technical-note">The Bayesian estimator returned <strong>{result.occupancy.occupancy_percentage.toFixed(1)}%</strong> from the submitted time period and motion evidence. These components have different inputs and purposes, so disagreement is an expected result rather than an error.</p>
    </Panel>
  </div>;
}

function scenarioChanges(first: ScenarioInputs, second: ScenarioInputs) {
  const changes: string[] = [];
  if (first.ambient_temp !== second.ambient_temp) changes.push(`Temperature: ${first.ambient_temp}°C → ${second.ambient_temp}°C`);
  if (first.relative_humidity !== second.relative_humidity) changes.push(`Humidity: ${first.relative_humidity}% → ${second.relative_humidity}%`);
  if (first.motion_sensor !== second.motion_sensor) changes.push(`Random Forest motion: ${first.motion_sensor ? "detected" : "none"} → ${second.motion_sensor ? "detected" : "none"}`);
  if (first.grid_power !== second.grid_power) changes.push(`Grid power: ${first.grid_power.toFixed(2)} → ${second.grid_power.toFixed(2)}`);
  if (first.hour !== second.hour) changes.push(`Hour: ${first.hour}:00 → ${second.hour}:00`);
  if (first.day_of_week !== second.day_of_week) changes.push(`Day: ${first.day_of_week} → ${second.day_of_week}`);
  if (first.occupancy_pct !== second.occupancy_pct) changes.push(`Fuzzy occupancy input: ${first.occupancy_pct}% → ${second.occupancy_pct}%`);
  if (first.occupancy_timestamp !== second.occupancy_timestamp || first.occupancy_motion_detected !== second.occupancy_motion_detected) changes.push("Bayesian time or motion evidence changed");
  return changes;
}

function ScenarioComparison({ first, second }: { first: ScenarioResult; second: ScenarioResult }) {
  const changes = scenarioChanges(first.inputs, second.inputs);
  const rows = [
    ["Random Forest", first.randomForest.predicted_mode, second.randomForest.predicted_mode],
    ["RF returned probability", first.randomForest.probability === null ? "Unavailable" : `${(first.randomForest.probability * 100).toFixed(1)}%`, second.randomForest.probability === null ? "Unavailable" : `${(second.randomForest.probability * 100).toFixed(1)}%`],
    ["Bayesian occupancy", `${first.occupancy.occupancy_percentage.toFixed(1)}%`, `${second.occupancy.occupancy_percentage.toFixed(1)}%`],
    ["Fuzzy suggestion", first.fuzzy.suggested_mode, second.fuzzy.suggested_mode],
    ["Fuzzy cooling demand", `${first.fuzzy.cooling_demand.toFixed(1)}/100`, `${second.fuzzy.cooling_demand.toFixed(1)}/100`],
    ["Threshold recommendation", first.recommendation.overall_recommendation.ac_mode, second.recommendation.overall_recommendation.ac_mode],
  ];
  return <Panel title="Scenario A/B comparison" action={<SourceBadge tone="live">TWO API RUNS</SourceBadge>}>
    <p className="helper-text">This compares returned outputs only. It does not claim energy savings or that one scenario is better.</p>
    <table className="comparison-table"><thead><tr><th>Output</th><th>Scenario A</th><th>Scenario B</th></tr></thead><tbody>{rows.map(([label, a, b]) => <tr key={label}><th>{label}</th><td><strong>{a}</strong></td><td><strong>{b}</strong></td></tr>)}</tbody></table>
    <div><strong className="metric-label">Changed inputs</strong>{changes.length ? <ul className="comparison-changes">{changes.map((change) => <li key={change}>{change}</li>)}</ul> : <p className="technical-note">No input differences.</p>}</div>
  </Panel>;
}

function AIRecommendations() {
  const [scenario, setScenario] = useState<ScenarioInputs>(defaultScenario);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [comparisonScenario, setComparisonScenario] = useState<ScenarioInputs>({ ...defaultScenario, ambient_temp: 38, relative_humidity: 85, motion_sensor: 1, grid_power: 0.9, hour: 18, occupancy_timestamp: "evening", occupancy_pct: 90 });
  const [comparisonResult, setComparisonResult] = useState<ScenarioResult | null>(null);
  const [command, setCommand] = useState("turn on the AC");
  const [commandResult, setCommandResult] = useState<CommandResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const runModels = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError("");
    try { setResult(await runScenario(scenario)); }
    catch { setError("The scenario could not be completed. Check that the backend and model files are available."); }
    setBusy(false);
  };

  const runComparison = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError("");
    try { setComparisonResult(await runScenario(comparisonScenario)); }
    catch { setError("Scenario B could not be completed. Check that the backend and model files are available."); }
    setBusy(false);
  };

  const runNlp = async (event: FormEvent) => {
    event.preventDefault(); setCommandResult(null); setError("");
    try { setCommandResult(await parseCommand(command)); }
    catch { setError("The NLP parser could not be reached."); }
  };

  return <div className="model-workspace">
    <div className="workspace-intro"><div><p className="overline">Interactive model laboratory</p><h2>Explore the HEMS intelligence</h2><p>Change inputs, run the backend components, and compare their outputs. These results are recommendations and estimates; they do not control physical appliances.</p></div><SourceBadge tone="live">FASTAPI MODELS</SourceBadge></div>
    <form onSubmit={runModels}>
      <Panel title="Interactive scenario inputs" action={<SourceBadge tone="demo">SYNTHETIC INPUTS</SourceBadge>}>
        <p className="helper-text">These values are submitted to the backend when you run the scenario. They are not live household sensor readings.</p>
        <ScenarioInputsForm value={scenario} onChange={setScenario} />
      </Panel>
      <button className="run-models" type="submit" disabled={busy}>{busy ? "Running scenario…" : "Run scenario"}</button>
    </form>
    {result && <><div className="run-summary"><SourceBadge tone="live">LIVE API RESULTS</SourceBadge><span>Completed at {result.ranAt}. Four endpoint responses are shown separately below.</span></div><ScenarioResultCards result={result} /></>}
    <Panel title="Scenario B" action={<SourceBadge tone="demo">SECOND SYNTHETIC INPUT SET</SourceBadge>}>
      <p className="helper-text">Use the same controls to create a second API-backed run, then compare it with Scenario A.</p>
      <form onSubmit={runComparison}><ScenarioInputsForm value={comparisonScenario} onChange={setComparisonScenario} /><button className="run-models" type="submit" disabled={busy}>{busy ? "Running scenario…" : "Run Scenario B"}</button></form>
    </Panel>
    {comparisonResult && <><div className="run-summary"><SourceBadge tone="live">SCENARIO B API RESULTS</SourceBadge><span>Completed at {comparisonResult.ranAt}.</span></div><ScenarioResultCards result={comparisonResult} /></>}
    {result && comparisonResult && <ScenarioComparison first={result} second={comparisonResult} />}
    <Panel title="NLP command parser" action={<SourceBadge>NLP · rule-based</SourceBadge>}>
      <p className="helper-text">Try natural language such as “turn on the AC” or “set the AC to dry”. This parses intent and asks for clarification when uncertain; it does not operate hardware.</p>
      <form className="command-form" onSubmit={runNlp}><input value={command} onChange={(e) => setCommand(e.target.value)} /><button type="submit">Parse command</button></form>
      {commandResult && <div className="nlp-result"><strong>{commandResult.intent}</strong><span>{commandResult.message}</span><span>Appliance: {commandResult.appliance ?? "Not identified"} · Action: {commandResult.action ?? "Not identified"} · Confidence: {(commandResult.confidence * 100).toFixed(0)}%</span><SourceBadge tone={commandResult.needs_clarification ? "demo" : "live"}>{commandResult.needs_clarification ? "CLARIFICATION NEEDED" : "PARSED"}</SourceBadge></div>}
    </Panel>
    {error && <div className="state-message error">{error}</div>}
  </div>;
}

function UtilityView({ tab }: { tab: string }) {
  const content: Record<string, { title: string; text: string; items: string[] }> = {
    Devices: { title: "Connected devices", text: "This project currently has recommendation inputs, not hardware telemetry or control connections.", items: ["Air Conditioner · Recommendation available · Hardware state not connected", "Refrigerator · No sensor integration configured", "Ceiling Fan · No sensor integration configured"] },
    Energy: { title: "Energy insights", text: "Measured energy telemetry is not connected yet. Grid power is currently an input feature for the Random Forest experiment, not a live energy meter.", items: ["Live consumption: Not available", "Energy savings: Not calculated", "Next step: connect a smart meter or load sensor"] },
    "System Status": { title: "System status", text: "Use this page to distinguish service availability from model availability.", items: ["FastAPI backend · Check the LIVE API badge", "Random Forest file · Used by POST /predict/ac", "Open-Meteo · Outdoor forecast provider", "NLP parser · Rule-based and available through POST /commands/parse"] },
    Settings: { title: "Dashboard settings", text: "Model inputs are intentionally changed from the AI Recommendations tab so every experiment is visible and reproducible.", items: ["API base URL: VITE_HEMS_API_URL", "Weather refresh: Every 15 minutes", "Physical control: Disabled until device integrations are added"] },
  };
  const view = content[tab];
  return <Panel title={view.title}><p className="utility-intro">{view.text}</p><div className="utility-list">{view.items.map((item) => <div key={item}><i className="activity-dot good" />{item}</div>)}</div></Panel>;
}

export function App() {
  const [mode, setMode] = useState<Mode>("live");
  const [scenario, setScenario] = useState(scenarios[0]);
  const [online, setOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [weather, setWeather] = useState<(WeatherData & { description: string }) | null>(null);
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [weatherError, setWeatherError] = useState("");
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [activity, setActivity] = useState(["Dashboard ready"]);
  const [command, setCommand] = useState("");
  const [commandResult, setCommandResult] = useState<CommandResponse | null>(null);
  const [commandError, setCommandError] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [activeNav, setActiveNav] = useState("Dashboard");

  const indoor = mode === "demo"
    ? scenario
    : { temperature: 32, humidity: 70, occupancy: 90 };
  const addActivity = useCallback((message: string) => setActivity((items) => [message, ...items].slice(0, 5)), []);

  const refreshWeather = useCallback(async () => {
    setWeatherLoading(true);
    setWeatherError("");
    try { setWeather(await getWeather()); addActivity("Outdoor weather updated"); }
    catch { setWeatherError("Weather service is unavailable. Indoor HEMS data is unaffected."); }
    finally { setWeatherLoading(false); }
  }, [addActivity]);

  const refreshHems = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      await getHealth();
      const result = await getRecommendation(indoor.temperature, indoor.humidity, indoor.occupancy ?? 90);
      setRecommendation(result);
      setOnline(true);
      addActivity("HEMS recommendation updated");
    } catch { setOnline(false); addActivity("HEMS backend unavailable"); }
    finally { setLoading(false); }
  }, [addActivity, indoor.humidity, indoor.occupancy, indoor.temperature]);

  useEffect(() => { void refreshWeather(); }, [refreshWeather]);
  useEffect(() => { void refreshHems(); }, [refreshHems]);
  useEffect(() => {
    const timer = window.setInterval(() => { void refreshWeather(); }, 15 * 60 * 1000);
    return () => window.clearInterval(timer);
  }, [refreshWeather]);

  const fuzzy = recommendation?.component_outputs.find((item) => item.component === "fuzzy_logic_controller")?.output;
  const acMode = recommendation?.overall_recommendation.ac_mode ?? "Unavailable";
  const fuzzyMode = fuzzy?.suggested_mode ?? "Unavailable";
  const explanation = useMemo(() => {
    if (!recommendation) return "Run a refresh when the HEMS backend is available.";
    if (acMode !== fuzzyMode) return `The temperature-based recommendation is ${acMode}, while the fuzzy controller suggests ${fuzzyMode}. HEMS keeps both outputs visible instead of hiding the disagreement.`;
    return `Both recommendation paths suggest ${acMode} for the available home conditions. This is decision support only; no physical appliance command was sent.`;
  }, [acMode, fuzzyMode, recommendation]);

  const submitCommand = async (event: FormEvent) => {
    event.preventDefault();
    if (!command.trim()) return;
    setCommandError(""); setCommandResult(null);
    try { const result = await parseCommand(command.trim()); setCommandResult(result); addActivity(`Command parsed · ${result.intent}`); }
    catch { setCommandError("Unable to parse. Check that the backend is running."); }
  };

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">H</span><div><strong>HEMS</strong><small>Cognitive Smart Home</small></div></div>
      <nav aria-label="Main navigation">{["Dashboard", "Devices", "Energy", "AI Recommendations", "System Status", "Settings"].map((item) => <button className={activeNav === item ? "nav-item active" : "nav-item"} key={item} onClick={() => setActiveNav(item)}><span>{["⌂", "◈", "↗", "✦", "●", "⚙"][["Dashboard", "Devices", "Energy", "AI Recommendations", "System Status", "Settings"].indexOf(item)]}</span>{item}</button>)}</nav>
      <div className="sidebar-footer"><SourceBadge tone={online ? "live" : "neutral"}>{online ? "LIVE API" : "API OFFLINE"}</SourceBadge><small>Backend status</small></div>
    </aside>
    <div className="content">
      <header className="page-header"><div><p className="overline">{new Intl.DateTimeFormat([], { weekday: "long", month: "long", day: "numeric" }).format(new Date())}</p><h1>{new Date().getHours() < 12 ? "Good morning" : new Date().getHours() < 18 ? "Good afternoon" : "Good evening"}, Daksh</h1><p className="subtitle">Here is what's happening in your smart home.</p></div><div className="header-actions"><SourceBadge tone={mode === "live" ? "live" : "demo"}>{mode === "live" ? "LIVE API MODE" : "DEMO MODE"}</SourceBadge><button className="refresh-button" onClick={() => { void refreshWeather(); void refreshHems(); }} disabled={loading || weatherLoading}>{loading || weatherLoading ? "Refreshing…" : "Refresh data"}</button></div></header>
      <div className="mode-switch" role="group" aria-label="Data mode"><button className={mode === "live" ? "selected" : ""} onClick={() => setMode("live")}>Live API mode</button><button className={mode === "demo" ? "selected" : ""} onClick={() => setMode("demo")}>Demo mode</button>{mode === "demo" && <select aria-label="Demo scenario" value={scenario.name} onChange={(event) => setScenario(scenarios.find((item) => item.name === event.target.value) ?? scenarios[0])}>{scenarios.map((item) => <option key={item.name}>{item.name}</option>)}</select>}</div>
      {activeNav === "AI Recommendations" ? <AIRecommendations /> : activeNav !== "Dashboard" ? <UtilityView tab={activeNav} /> : <div className="dashboard-columns">
        <div className="main-column">
          <Panel title="Outdoor weather" action={<SourceBadge>Open-Meteo estimate</SourceBadge>}><div className="location">Shree L. R. Tiwari College of Engineering, Mira Road</div>{weatherLoading ? <div className="state-message">Loading local forecast…</div> : weatherError ? <div className="state-message error">{weatherError}</div> : weather && <div className="weather-grid"><div className="weather-temperature"><strong>{formatValue(weather.temperature, "°C")}</strong><span>{weather.description}</span><small>Feels like {formatValue(weather.apparentTemperature, "°C")}</small></div><div><span className="metric-label">Humidity</span><strong>{formatValue(weather.humidity, "%")}</strong></div><div><span className="metric-label">Wind</span><strong>{formatValue(weather.windSpeed, " km/h")}</strong></div><div><span className="metric-label">Today</span><strong>{formatValue(weather.high, "°")} / {formatValue(weather.low, "°")}</strong></div></div>}<p className="data-note">Provider forecast for this location, not an indoor sensor reading. {weather && `Last updated ${weather.updatedAt}.`}</p></Panel>
          <Panel title="Home overview" action={<SourceBadge tone={mode === "demo" ? "demo" : "neutral"}>{mode === "demo" ? "Synthetic indoor values" : "Backend + inputs"}</SourceBadge>}><div className="overview-grid"><div><span className="metric-label">Indoor temperature</span><strong>{formatValue(indoor.temperature, "°C")}</strong><small>{mode === "demo" ? "Demo scenario" : "Configured input"}</small></div><div><span className="metric-label">Indoor humidity</span><strong>{formatValue(indoor.humidity, "%")}</strong><small>{mode === "demo" ? "Demo scenario" : "Configured input"}</small></div><div><span className="metric-label">Occupancy estimate</span><strong>{formatValue(indoor.occupancy, "%")}</strong><small>Model estimate</small></div><div><span className="metric-label">AC recommendation</span><strong>{acMode}</strong><small>Recommendation only</small></div></div></Panel>
          <Panel title="Appliances" action={<span className="muted">Software integration status</span>}><p className="helper-text">“Not connected” means the dashboard has no sensor, smart-plug, or control API for that appliance. It is not a prediction failure.</p><div className="appliance-grid">{[["⌁", "Air Conditioner", acMode, "Recommendation only · no hardware command"], ["▥", "Refrigerator", "Not connected", "Add refrigerator telemetry or a smart-plug API to show its live state"], ["✣", "Ceiling Fan", "Not connected", "Add fan telemetry or a control API to show its live state"]].map(([icon, name, value, note]) => <article className="appliance-card" key={name}><span className="appliance-icon">{icon}</span><div><h3>{name}</h3><p>{note}</p></div><strong className={value === "Not connected" ? "unavailable" : ""}>{value}</strong></article>)}</div></Panel>
          <Panel title="AI command center" action={<SourceBadge>NLP parser</SourceBadge>}><form className="command-form" onSubmit={submitCommand}><input value={command} onChange={(event) => setCommand(event.target.value)} placeholder="Try “turn on the AC”" /><button type="submit">Parse command</button></form><div className="command-result" role="status">{commandError || (commandResult ? <><strong>{commandResult.intent} · {commandResult.message}</strong><span>Confidence {(commandResult.confidence * 100).toFixed(0)}% · {commandResult.needs_clarification ? "Clarification needed" : "Recommendation only"}</span></> : "Commands are parsed for demonstration; they do not control physical devices.")}</div></Panel>
        </div>
        <aside className="intelligence-column"><Panel title="HEMS intelligence" action={<SourceBadge tone="live">BACKEND</SourceBadge>}><div className="recommendation-hero"><span>Recommended AC mode</span><strong>{fuzzyMode}</strong><p>{explanation}</p></div><div className="intelligence-row"><span>Occupancy estimate</span><strong>{formatValue(indoor.occupancy, "%")}</strong></div><div className="intelligence-row"><span>Temperature-based model</span><strong>{acMode}</strong></div><div className="intelligence-row"><span>Cooling demand</span><strong>{fuzzy?.cooling_demand === undefined ? "Unavailable" : `${fuzzy.cooling_demand}/100`}</strong></div><button className="details-button" onClick={() => setDetailsOpen(!detailsOpen)}>{detailsOpen ? "Hide technical details" : "Show technical details"} <span>{detailsOpen ? "−" : "+"}</span></button>{detailsOpen && <div className="technical-details"><p>Outputs remain distinct: the temperature-based mode is not the fuzzy recommendation, and the occupancy estimate is provisional.</p><p>Disclaimer: {recommendation?.disclaimer ?? "Unavailable until the backend responds."}</p></div>}</Panel><Panel title="System activity"><div className="activity-list">{activity.map((item, index) => <div className="activity-item" key={`${item}-${index}`}><i className={index === 0 ? "activity-dot good" : "activity-dot"} />{item}<small>Just now</small></div>)}</div></Panel></aside>
      </div>}
    </div>
  </main>;
}
