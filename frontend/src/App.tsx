import { useState, useEffect } from "react";
import { fetchMeta, predict } from "./api";
import type { PredictRequest, PredictResponse, FeatureMeta } from "./api";
import "./App.css";

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
];

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

function App() {
  const [meta, setMeta] = useState<FeatureMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState<PredictRequest>({
    repair_type: "road_repair",
    object_type: "строителство",
    season: "summer",
    authority_type: "municipality",
    value_bgn: 150000,
    num_offers: 3,
    eu_financed: 0,
    month: 6,
    day_of_week: 2,
    has_annex: 0,
    annex_extension_days: 0,
  });

  useEffect(() => {
    fetchMeta()
      .then((m) => {
        setMeta(m);
        if (m.repair_types.length > 0) {
          setForm((f) => ({ ...f, repair_type: m.repair_types[0] }));
        }
      })
      .catch((e) => setError("Failed to load metadata: " + e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    setError(null);
    setResult(null);
    try {
      const res = await predict(form);
      setResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setPredicting(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    const val =
      type === "number" || type === "range" ? Number(value) : value;
    setForm((f) => ({ ...f, [name]: val }));
  };

  if (loading) return <div className="loading">Loading feature metadata...</div>;

  const riskColors: Record<string, string> = {
    low: "#22c55e",
    medium: "#f59e0b",
    high: "#ef4444",
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Infrastructure Repair Predictor</h1>
        <p className="subtitle">
          XGBoost model trained on {meta?.num_samples ?? "—"} real Bulgarian public procurement records
        </p>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="form">
          <div className="grid">
            <label>
              Repair Type
              <select name="repair_type" value={form.repair_type} onChange={handleChange}>
                {meta?.repair_types.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>

            <label>
              Object Type
              <select name="object_type" value={form.object_type} onChange={handleChange}>
                {meta?.object_types.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>

            <label>
              Season
              <select name="season" value={form.season} onChange={handleChange}>
                {meta?.seasons.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>

            <label>
              Authority Type
              <select name="authority_type" value={form.authority_type} onChange={handleChange}>
                {meta?.authority_types.map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </label>

            <label>
              Contract Value (BGN)
              <input
                type="number"
                name="value_bgn"
                value={form.value_bgn}
                onChange={handleChange}
                min={0}
                step={1000}
              />
            </label>

            <label>
              Number of Offers
              <input
                type="number"
                name="num_offers"
                value={form.num_offers}
                onChange={handleChange}
                min={0}
                max={50}
              />
            </label>

            <label>
              Contract Month
              <select name="month" value={form.month} onChange={handleChange}>
                {MONTHS.map((m, i) => (
                  <option key={i} value={i + 1}>{m}</option>
                ))}
              </select>
            </label>

            <label>
              Day of Week
              <select name="day_of_week" value={form.day_of_week} onChange={handleChange}>
                {DAYS.map((d, i) => (
                  <option key={i} value={i}>{d}</option>
                ))}
              </select>
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="eu_financed"
                checked={form.eu_financed === 1}
                onChange={(e) =>
                  setForm((f) => ({ ...f, eu_financed: e.target.checked ? 1 : 0 }))
                }
              />
              EU Financed
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="has_annex"
                checked={form.has_annex === 1}
                onChange={(e) =>
                  setForm((f) => ({ ...f, has_annex: e.target.checked ? 1 : 0 }))
                }
              />
              Has Annexes
            </label>
          </div>

          <button type="submit" disabled={predicting} className="submit-btn">
            {predicting ? "Predicting..." : "Predict Repair Duration"}
          </button>
        </form>

        {error && <div className="error">Error: {error}</div>}

        {result && (
          <div className="result">
            <div className="result-main">
              <span className="result-label">Predicted Duration</span>
              <span className="result-days">{result.predicted_days} days</span>
              <span className="result-months">
                (~{(result.predicted_days / 30).toFixed(1)} months)
              </span>
            </div>

            <div className="result-risk">
              <span className="result-label">Delay Risk</span>
              <span
                className="risk-badge"
                style={{ background: riskColors[result.delay_risk] }}
              >
                {result.delay_risk.toUpperCase()}
              </span>
            </div>

            <div className="result-meta">
              <span>Model MAE: ±{result.model_info.test_mae?.toFixed(0)} days</span>
              <span>Trained: {result.model_info.trained_at?.slice(0, 10)}</span>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        Data source: Bulgarian Public Procurement Agency via data.egov.bg
      </footer>
    </div>
  );
}

export default App;
