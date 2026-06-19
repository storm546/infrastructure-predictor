import { useState, useEffect } from "react";
import { fetchMeta, predict } from "./api";
import type { PredictRequest, PredictResponse, FeatureMeta } from "./api";
import "./App.css";

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
];

const METHODS = ["open", "collect_offers", "direct", "restricted", "other"];

function App() {
  const [meta, setMeta] = useState<FeatureMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState<PredictRequest>({
    repair_type: "roads_highways",
    method: "open",
    buyer_type: "municipality",
    season: "summer",
    value_bgn: 150000,
    num_offers: 3,
    month: 6,
    postal_region: "",
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
        <h1>Infrastructure Contract Duration Predictor</h1>
        <p className="subtitle">
          XGBoost trained on {meta?.num_samples ?? "—"} real Bulgarian construction
          contracts (CPV&nbsp;45, 2020–2023). Predicts the <strong>contracted</strong>{" "}
          execution period agreed at signing — not actual on-site completion time.
        </p>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="form">
          <div className="grid">
            <label>
              Construction Type
              <select name="repair_type" value={form.repair_type} onChange={handleChange}>
                {meta?.repair_types.map((t) => (
                  <option key={t} value={t}>{t.replace(/_/g, " ")}</option>
                ))}
              </select>
            </label>

            <label>
              Procurement Method
              <select name="method" value={form.method} onChange={handleChange}>
                {(meta?.methods?.length ? meta.methods : METHODS).map((t) => (
                  <option key={t} value={t}>{t.replace(/_/g, " ")}</option>
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
              Buyer Type
              <select name="buyer_type" value={form.buyer_type} onChange={handleChange}>
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
              Number of Bids
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
              Postal Region (first 2 digits)
              <input
                type="text"
                name="postal_region"
                value={form.postal_region}
                onChange={handleChange}
                maxLength={2}
                placeholder="e.g. 55"
              />
            </label>
          </div>

          <button type="submit" disabled={predicting} className="submit-btn">
            {predicting ? "Predicting..." : "Predict Contracted Duration"}
          </button>
        </form>

        {error && <div className="error">Error: {error}</div>}

        {result && (
          <div className="result">
            <div className="result-main">
              <span className="result-label">Predicted Contracted Duration</span>
              <span className="result-days">{result.predicted_days} days</span>
              <span className="result-months">
                (~{(result.predicted_days / 30).toFixed(1)} months)
              </span>
            </div>

            <div className="result-risk">
              <span className="result-label">Relative Duration</span>
              <span
                className="risk-badge"
                style={{ background: riskColors[result.delay_risk] }}
              >
                {result.delay_risk.toUpperCase()}
              </span>
            </div>

            <div className="result-meta">
              <span>Model MAE: ±{result.model_info.test_mae?.toFixed(0)} days</span>
              <span>Baseline MAE: ±{result.model_info.baseline_mae?.toFixed(0)} days</span>
              <span>R²: {result.model_info.test_r2?.toFixed(2)}</span>
              <span>Trained: {result.model_info.trained_at?.slice(0, 10)}</span>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        Data: Open Contracting / DIGIWHIST Bulgaria (opentender.eu) · CC BY-NC-SA 4.0
      </footer>
    </div>
  );
}

export default App;
