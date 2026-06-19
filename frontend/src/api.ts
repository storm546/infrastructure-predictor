// API client for the infrastructure predictor backend
const API_BASE = "/api";

export interface PredictRequest {
  repair_type: string;
  object_type: string;
  season: string;
  authority_type: string;
  value_bgn: number;
  num_offers: number;
  eu_financed: number;
  month: number;
  day_of_week: number;
  has_annex: number;
  annex_extension_days: number;
}

export interface PredictResponse {
  predicted_days: number;
  delay_risk: "low" | "medium" | "high";
  confidence: string;
  features_used: Record<string, unknown>;
  model_info: {
    test_mae: number;
    test_r2: number;
    trained_at: string;
  };
}

export interface FeatureMeta {
  repair_types: string[];
  contractors: string[];
  object_types: string[];
  seasons: string[];
  authority_types: string[];
  mean_days: number;
  num_samples: number;
}

export async function fetchMeta(): Promise<FeatureMeta> {
  const res = await fetch(`${API_BASE}/meta`);
  if (!res.ok) throw new Error("Failed to fetch metadata");
  return res.json();
}

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}
