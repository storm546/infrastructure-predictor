// API client for the infrastructure duration predictor backend
const API_BASE = "/api";

export interface PredictRequest {
  repair_type: string;
  method: string;
  buyer_type: string;
  season: string;
  value_bgn: number;
  num_offers: number;
  month: number;
  postal_region: string;
}

export interface PredictResponse {
  predicted_days: number;
  delay_risk: "low" | "medium" | "high";
  confidence: string;
  features_used: Record<string, unknown>;
  model_info: {
    target: string;
    target_note: string;
    test_mae: number;
    test_r2: number;
    baseline_mae: number;
    trained_at: string;
  };
}

export interface FeatureMeta {
  repair_types: string[];
  methods: string[];
  object_types: string[];
  seasons: string[];
  authority_types: string[];
  towns: string[];
  mean_days: number;
  num_samples: number;
  target_note: string;
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
