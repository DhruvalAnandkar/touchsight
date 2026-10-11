export type StageStatus = "pending" | "running" | "complete" | "skipped" | "failed";
export type JobStatus = "queued" | "running" | "complete" | "failed";

export interface StageProgress {
  name: string;
  status: StageStatus;
  elapsed_ms: number | null;
  message: string | null;
  provider: string | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  image_hash: string;
  cached: boolean;
  stages: StageProgress[];
  artifacts: Record<string, string>;
  error: string | null;
  region_count: number | null;
  source: string | null;
  created_at: string;
}

export interface Sample {
  id: string;
  title: string;
  description: string;
  hash: string;
  image_url: string;
}

export interface RegionMap {
  subject?: string;
  summary?: string;
  regions?: Array<{
    id: string;
    name: string;
    description: string;
    pattern?: string;
    importance?: number;
    polygon?: number[][];
  }>;
}
