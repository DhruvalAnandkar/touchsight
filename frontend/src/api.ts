import { answerQuestion } from "./features/explore/answers";
import type { Job, RegionMap, Sample } from "./types";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string" && body.detail) return body.detail;
  } catch {
    /* response was not JSON */
  }
  return "The server could not read that image.";
}

export async function askDiagram(question: string, map: RegionMap): Promise<string> {
  const local = answerQuestion(question, map);
  const parts = (map.regions ?? [])
    .map((region) => ({ name: region.name, description: region.description }))
    .filter((region) => region.name.trim())
    .slice(0, 12);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question.slice(0, 400),
        subject: map.subject ?? "",
        summary: map.summary ?? "",
        parts,
      }),
    });
    if (!response.ok) return local;
    const body = (await response.json()) as { answer?: string; source?: string };
    if (body.source === "model" && typeof body.answer === "string" && body.answer.trim()) {
      return body.answer.trim();
    }
  } catch {
    /* the local answer still speaks */
  }
  return local;
}

export async function fetchSamples(): Promise<Sample[]> {
  const response = await fetch("/api/samples");
  if (!response.ok) return [];
  return (await response.json()) as Sample[];
}

export async function convertFile(file: File, onJob: (job: Job) => void, hint = ""): Promise<Job> {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("hint", hint.trim());
  const created = await fetch("/api/convert", { method: "POST", body });
  if (!created.ok) throw new Error(await errorMessage(created));
  const { job_id } = (await created.json()) as { job_id: string };
  const started = Date.now();
  while (Date.now() - started < 180_000) {
    const response = await fetch(`/api/jobs/${job_id}`);
    if (!response.ok) throw new Error("The job disappeared before it finished.");
    const job = (await response.json()) as Job;
    onJob(job);
    if (job.status === "complete" || job.status === "failed") return job;
    await new Promise((resolve) => window.setTimeout(resolve, 400));
  }
  throw new Error("That image took too long. Try a smaller picture.");
}
