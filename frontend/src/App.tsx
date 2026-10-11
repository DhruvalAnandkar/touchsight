import { useEffect, useId, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { convertFile, fetchSamples } from "./api";
import { ExploreView } from "./features/explore/ExploreView";
import { PipelineView } from "./features/pipeline/PipelineView";
import { ReliefView } from "./features/relief/ReliefView";
import { PageCapture } from "./features/upload/PageCapture";
import { LandingPage } from "./pages/Landing";
import type { Job, RegionMap, Sample } from "./types";
import { speak } from "./voice/speak";
import { StartGate } from "./voice/StartGate";
import { VoiceGuideToggle } from "./voice/VoiceGuideToggle";

const TABS = ["Tactile", "Understanding", "Explore", "Relief"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Tactile");
  const [showPrint, setShowPrint] = useState(true);
  const [hint, setHint] = useState("");
  const [sheetMarkup, setSheetMarkup] = useState<string | null>(null);
  const [regionMap, setRegionMap] = useState<RegionMap | null>(null);
  const inputId = useId();
  const hintId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    void fetchSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function run(file: File) {
    if (busy) return;
    setBusy(true);
    setError(null);
    setJob(null);
    setFileName(file.name);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(file);
    });
    try {
      const finished = await convertFile(file, setJob, hint);
      if (finished.status === "failed") {
        setError(finished.error ?? "Could not build a tactile outline from that image.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach TouchSight.");
    } finally {
      setBusy(false);
    }
  }

  async function useSample(sample: Sample) {
    const response = await fetch(sample.image_url);
    if (!response.ok) {
      setError("That sample is missing from the demo cache.");
      return;
    }
    const blob = await response.blob();
    await run(new File([blob], `${sample.id}.png`, { type: "image/png" }));
  }

  const statusText = describeStatus(busy, job, error);
  const svgUrl = job?.status === "complete" ? job.artifacts.tactile_svg ?? job.artifacts.outline_svg : undefined;
  const pdfUrl = job?.status === "complete" ? job.artifacts.tactile_pdf : undefined;
  const mapUrl = job?.status === "complete" ? job.artifacts.region_map : undefined;
  const stlUrl = job?.status === "complete" ? job.artifacts.relief_stl ?? null : null;
  const heightmapUrl = job?.status === "complete" ? job.artifacts.relief_height ?? null : null;
  const regionIdUrl = job?.status === "complete" ? job.artifacts.relief_regions ?? null : null;

  useEffect(() => {
    if (!svgUrl) {
      setSheetMarkup(null);
      return;
    }
    let cancel = false;
    void fetch(svgUrl)
      .then((response) => (response.ok ? response.text() : null))
      .then((markup) => {
        if (!cancel) setSheetMarkup(markup);
      })
      .catch(() => {
        if (!cancel) setSheetMarkup(null);
      });
    return () => {
      cancel = true;
    };
  }, [svgUrl]);

  useEffect(() => {
    if (!mapUrl) {
      setRegionMap(null);
      return;
    }
    let cancel = false;
    void fetch(mapUrl)
      .then((response) => (response.ok ? response.json() : null))
      .then((body: RegionMap | null) => {
        if (!cancel) setRegionMap(body);
      })
      .catch(() => {
        if (!cancel) setRegionMap(null);
      });
    return () => {
      cancel = true;
    };
  }, [mapUrl]);

  useEffect(() => {
    if (job?.status !== "complete" || !regionMap) return;
    const subject = regionMap.subject?.trim() || "Diagram";
    const count = regionMap.regions?.length ?? job.region_count ?? 0;
    speak(
      `${subject}. ${count} parts. Open Explore to feel your way around, or 3D to hold it in your hands.`,
      { priority: true },
    );
  }, [job?.id, job?.status, job?.region_count, regionMap]);

  return (
    <StartGate>
      <a
        href="#converter"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-10 focus:bg-gold focus:px-4 focus:py-2 focus:text-gold-deep"
      >
        Skip to the converter
      </a>
      <LandingPage
        onTry={() => document.getElementById("converter")?.scrollIntoView({ behavior: "smooth" })}
        onPhoto={() => inputRef.current?.click()}
        onHow={() => document.getElementById("ts-how-title")?.scrollIntoView({ behavior: "smooth" })}
      />

      <main id="converter" className="mx-auto grid max-w-6xl gap-8 px-6 py-10 lg:grid-cols-[minmax(0,22rem)_1fr]">
        <section aria-labelledby="upload-heading" className="flex w-full min-w-0 flex-col gap-6">
          <h2 id="upload-heading" className="font-serif text-2xl">
            Start with an image
          </h2>
          <VoiceGuideToggle />
          <PageCapture
            inputId={inputId}
            inputRef={inputRef}
            fileName={fileName}
            onFile={(file) => void run(file)}
          />
          <label htmlFor={hintId} className="flex flex-col gap-2">
            <span className="font-bold text-paper">What is this diagram?</span>
            <input
              id={hintId}
              type="text"
              value={hint}
              placeholder="e.g. human heart, plant cell, map of Ohio"
              onChange={(event) => setHint(event.target.value)}
              className="rounded-xl bg-desk-2 px-4 py-3 text-paper placeholder:text-mist"
            />
          </label>

          {samples.length > 0 ? (
            <div className="flex flex-col gap-5">
              <SampleGrid heading="Or try a sample" samples={samples.slice(0, 4)} busy={busy} onPick={useSample} />
              {samples.length > 4 ? (
                <SampleGrid heading="More diagrams" samples={samples.slice(4)} busy={busy} onPick={useSample} />
              ) : null}
            </div>
          ) : null}

          <p role="status" aria-live="polite" className="min-h-8 text-mist">
            {statusText}
          </p>
          {error ? (
            <p role="alert" className="text-danger">
              {error}
            </p>
          ) : null}
        </section>

        <section aria-labelledby="result-heading" aria-busy={busy} className="min-h-[28rem]">
          <h2 id="result-heading" className="font-serif text-2xl">
            Result
          </h2>
          {job ? (
            <PipelineView
              stages={job.stages}
              active={job.status === "running" || job.status === "complete"}
              thumbs={stageThumbs(previewUrl, job)}
            />
          ) : null}
          <div role="tablist" aria-label="Result views" className="mt-4 flex flex-wrap gap-2">
            {TABS.map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={tab === name}
                onClick={() => setTab(name)}
                className={
                  tab === name
                    ? "min-h-11 rounded-full bg-gold px-4 font-bold text-gold-deep"
                    : "min-h-11 rounded-full bg-desk-2 px-4 font-bold text-paper"
                }
              >
                {name}
              </button>
            ))}
          </div>

          {tab === "Tactile" ? (
            <div className="mt-4" role="tabpanel">
              <div className="mb-3 flex flex-wrap items-center gap-4">
                <label className="flex items-center gap-2 font-bold">
                  <input
                    type="checkbox"
                    checked={showPrint}
                    onChange={(event) => setShowPrint(event.target.checked)}
                  />
                  Show print labels
                </label>
                {svgUrl ? (
                  <a href={svgUrl} download="touchsight-tactile.svg" className="inline-flex min-h-11 items-center rounded-full bg-desk-2 px-4 font-bold text-gold">
                    Download SVG
                  </a>
                ) : null}
                {pdfUrl ? (
                  <a href={pdfUrl} download="touchsight-tactile.pdf" className="inline-flex min-h-11 items-center rounded-full bg-gold px-4 font-bold text-gold-deep">
                    Download PDF
                  </a>
                ) : null}
              </div>
              {previewUrl ? (
                <img src={previewUrl} alt="The image you uploaded" className="mb-3 max-h-36 rounded-lg bg-paper object-contain" />
              ) : null}
              {sheetMarkup ? (
                <motion.div
                  key={svgUrl}
                  className={showPrint ? "sheet-view rounded-2xl bg-paper p-2" : "sheet-view hide-print rounded-2xl bg-paper p-2"}
                  initial={reduceMotion ? false : { opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35 }}
                  dangerouslySetInnerHTML={{ __html: sheetMarkup }}
                  aria-label={outlineAlt(job)}
                />
              ) : (
                <div className="flex h-64 items-center justify-center rounded-2xl bg-paper text-ink">
                  {busy ? "Drawing the sheet…" : "The tactile sheet will land here"}
                </div>
              )}
            </div>
          ) : null}

          {tab === "Understanding" ? (
            <div className="mt-4" role="tabpanel">
              {regionMap ? (
                <>
                  <h3 className="font-serif text-2xl">{regionMap.subject || "Diagram"}</h3>
                  <p className="mt-2 max-w-2xl text-mist">{regionMap.summary}</p>
                  <ul className="mt-4 divide-y divide-white/10">
                    {(regionMap.regions ?? []).map((region) => (
                      <li key={region.id} className="py-3">
                        <p className="font-bold">
                          {region.name}
                          {region.pattern ? <span className="ml-2 font-normal text-mist">{region.pattern}</span> : null}
                        </p>
                        {region.description ? <p className="text-mist">{region.description}</p> : null}
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="text-mist">{busy ? "Reading the picture…" : "A summary will land here after you choose an image."}</p>
              )}
            </div>
          ) : null}

          {tab === "Explore" ? (
            <div className="mt-4" role="tabpanel">
              <ExploreView imageUrl={previewUrl} map={regionMap} />
            </div>
          ) : null}
          {tab === "Relief" ? (
            <div className="mt-4" role="tabpanel">
              <ReliefView
                stlUrl={stlUrl}
                heightmapUrl={heightmapUrl}
                regionIdUrl={regionIdUrl}
                textureUrl={previewUrl}
                regions={(regionMap?.regions ?? []).map((region) => ({
                  name: region.name,
                  description: region.description,
                }))}
              />
            </div>
          ) : null}

        </section>
      </main>
    </StartGate>
  );
}

function SampleGrid({
  heading,
  samples,
  busy,
  onPick,
}: {
  heading: string;
  samples: Sample[];
  busy: boolean;
  onPick: (sample: Sample) => Promise<void>;
}) {
  return (
    <div>
      <h3 className="text-base font-bold">{heading}</h3>
      <ul className="mt-3 grid grid-cols-2 gap-3">
        {samples.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              disabled={busy}
              onClick={() => void onPick(sample)}
              className="flex w-full flex-col overflow-hidden rounded-xl bg-desk-2 text-left disabled:opacity-60"
            >
              <img src={sample.image_url} alt="" className="h-24 w-full bg-paper object-contain" />
              <span className="px-3 py-2 font-bold">{sample.title}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function stageThumbs(previewUrl: string | null, job: Job): Partial<Record<string, string>> {
  const thumbs: Partial<Record<string, string>> = {};
  if (previewUrl) thumbs.understand = previewUrl;
  const svg = job.status === "complete" ? job.artifacts.tactile_svg : undefined;
  if (job.status === "complete" && job.artifacts.rectify_before) thumbs.rectify_before = job.artifacts.rectify_before;
  if (job.status === "complete" && job.artifacts.rectify_after) thumbs.rectify = job.artifacts.rectify_after;
  if (svg) {
    thumbs.simplify = svg;
    thumbs.texture = svg;
    thumbs.braille = svg;
  }
  return thumbs;
}

function describeStatus(busy: boolean, job: Job | null, error: string | null): string {
  if (error && !busy) return "";
  if (job?.status === "complete") {
    const count = job.region_count ?? 0;
    const where = job.cached ? "Loaded from cache. " : "";
    return `${where}${count} region${count === 1 ? "" : "s"} on the sheet.`;
  }
  if (!busy) return "Choose an image to begin.";
  const running = job?.stages.find((stage) => stage.status === "running");
  if (running?.name === "segment") return "Finding regions in the picture.";
  if (running?.name === "simplify") return "Smoothing lines so a finger can tell them apart.";
  if (running?.name === "export") return "Drawing the sheet.";
  return "Starting.";
}

function outlineAlt(job: Job | null): string {
  const count = job?.region_count ?? 0;
  return `Tactile outline with ${count} region${count === 1 ? "" : "s"} drawn as black lines on a white sheet.`;
}
