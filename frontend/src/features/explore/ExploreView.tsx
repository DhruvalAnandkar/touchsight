import { useEffect, useId, useMemo, useRef, useState, type JSX } from "react";
import type { RegionMap } from "../../types";
import { speak, stopSpeaking } from "../../voice/speak";

type Region = NonNullable<RegionMap["regions"]>[number];

const EMPTY_REGIONS: Region[] = [];
const TAP_SLOP_PX = 12;
const DOUBLE_TAP_MS = 400;
const DOUBLE_TAP_PX = 48;

/**
 * Even-odd ray cast. Polygon points are [x, y] in the same 0..1 space as the region map.
 */
function pointInPolygon(x: number, y: number, polygon: readonly (readonly number[])[]): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const current = polygon[index];
    const prior = polygon[previous];
    if (!current || !prior) continue;
    const x1 = current[0];
    const y1 = current[1];
    const x2 = prior[0];
    const y2 = prior[1];
    if (x1 == null || y1 == null || x2 == null || y2 == null || y1 === y2) continue;
    const crosses = y1 > y !== y2 > y;
    if (crosses && x < ((x2 - x1) * (y - y1)) / (y2 - y1) + x1) inside = !inside;
  }
  return inside;
}

function usablePolygon(polygon: number[][] | undefined): number[][] | null {
  if (!polygon || polygon.length < 3) return null;
  const points = polygon.filter(
    (point) => point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]),
  );
  return points.length >= 3 ? points : null;
}

function hitRegion(x: number, y: number, regions: readonly Region[]): Region | null {
  let best: Region | null = null;
  let bestImportance = Number.NEGATIVE_INFINITY;
  let bestIndex = -1;
  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index];
    if (!region) continue;
    const polygon = usablePolygon(region.polygon);
    if (!polygon || !pointInPolygon(x, y, polygon)) continue;
    const importance = region.importance ?? 0;
    if (importance > bestImportance || (importance === bestImportance && index > bestIndex)) {
      best = region;
      bestImportance = importance;
      bestIndex = index;
    }
  }
  return best;
}

function regionFromClient(
  surface: HTMLElement,
  clientX: number,
  clientY: number,
  regions: readonly Region[],
): Region | null {
  const bounds = surface.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return null;
  const x = (clientX - bounds.left) / bounds.width;
  const y = (clientY - bounds.top) / bounds.height;
  if (x < 0 || y < 0 || x > 1 || y > 1) return null;
  return hitRegion(x, y, regions);
}

function polygonPoints(polygon: readonly (readonly number[])[]): string {
  return polygon.map((point) => `${point[0]},${point[1]}`).join(" ");
}

function byImportance(regions: readonly Region[], direction: "asc" | "desc"): Region[] {
  return regions
    .map((region, index) => ({ region, index }))
    .sort((a, b) => {
      const importance = (a.region.importance ?? 0) - (b.region.importance ?? 0);
      const signed = direction === "asc" ? importance : -importance;
      return signed || a.index - b.index;
    })
    .map((item) => item.region);
}

function answerQuestion(raw: string, map: RegionMap): string {
  const text = raw.trim().toLowerCase().replace(/[?.!]/g, "");
  if (!text) return "Ask for a part, or say what is this.";
  if (text.includes("what is this") || text === "summary") {
    return map.summary?.trim() || map.subject?.trim() || "This is a diagram.";
  }
  if (text.includes("list the parts") || text.includes("list parts")) {
    const names = (map.regions ?? []).map((region) => region.name.trim()).filter(Boolean);
    return names.length > 0 ? `The parts are ${names.join(", ")}.` : "There are no named parts.";
  }
  let best: Region | null = null;
  let bestScore = 0;
  for (const region of map.regions ?? []) {
    const score = nameScore(text, region.name.toLowerCase());
    if (score > bestScore) {
      best = region;
      bestScore = score;
    }
  }
  if (best && bestScore >= 0.45) {
    const description = best.description.trim();
    return description ? `${best.name}. ${description}` : best.name;
  }
  return "I could not match that to a part. Say list the parts.";
}

function nameScore(query: string, name: string): number {
  if (!name) return 0;
  if (query.includes(name) || name.includes(query)) return 1;
  const queryWords = query.split(/\W+/).filter((word) => word.length > 2);
  const nameWords = name.split(/\W+/).filter((word) => word.length > 2);
  if (queryWords.length === 0 || nameWords.length === 0) return 0;
  const hits = nameWords.filter((word) =>
    queryWords.some((token) => token === word || token.includes(word) || word.includes(token)),
  ).length;
  return hits / nameWords.length;
}

type SpeechListener = {
  lang: string;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: (() => void) | null;
  start: () => void;
};

function speechRecognizer(): (new () => SpeechListener) | null {
  const host = window as Window & {
    SpeechRecognition?: new () => SpeechListener;
    webkitSpeechRecognition?: new () => SpeechListener;
  };
  return host.SpeechRecognition ?? host.webkitSpeechRecognition ?? null;
}

export function ExploreView(props: { imageUrl: string | null; map: RegionMap | null }): JSX.Element {
  const { imageUrl, map } = props;
  const hintId = useId();
  const surfaceRef = useRef<HTMLDivElement>(null);
  const hoverIdRef = useRef<string | null>(null);
  const gestureRef = useRef({ x: 0, y: 0, moved: false, tracking: false });
  const lastEmptyTapRef = useRef<{ at: number; x: number; y: number } | null>(null);
  const summaryAtRef = useRef(0);
  const skipResetRef = useRef(true);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [question, setQuestion] = useState("");
  const [listening, setListening] = useState(false);
  const canListen = speechRecognizer() != null;

  function say(text: string) {
    const spoken = text.trim();
    if (!spoken) return;
    setCaption(spoken);
    speak(spoken, { priority: true });
  }
  const regions = map?.regions ?? EMPTY_REGIONS;
  const painted = useMemo(() => byImportance(regions, "asc"), [regions]);
  const ordered = useMemo(() => byImportance(regions, "desc"), [regions]);

  useEffect(() => {
    if (skipResetRef.current) {
      skipResetRef.current = false;
      return;
    }
    hoverIdRef.current = null;
    lastEmptyTapRef.current = null;
    setActiveId(null);
    setCaption("");
    stopSpeaking();
  }, [imageUrl, map]);

  if (!imageUrl || !map) {
    return (
      <div className="flex h-64 items-center justify-center rounded-2xl bg-paper px-6 text-center text-ink">
        Choose an image to explore it by touch.
      </div>
    );
  }

  const active = regions.find((region) => region.id === activeId) ?? null;
  const activePolygon = active ? usablePolygon(active.polygon) : null;
  const subject = map.subject?.trim() || "Picture";

  function announce(region: Region) {
    hoverIdRef.current = region.id;
    setActiveId(region.id);
    const name = region.name.trim();
    if (name) say(name);
    if (typeof navigator.vibrate === "function") navigator.vibrate(30);
  }

  function regionAt(surface: HTMLElement, clientX: number, clientY: number): Region | null {
    return regionFromClient(surface, clientX, clientY, regions);
  }

  function movePointer(surface: HTMLElement, clientX: number, clientY: number) {
    const hit = regionAt(surface, clientX, clientY);
    const nextId = hit?.id ?? null;
    if (nextId === hoverIdRef.current) return;
    hoverIdRef.current = nextId;
    if (hit) announce(hit);
    else setActiveId(null);
  }

  function speakSummary() {
    const summary = map?.summary?.trim();
    if (!summary) return;
    const now = performance.now();
    if (now - summaryAtRef.current < 500) return;
    summaryAtRef.current = now;
    say(summary);
  }

  return (
    <div className="flex flex-col gap-3">
      {map.subject?.trim() ? <p className="font-serif text-2xl">{map.subject.trim()}</p> : null}
      <div
        ref={surfaceRef}
        tabIndex={0}
        role="application"
        aria-label={`${subject}. Slide across a region to hear its name. Tap a region for a description. Double-tap empty space for the summary. Arrow keys move between regions, Enter speaks a description, Escape stops speech.`}
        aria-describedby={hintId}
        onPointerDown={(event) => {
          gestureRef.current = { x: event.clientX, y: event.clientY, moved: false, tracking: true };
          if (event.pointerType === "touch") hoverIdRef.current = null;
          try {
            event.currentTarget.setPointerCapture(event.pointerId);
          } catch {
            // Hit testing still works when the browser rejects capture.
          }
          event.currentTarget.focus({ preventScroll: true });
          movePointer(event.currentTarget, event.clientX, event.clientY);
        }}
        onPointerMove={(event) => {
          const gesture = gestureRef.current;
          if (gesture.tracking) {
            const distance = Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y);
            if (distance > TAP_SLOP_PX) gesture.moved = true;
          }
          movePointer(event.currentTarget, event.clientX, event.clientY);
        }}
        onPointerUp={(event) => {
          gestureRef.current.tracking = false;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          if (event.pointerType === "mouse" || gestureRef.current.moved) return;
          const hit = regionAt(event.currentTarget, event.clientX, event.clientY);
          if (hit) {
            lastEmptyTapRef.current = null;
            return;
          }
          const now = performance.now();
          const last = lastEmptyTapRef.current;
          const near =
            last != null &&
            now - last.at < DOUBLE_TAP_MS &&
            Math.hypot(event.clientX - last.x, event.clientY - last.y) < DOUBLE_TAP_PX;
          if (near) {
            lastEmptyTapRef.current = null;
            speakSummary();
            return;
          }
          lastEmptyTapRef.current = { at: now, x: event.clientX, y: event.clientY };
        }}
        onPointerCancel={(event) => {
          gestureRef.current.tracking = false;
          hoverIdRef.current = null;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onPointerLeave={() => {
          hoverIdRef.current = null;
        }}
        onClick={(event) => {
          if (gestureRef.current.moved) return;
          const hit = regionAt(event.currentTarget, event.clientX, event.clientY);
          const description = hit?.description.trim();
          if (description) say(description);
        }}
        onDoubleClick={(event) => {
          const hit = regionAt(event.currentTarget, event.clientX, event.clientY);
          if (!hit) speakSummary();
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            stopSpeaking();
            setCaption("");
            return;
          }
          if (event.key === "Enter") {
            event.preventDefault();
            const description = active?.description.trim();
            if (description) say(description);
            return;
          }
          const forward = event.key === "ArrowRight" || event.key === "ArrowDown";
          const backward = event.key === "ArrowLeft" || event.key === "ArrowUp";
          if (!forward && !backward) return;
          event.preventDefault();
          if (ordered.length === 0) return;
          const index = ordered.findIndex((region) => region.id === activeId);
          const nextIndex = forward
            ? index < 0
              ? 0
              : (index + 1) % ordered.length
            : index <= 0
              ? ordered.length - 1
              : index - 1;
          const next = ordered[nextIndex];
          if (next) announce(next);
        }}
        onContextMenu={(event) => event.preventDefault()}
        className="relative w-full touch-none overflow-hidden rounded-2xl bg-paper select-none"
      >
        <img
          src={imageUrl}
          alt=""
          draggable={false}
          decoding="async"
          className="block h-auto w-full bg-paper"
        />
        <svg
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 h-full w-full"
        >
          {painted.map((region) => {
            const polygon = usablePolygon(region.polygon);
            if (!polygon) return null;
            return (
              <polygon
                key={region.id}
                points={polygonPoints(polygon)}
                fill="rgba(240, 193, 75, 0.18)"
                stroke="#14140f"
                strokeWidth={2}
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
                fillRule="evenodd"
              />
            );
          })}
          {activePolygon ? (
            <polygon
              points={polygonPoints(activePolygon)}
              fill="rgba(240, 193, 75, 0.48)"
              stroke="#f0c14b"
              strokeWidth={5}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              fillRule="evenodd"
            />
          ) : null}
        </svg>
      </div>
      <p id={hintId} className="text-mist">
        Slide across a region to hear its name, or use the arrow keys. Tap a region for a description. Double-tap
        empty space for the summary.
      </p>
      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!map) return;
          say(answerQuestion(question, map));
        }}
      >
        <label htmlFor={`${hintId}-ask`} className="font-bold">
          Ask about this diagram
        </label>
        <div className="flex flex-wrap gap-2">
          <input
            id={`${hintId}-ask`}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Try: what is the left atrium?"
            className="min-h-11 min-w-0 flex-1 rounded-xl bg-desk-2 px-4 text-paper placeholder:text-mist"
          />
          {canListen ? (
            <button
              type="button"
              className="min-h-11 rounded-full bg-desk-2 px-4 font-bold text-gold"
              onClick={() => {
                const Ctor = speechRecognizer();
                if (!Ctor || !map) return;
                const heard = new Ctor();
                heard.lang = "en-US";
                heard.onresult = (event) => {
                  const transcript = event.results[0]?.[0]?.transcript ?? "";
                  setQuestion(transcript);
                  setListening(false);
                  say(answerQuestion(transcript, map));
                };
                heard.onerror = () => setListening(false);
                setListening(true);
                heard.start();
              }}
            >
              {listening ? "Listening…" : "Ask about this diagram"}
            </button>
          ) : null}
          <button type="submit" className="min-h-11 rounded-full bg-gold px-4 font-bold text-gold-deep">
            Ask
          </button>
        </div>
      </form>
      <p role="status" aria-live="polite" aria-atomic="true" className="min-h-16 rounded-2xl bg-desk-2 px-4 py-3 text-lg font-bold text-gold">
        {caption ? (
          caption
        ) : (
          <span className="font-normal text-mist" aria-hidden="true">
            Spoken names and descriptions show up here.
          </span>
        )}
      </p>
    </div>
  );
}
