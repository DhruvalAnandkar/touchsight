import { useCallback, useEffect, useId, useMemo, useRef, useState, type JSX } from "react";
import type { RegionMap } from "../../types";

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

function useSpokenCaption(): { caption: string; speak: (text: string) => void; cancel: () => void } {
  const generation = useRef(0);
  const timer = useRef<number | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const [caption, setCaption] = useState("");

  const clearTimer = useCallback(() => {
    if (timer.current != null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    generation.current += 1;
    clearTimer();
    window.speechSynthesis?.cancel();
    setCaption("");
  }, [clearTimer]);

  const speak = useCallback((text: string) => {
    const spoken = text.trim();
    const synth = window.speechSynthesis;
    if (!spoken || !synth) return;
    const id = ++generation.current;
    clearTimer();
    synth.cancel();
    setCaption(spoken);
    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = "en-US";
    utteranceRef.current = utterance;
    utterance.onend = () => {
      if (generation.current === id) setCaption("");
    };
    utterance.onerror = (event) => {
      if (generation.current !== id) return;
      if (event.error === "interrupted" || event.error === "canceled") return;
      setCaption("");
    };
    // Chrome drops a speak() that shares a turn with cancel().
    timer.current = window.setTimeout(() => {
      timer.current = null;
      if (generation.current !== id) return;
      synth.speak(utterance);
      if (synth.paused) synth.resume();
    }, 0);
  }, [clearTimer]);

  useEffect(() => {
    return () => {
      generation.current += 1;
      if (timer.current != null) window.clearTimeout(timer.current);
      window.speechSynthesis?.cancel();
    };
  }, []);

  return { caption, speak, cancel };
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
  const { caption, speak, cancel } = useSpokenCaption();
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
    cancel();
  }, [imageUrl, map, cancel]);

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
    if (name) speak(name);
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
    speak(summary);
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
          if (description) speak(description);
        }}
        onDoubleClick={(event) => {
          const hit = regionAt(event.currentTarget, event.clientX, event.clientY);
          if (!hit) speakSummary();
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            cancel();
            return;
          }
          if (event.key === "Enter") {
            event.preventDefault();
            const description = active?.description.trim();
            if (description) speak(description);
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
