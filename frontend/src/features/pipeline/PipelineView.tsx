import { useId, type JSX } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { StageProgress } from "../../types";

const SEQUENCE = ["understand", "segment", "simplify", "texture", "braille", "relief"] as const;

const STAGE_LABELS: Record<string, string> = {
  understand: "Understand",
  segment: "Segment",
  simplify: "Simplify",
  texture: "Texture",
  braille: "Braille",
  relief: "Relief",
  export: "Export",
};

export function PipelineView(props: {
  stages: StageProgress[];
  active: boolean;
  thumbs?: Partial<Record<string, string>>;
}): JSX.Element {
  const headingId = useId();
  const reduceMotion = useReducedMotion() === true;
  const travel = props.active && !reduceMotion;
  const ordered = orderStages(props.stages);

  return (
    <div className="mt-4">
      <h3 id={headingId} className="font-bold text-paper">
        Pipeline
      </h3>
      <motion.ol aria-labelledby={headingId} className="mt-2 flex flex-col gap-2">
        {ordered.map((stage, index) => (
          <StageRow
            key={stage.name}
            stage={stage}
            index={index}
            travel={travel}
            thumb={thumbFor(props.thumbs, stage.name)}
            before={stage.name === "rectify" ? thumbFor(props.thumbs, "rectify_before") : null}
          />
        ))}
      </motion.ol>
    </div>
  );
}

function StageRow({
  stage,
  index,
  travel,
  thumb,
  before,
}: {
  stage: StageProgress;
  index: number;
  travel: boolean;
  thumb: string | null;
  before: string | null;
}) {
  const badge = badgeText(stage);
  return (
    <motion.li
      initial={travel ? { opacity: 0, y: 12 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: travel ? 0.32 : 0, delay: travel ? index * 0.09 : 0, ease: "easeOut" }}
      className="flex items-center gap-3 rounded-xl bg-desk-2 px-3 py-2"
    >
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-bold text-paper">{STAGE_LABELS[stage.name] ?? stage.name}</span>
        <span className={badgeClass(badge)}>{badge}</span>
        {stage.elapsed_ms != null ? (
          <span className="text-mist">{Math.round(stage.elapsed_ms)} ms</span>
        ) : null}
      </div>
      {before || thumb ? (
        <span className="flex shrink-0 gap-1">
          {before ? <img src={before} alt="Page before straightening" className="h-12 w-16 rounded bg-paper object-contain" /> : null}
          {thumb ? (
            <img
              src={thumb}
              alt={before ? "Page after straightening" : ""}
              className="h-12 w-16 rounded bg-paper object-contain"
            />
          ) : null}
        </span>
      ) : null}
    </motion.li>
  );
}

function orderStages(stages: StageProgress[]): StageProgress[] {
  const byName = new Map(stages.map((stage) => [stage.name, stage]));
  const ordered: StageProgress[] = [];
  for (const name of SEQUENCE) {
    const stage = byName.get(name);
    if (stage) ordered.push(stage);
  }
  const sequenced = new Set<string>(SEQUENCE);
  for (const stage of stages) {
    if (!sequenced.has(stage.name)) ordered.push(stage);
  }
  return ordered;
}

function thumbFor(thumbs: Partial<Record<string, string>> | undefined, name: string): string | null {
  const value = thumbs?.[name];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function badgeText(stage: StageProgress): string {
  const provider = stage.provider?.trim() ? stage.provider.trim() : null;
  if (stage.name === "understand" && stage.status === "complete" && stage.message?.includes(" · ")) {
    return stage.message;
  }
  if (stage.status === "complete" && provider === "vision-llm") return "vision-llm";
  if (stage.status === "complete" && provider === "curated") return "curated";
  if (stage.status === "skipped" || provider === "fallback" || provider === "sample-labels") {
    return "Fallback";
  }
  if (stage.status === "complete") {
    return provider ? `Done · ${provider}` : "Done";
  }
  if (stage.status === "running") return "Working";
  if (stage.status === "failed") return "Failed";
  return "Waiting";
}

function badgeClass(badge: string): string {
  const base = "rounded-full px-2.5 py-0.5 text-sm font-bold";
  if (badge === "vision-llm" || badge.includes(" · ")) return `${base} bg-gold text-gold-deep`;
  if (badge === "curated") return `${base} bg-desk text-mist`;
  if (badge.startsWith("Done")) return `${base} bg-gold text-gold-deep`;
  if (badge === "Working") return `${base} bg-desk text-gold`;
  if (badge === "Failed") return `${base} bg-desk text-danger`;
  return `${base} bg-desk text-mist`;
}
