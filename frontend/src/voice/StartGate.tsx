import { useEffect, useRef, useState, type ReactNode } from "react";
import { installVoiceGuide } from "./guide";
import { setVoiceEnabled, speak } from "./speech";
import { VoiceChrome } from "./VoiceChrome";

const WELCOME =
  "Welcome to TouchSight. We turn pictures into something you can feel and hear. Press H anytime for help. Tab moves between buttons.";

export function VoiceGuideRoot({ children }: { children: ReactNode }) {
  useEffect(() => {
    installVoiceGuide();
  }, []);

  return (
    <div className="ts-shell">
      {children}
      <VoiceChrome />
    </div>
  );
}

export function StartGate({ children }: { children: ReactNode }) {
  const [started, setStarted] = useState(false);
  const startedRef = useRef(false);

  const begin = (withVoice: boolean) => {
    if (startedRef.current) return;
    startedRef.current = true;
    setVoiceEnabled(withVoice);
    installVoiceGuide();
    if (withVoice) speak(WELCOME, { priority: true });
    setStarted(true);
  };

  const beginRef = useRef(begin);
  beginRef.current = begin;

  useEffect(() => {
    if (started) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (startedRef.current) return;
      if (event.key === "Tab") return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest("[data-start-silent]")) {
        if (event.key === "Enter" || event.key === " " || event.code === "Space") return;
      }
      event.preventDefault();
      beginRef.current(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [started]);

  if (!started) {
    return (
      <section className="ts-start" aria-labelledby="ts-start-title" onClick={() => begin(true)}>
        <h1 id="ts-start-title">TouchSight</h1>
        <p>tap anywhere or press any key to start</p>
        <button
          type="button"
          data-start-silent=""
          onClick={(event) => {
            event.stopPropagation();
            begin(false);
          }}
        >
          Start without voice
        </button>
      </section>
    );
  }

  return <VoiceGuideRoot>{children}</VoiceGuideRoot>;
}
