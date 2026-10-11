import { useEffect, useId, useState } from "react";
import {
  getVoiceRate,
  isVoiceEnabled,
  setVoiceEnabled,
  setVoiceRate,
  subscribeVoice,
  type VoiceRate,
} from "./speech";

const SPEEDS: ReadonlyArray<{ id: VoiceRate; label: string }> = [
  { id: "slow", label: "Slow" },
  { id: "normal", label: "Normal" },
  { id: "fast", label: "Fast" },
];

export function VoiceGuideToggle() {
  const speedLabelId = useId();
  const [enabled, setEnabled] = useState(isVoiceEnabled);
  const [rate, setRate] = useState<VoiceRate>(getVoiceRate);

  useEffect(() => {
    return subscribeVoice(() => {
      setEnabled(isVoiceEnabled());
      setRate(getVoiceRate());
    });
  }, []);

  return (
    <div className="ts-voice-controls">
      <button
        type="button"
        aria-pressed={enabled}
        onClick={() => setVoiceEnabled(!isVoiceEnabled())}
      >
        {enabled ? "Voice guide on" : "Voice guide off"}
      </button>
      <div className="ts-speeds" role="group" aria-labelledby={speedLabelId}>
        <span id={speedLabelId}>Speaking speed</span>
        {SPEEDS.map((speed) => (
          <button
            key={speed.id}
            type="button"
            aria-pressed={rate === speed.id}
            onClick={() => setVoiceRate(speed.id)}
          >
            {speed.label}
          </button>
        ))}
      </div>
    </div>
  );
}
