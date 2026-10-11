export type VoiceRate = "slow" | "normal" | "fast";

const STORAGE_KEY = "touchsight-voice";

const RATES: Record<VoiceRate, number> = {
  slow: 0.8,
  normal: 1,
  fast: 1.2,
};

const HELP_TEXT =
  "Tab moves between buttons. Space repeats the last message. H speaks this help. Escape stops speaking. On a touch screen, the first tap says the button name and the second tap presses it.";

let enabled = true;
let rate: VoiceRate = "normal";
let queue: string[] = [];
let speaking = false;
let speakToken = 0;
let lastMessage = "";
let caption = "";
let preferredVoice: SpeechSynthesisVoice | null = null;

const stateListeners = new Set<() => void>();
const captionListeners = new Set<(text: string) => void>();

function notifyState(): void {
  stateListeners.forEach((listener) => listener());
}

function notifyCaption(text: string): void {
  caption = text;
  captionListeners.forEach((listener) => listener(text));
}

function loadPrefs(): void {
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data: unknown = JSON.parse(raw);
    if (!data || typeof data !== "object") return;
    const record = data as { enabled?: unknown; rate?: unknown };
    if (typeof record.enabled === "boolean") enabled = record.enabled;
    if (record.rate === "slow" || record.rate === "normal" || record.rate === "fast") {
      rate = record.rate;
    }
  } catch {
    /* storage missing or blocked */
  }
}

function savePrefs(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ enabled, rate }));
  } catch {
    /* storage missing or blocked */
  }
}

function pickVoice(): void {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  const voices = window.speechSynthesis.getVoices();
  const english = voices.filter((voice) => voice.lang.toLowerCase().startsWith("en"));
  preferredVoice =
    english.find((voice) => voice.lang.toLowerCase().startsWith("en-us") && voice.localService) ??
    english.find((voice) => voice.default) ??
    english[0] ??
    null;
}

loadPrefs();

if (typeof window !== "undefined" && "speechSynthesis" in window) {
  pickVoice();
  window.speechSynthesis.addEventListener("voiceschanged", pickVoice);
}

function synth(): SpeechSynthesis | null {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  return window.speechSynthesis;
}

function utter(text: string, token: number): void {
  const engine = synth();
  if (!engine) {
    speaking = false;
    lastMessage = text;
    notifyCaption(text);
    return;
  }

  pickVoice();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = RATES[rate];
  utterance.lang = preferredVoice?.lang ?? "en-US";
  if (preferredVoice) utterance.voice = preferredVoice;

  utterance.onend = () => {
    if (token !== speakToken) return;
    speaking = false;
    drain();
  };

  utterance.onerror = (event) => {
    if (token !== speakToken) return;
    speaking = false;
    if (event.error === "canceled" || event.error === "interrupted") return;
    drain();
  };

  speaking = true;
  lastMessage = text;
  notifyCaption(text);
  if (engine.paused) engine.resume();
  engine.speak(utterance);
}

function drain(): void {
  if (!enabled || speaking || queue.length === 0) return;
  const next = queue.shift();
  if (!next) return;
  utter(next, speakToken);
}

function halt(): void {
  speakToken += 1;
  queue = [];
  speaking = false;
  synth()?.cancel();
}

export function isVoiceEnabled(): boolean {
  return enabled;
}

export function getVoiceRate(): VoiceRate {
  return rate;
}

export function getCaption(): string {
  return caption;
}

export function subscribeVoice(listener: () => void): () => void {
  stateListeners.add(listener);
  return () => stateListeners.delete(listener);
}

export function subscribeCaption(listener: (text: string) => void): () => void {
  captionListeners.add(listener);
  return () => captionListeners.delete(listener);
}

export function setVoiceEnabled(next: boolean): void {
  enabled = next;
  if (!next) halt();
  savePrefs();
  notifyState();
}

export function setVoiceRate(next: VoiceRate): void {
  rate = next;
  savePrefs();
  notifyState();
}

export function speak(text: string, options?: { priority?: boolean }): void {
  if (!enabled) return;
  const cleaned = text.trim();
  if (!cleaned) return;

  const engine = synth();
  if (!engine) {
    lastMessage = cleaned;
    notifyCaption(cleaned);
    return;
  }

  try {
    if (options?.priority) {
      const token = ++speakToken;
      queue = [];
      const busy = engine.speaking || engine.pending || speaking;
      speaking = true;
      if (busy) engine.cancel();
      const start = () => {
        if (token !== speakToken) return;
        if (!enabled) {
          speaking = false;
          return;
        }
        utter(cleaned, token);
      };
      if (busy) window.setTimeout(start, 40);
      else start();
      return;
    }

    queue.push(cleaned);
    drain();
  } catch {
    lastMessage = cleaned;
    notifyCaption(cleaned);
    speaking = false;
  }
}

export function stopSpeaking(): void {
  halt();
}

export function repeatLast(): void {
  if (!lastMessage) return;
  speak(lastMessage, { priority: true });
}

export function speakHelp(): void {
  speak(HELP_TEXT, { priority: true });
}
