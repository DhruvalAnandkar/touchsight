import { isVoiceEnabled, repeatLast, speak, speakHelp, stopSpeaking } from "./speech";

const CONTROL_SELECTOR = 'a[href], button, [role="button"], [role="link"]';
const TAP_WINDOW_MS = 1500;

let installed = false;
let lastPointerType = "mouse";
let lastInputWasKeyboard = false;
let suppressFocusAnnounce = false;
let lastTapElement: HTMLElement | null = null;
let lastTapAt = 0;
let lastAnnouncedElement: HTMLElement | null = null;
let lastAnnouncedAt = 0;

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.getAttribute("role") === "textbox") return true;
  return false;
}

function controlFrom(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) return null;
  const element = target.closest(CONTROL_SELECTOR);
  return element instanceof HTMLElement ? element : null;
}

function accessibleName(element: HTMLElement): string {
  const label = element.getAttribute("aria-label");
  if (label && label.trim()) return label.trim();
  return (element.textContent || "").replace(/\s+/g, " ").trim();
}

function usesTapPreview(pointerType: string): boolean {
  if (pointerType === "mouse") return false;
  if (window.matchMedia("(pointer: coarse)").matches) return true;
  return pointerType === "touch";
}

function announce(element: HTMLElement): void {
  if (!isVoiceEnabled()) return;
  const now = Date.now();
  if (element === lastAnnouncedElement && now - lastAnnouncedAt < 500) return;
  const name = accessibleName(element);
  if (!name) return;
  lastAnnouncedElement = element;
  lastAnnouncedAt = now;
  speak(name, { priority: true });
}

function onKeyDown(event: KeyboardEvent): void {
  if (event.key === "Tab" || event.key === "Enter") {
    lastInputWasKeyboard = true;
    suppressFocusAnnounce = false;
  }

  if (event.key === "Escape") {
    stopSpeaking();
    return;
  }

  if (isEditable(event.target)) return;
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;

  if (event.key === " " || event.key === "Spacebar" || event.code === "Space") {
    event.preventDefault();
    repeatLast();
    return;
  }

  if (event.key === "h" || event.key === "H") {
    event.preventDefault();
    speakHelp();
  }
}

function onPointerDown(event: PointerEvent): void {
  lastPointerType = event.pointerType || "mouse";
  lastInputWasKeyboard = false;
  suppressFocusAnnounce = usesTapPreview(lastPointerType);
}

function onPointerOver(event: PointerEvent): void {
  if (!isVoiceEnabled()) return;
  if (event.pointerType === "touch") return;
  if (window.matchMedia("(pointer: coarse)").matches) return;
  const control = controlFrom(event.target);
  if (!control) return;
  announce(control);
}

function onFocusIn(event: FocusEvent): void {
  if (!isVoiceEnabled() || suppressFocusAnnounce) return;
  const control = controlFrom(event.target);
  if (!control) return;
  announce(control);
}

function onClick(event: MouseEvent): void {
  if (!isVoiceEnabled()) return;
  if (event.detail === 0 || lastInputWasKeyboard) {
    lastInputWasKeyboard = false;
    suppressFocusAnnounce = false;
    return;
  }

  const control = controlFrom(event.target);
  if (!control || !usesTapPreview(lastPointerType)) return;

  const now = Date.now();
  const secondTap = control === lastTapElement && now - lastTapAt < TAP_WINDOW_MS;
  if (secondTap) {
    lastTapElement = null;
    lastTapAt = 0;
    return;
  }

  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation();
  lastTapElement = control;
  lastTapAt = now;
  announce(control);
}

export function installVoiceGuide(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("pointerdown", onPointerDown, true);
  window.addEventListener("pointerover", onPointerOver, true);
  window.addEventListener("focusin", onFocusIn, true);
  window.addEventListener("click", onClick, true);
}
