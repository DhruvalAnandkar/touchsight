export {
  getCaption,
  getVoiceRate,
  isVoiceEnabled,
  repeatLast,
  setVoiceEnabled,
  setVoiceRate,
  speak,
  speakHelp,
  stopSpeaking,
  subscribeCaption,
  subscribeVoice,
} from "./speech";
export type { VoiceRate } from "./speech";
export { installVoiceGuide } from "./guide";
export { VoiceChrome } from "./VoiceChrome";
export { VoiceGuideToggle } from "./VoiceGuideToggle";
export { StartGate, VoiceGuideRoot } from "./StartGate";
