import { useEffect, useState } from "react";
import { getCaption, subscribeCaption } from "./speech";

export function VoiceChrome() {
  const [visual, setVisual] = useState(getCaption);
  const [live, setLive] = useState("");

  useEffect(() => {
    let timer = 0;
    const publish = (text: string) => {
      setVisual(text);
      setLive("");
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setLive(text), 30);
    };

    const current = getCaption();
    if (current) publish(current);
    const unsubscribe = subscribeCaption(publish);
    return () => {
      window.clearTimeout(timer);
      unsubscribe();
    };
  }, []);

  return (
    <>
      <div className="ts-caption-bar" aria-hidden="true">
        <p>{visual}</p>
      </div>
      <div className="ts-live sr-only" aria-live="polite" aria-atomic="true">
        {live}
      </div>
    </>
  );
}
