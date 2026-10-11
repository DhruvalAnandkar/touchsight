import { VoiceGuideToggle } from "../voice";

type LandingPageProps = {
  onTry: () => void;
  onPhoto: () => void;
  onHow: () => void;
};

const BARS = [
  { x: 40, y: 118, height: 92, pattern: "url(#ts-stripe)" },
  { x: 145, y: 48, height: 162, pattern: "url(#ts-grain)" },
  { x: 250, y: 84, height: 126, pattern: "url(#ts-cross)" },
] as const;

function brailleDots(x: number, top: number): Array<{ cx: number; cy: number }> {
  const center = x + 35;
  const dots: Array<{ cx: number; cy: number }> = [];
  for (let column = 0; column < 2; column += 1) {
    for (let row = 0; row < 3; row += 1) {
      dots.push({ cx: center - 9 + column * 18, cy: top + 20 + row * 16 });
    }
  }
  return dots;
}

function SampleChart() {
  const dots = BARS.flatMap((bar) => brailleDots(bar.x, bar.y));

  return (
    <figure className="ts-diagram">
      <svg viewBox="0 0 360 260" aria-hidden="true" focusable="false">
        <defs>
          <pattern id="ts-stripe" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(40)">
            <rect width="10" height="10" fill="var(--color-paper)" />
            <line x1="0" y1="0" x2="0" y2="10" stroke="var(--color-ink)" strokeWidth="3" />
          </pattern>
          <pattern id="ts-grain" width="14" height="14" patternUnits="userSpaceOnUse">
            <rect width="14" height="14" fill="var(--color-paper)" />
            <circle cx="4" cy="4" r="2.4" fill="var(--color-ink)" />
            <circle cx="11" cy="11" r="2.4" fill="var(--color-ink)" />
          </pattern>
          <pattern id="ts-cross" width="12" height="12" patternUnits="userSpaceOnUse">
            <rect width="12" height="12" fill="var(--color-paper)" />
            <path d="M0 0 H12 M0 0 V12" stroke="var(--color-ink)" strokeWidth="2" />
          </pattern>
        </defs>
        <rect className="ts-sheet" x="8" y="8" width="344" height="244" rx="28" />
        <line className="ts-draw" x1="32" y1="210" x2="328" y2="210" style={{ animationDelay: "0s" }} />
        {BARS.map((bar) => (
          <rect
            key={`fill-${bar.x}`}
            className="ts-texture"
            x={bar.x}
            y={bar.y}
            width="70"
            height={bar.height}
            rx="16"
            fill={bar.pattern}
          />
        ))}
        {BARS.map((bar, index) => (
          <rect
            key={`line-${bar.x}`}
            className="ts-draw"
            x={bar.x}
            y={bar.y}
            width="70"
            height={bar.height}
            rx="16"
            style={{ animationDelay: `${0.15 + index * 0.18}s` }}
          />
        ))}
        {dots.map((dot, index) => (
          <g
            key={`${dot.cx}-${dot.cy}`}
            className="ts-dot"
            style={{ animationDelay: `${2.55 + index * 0.045}s` }}
          >
            <circle cx={dot.cx} cy={dot.cy} r="7.5" fill="var(--color-paper)" />
            <circle cx={dot.cx} cy={dot.cy} r="4.2" fill="var(--color-ink)" />
          </g>
        ))}
      </svg>
      <figcaption>
        A sample chart from class. Monday is medium, Tuesday is tallest, Wednesday sits between them.
        Outlines draw in, textures fill the bars, then raised dots pop in.
      </figcaption>
    </figure>
  );
}

function SnapIcon() {
  return (
    <svg className="ts-icon" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
      <rect x="18" y="10" width="36" height="52" rx="8" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="36" cy="40" r="8" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle className="ts-icon-motion" cx="36" cy="22" r="3" fill="currentColor" />
    </svg>
  );
}

function UnderstandIcon() {
  return (
    <svg className="ts-icon" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
      <path d="M16 36 H56" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <circle className="ts-icon-motion ts-delay-1" cx="16" cy="36" r="6" fill="currentColor" />
      <circle className="ts-icon-motion ts-delay-2" cx="36" cy="36" r="6" fill="currentColor" />
      <circle className="ts-icon-motion ts-delay-3" cx="56" cy="36" r="6" fill="currentColor" />
    </svg>
  );
}

function FeelIcon() {
  return (
    <svg className="ts-icon" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
      <rect x="14" y="16" width="32" height="40" rx="6" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="24" cy="30" r="3" fill="currentColor" />
      <circle cx="34" cy="30" r="3" fill="currentColor" />
      <circle cx="24" cy="40" r="3" fill="currentColor" />
      <path
        className="ts-icon-motion"
        d="M52 28c8 4 8 12 0 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path d="M18 62 H42" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

const STEPS = [
  {
    title: "Snap or upload",
    text: "Use your camera on a worksheet, or pick a picture you already have.",
    icon: <SnapIcon />,
    num: "1",
  },
  {
    title: "AI understands",
    text: "We read the layout, the shapes, and the labels so the meaning stays put.",
    icon: <UnderstandIcon />,
    num: "2",
  },
  {
    title: "Feel it hear it print it",
    text: "Trace the raised graphic, listen as it is described, or print a copy for later.",
    icon: <FeelIcon />,
    num: "3",
  },
] as const;

export function LandingPage({ onTry, onPhoto, onHow }: LandingPageProps) {
  return (
    <main className="ts-landing">
      <section className="ts-hero" aria-labelledby="ts-title">
        <div className="ts-glow" aria-hidden="true" />
        <div className="ts-hero-grid">
          <div className="ts-hero-copy">
            <h1 id="ts-title">TouchSight</h1>
            <h2>Pictures you can feel and hear</h2>
            <p>
              Bring a diagram from class. We turn it into something you can feel with your fingers and
              follow with your ears.
            </p>
            <div className="ts-actions">
              <button type="button" className="ts-cta ts-cta-gold" onClick={onTry}>
                Try a diagram
              </button>
              <button type="button" className="ts-cta ts-cta-paper" onClick={onPhoto}>
                Take a photo of a page
              </button>
              <button type="button" className="ts-cta ts-cta-line" onClick={onHow}>
                How it works
              </button>
            </div>
            <VoiceGuideToggle />
            <p className="ts-help">
              Tab moves between buttons. Space repeats the last message. H speaks this help. Escape stops
              speaking. On a touch screen, the first tap says the button name and the second tap presses
              it.
            </p>
          </div>
          <SampleChart />
        </div>
      </section>

      <section className="ts-how" aria-labelledby="ts-how-title">
        <div className="ts-wrap">
          <h2 id="ts-how-title">How it works</h2>
          <p className="ts-intro">Three moves, from a photo to something you can feel.</p>
          <ol className="ts-steps">
            {STEPS.map((step, index) => (
              <li key={step.title} className="ts-card ts-step" style={{ animationDelay: `${index * 0.12}s` }}>
                {step.icon}
                <p className="ts-num" aria-hidden="true">
                  {step.num}
                </p>
                <h3>{step.title}</h3>
                <p>{step.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="ts-who" aria-labelledby="ts-who-title">
        <div className="ts-wrap">
          <h2 id="ts-who-title">Who it&apos;s for</h2>
          <p className="ts-intro">Made for the people who learn, teach, and share diagrams.</p>
          <ul className="ts-audience">
            <li className="ts-card ts-card-gold">
              <h3>Blind and low-vision students</h3>
              <p>Study the picture with sound and touch, at your own pace.</p>
            </li>
            <li className="ts-card ts-card-mint">
              <h3>Teachers</h3>
              <p>Make a tactile version of the page you already teach.</p>
            </li>
            <li className="ts-card ts-card-sky">
              <h3>Schools</h3>
              <p>Share it in class. No login and no password.</p>
            </li>
          </ul>
        </div>
      </section>
    </main>
  );
}
