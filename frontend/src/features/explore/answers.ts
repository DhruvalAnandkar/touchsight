import type { RegionMap } from "../../types";

type Region = NonNullable<RegionMap["regions"]>[number];

const STOP_WORDS = new Set([
  "why",
  "what",
  "whats",
  "how",
  "who",
  "where",
  "when",
  "which",
  "does",
  "do",
  "did",
  "is",
  "are",
  "was",
  "the",
  "a",
  "an",
  "this",
  "that",
  "it",
  "we",
  "our",
  "you",
  "your",
  "have",
  "has",
  "had",
  "for",
  "from",
  "with",
  "about",
  "tell",
  "me",
  "please",
  "there",
  "here",
  "diagram",
  "picture",
  "figure",
  "part",
  "parts",
]);

export function answerQuestion(raw: string, map: RegionMap): string {
  const text = normalize(raw);
  if (!text) return "Ask about a part, or ask why this diagram matters.";
  if (text.includes("list the parts") || text.includes("list parts") || text.includes("name the parts")) {
    return listParts(map);
  }

  const matches = rankedMatches(text, map.regions ?? []);
  const best = matches[0];
  if (best && isSpecificPart(text, best)) {
    return speakPart(best.region, text);
  }

  const group = groupRegions(text, matches);
  if (group.length > 0) return speakGroup(group, text);

  if (best && best.score >= 0.45) return speakPart(best.region, text);

  return speakDiagram(text, map);
}

function isSpecificPart(text: string, match: { region: Region; score: number }): boolean {
  if (match.score < 0.45 || asksCount(text)) return false;
  const name = match.region.name.toLowerCase();
  if (text.includes(name)) return true;
  if (match.score >= 0.99) return true;
  const side = text.match(/\b(left|right|upper|lower|superior|inferior)\b/);
  return side != null && name.includes(side[1]) && match.score >= 0.75;
}

function groupRegions(text: string, matches: { region: Region; score: number }[]): Region[] {
  const top = matches[0];
  if (!top || top.score < 0.45) return [];
  const close = matches.filter((match) => match.score >= top.score - 0.2 && match.score >= 0.45);
  if (close.length >= 2) return close.slice(0, 4).map((match) => match.region);
  if (asksCount(text)) return [top.region];
  return [];
}

function speakPart(region: Region, question: string): string {
  const sentence = partSentence(region);
  if (!sentence) return "That part is on the sheet.";
  if (asksHow(question) && region.description.trim()) return `${region.name.trim()} works like this. ${region.description.trim()}`;
  return sentence;
}

function speakGroup(regions: Region[], question: string): string {
  const jobs = regions.map((region) => partSentence(region)).filter(Boolean);
  if (asksWhy(question) || asksHow(question) || asksCount(question)) {
    return joinSentences([groupLead(regions), ...jobs]);
  }
  return joinSentences(jobs);
}

function groupLead(regions: Region[]): string {
  const stem = sharedStem(regions);
  if (stem.includes("ventric")) {
    return "The heart has two ventricles so one side can send blood to the lungs and the other side can send blood out to the rest of the body.";
  }
  if (stem.includes("atri")) {
    return "The heart has two atria so blood from the body and blood from the lungs can arrive separately before they are pumped on.";
  }
  const label = stem ? pluralWord(stem) : "parts";
  const count = regions.length === 2 ? "two" : String(regions.length);
  return `There are ${count} ${label} so each one can do its own job.`;
}

function partSentence(region: Region): string {
  const name = region.name.trim();
  const description = region.description.trim();
  if (!name && !description) return "";
  if (!description) return name;
  const first = description.split(/\s+/)[0]?.toLowerCase().replace(/[^a-z]/g, "") ?? "";
  if (/^(pump|receive|carry|send|hold|store|move|bring|take|filter|protect|support|connect|separat)/.test(first)) {
    return `The ${name.toLowerCase()} ${lowerFirst(description)}`;
  }
  return `${name}. ${description}`;
}

function sharedStem(regions: Region[]): string {
  const sides = new Set(["left", "right", "upper", "lower", "superior", "inferior"]);
  const lists = regions.map((region) =>
    region.name
      .toLowerCase()
      .split(/\W+/)
      .filter((word) => word.length > 2 && !sides.has(word)),
  );
  const first = lists[0] ?? [];
  return (
    first.find((word) => lists.every((list) => list.some((other) => other === word || other.includes(word) || word.includes(other)))) ??
    ""
  );
}

function pluralWord(stem: string): string {
  if (stem.endsWith("s")) return stem;
  if (stem.endsWith("y")) return `${stem.slice(0, -1)}ies`;
  return `${stem}s`;
}

function asksCount(question: string): boolean {
  return /\b(two|three|four|both|pair|all)\b/.test(question);
}

function speakDiagram(question: string, map: RegionMap): string {
  const subject = map.subject?.trim() || "This diagram";
  const summary = map.summary?.trim() || "";
  const names = (map.regions ?? []).map((region) => region.name.trim()).filter(Boolean);
  const lead = whyLead(subject);
  const partList = names.length > 0 ? `You can ask about ${names.join(", ")}.` : "";

  if (asksWhy(question)) {
    return joinSentences([
      lead || summary || `${subject} is in this picture because it has a job.`,
      partList,
    ]);
  }
  if (asksHow(question)) {
    const steps = (map.regions ?? [])
      .filter((region) => region.name.trim() && region.description.trim())
      .slice(0, 4)
      .map((region) => `${region.name.trim()}: ${region.description.trim()}`);
    return joinSentences([summary || lead || `${subject} is made of the parts on this sheet.`, ...steps]);
  }
  return joinSentences([
    summary || lead || `${subject} is the picture on this sheet.`,
    partList,
  ]);
}

function whyLead(subject: string): string {
  const topic = subject.toLowerCase();
  if (topic.includes("heart")) {
    return "We have a heart so it can keep blood moving. The right side sends blood to the lungs. The left side sends blood that is carrying oxygen out to the rest of the body.";
  }
  if (topic.includes("cell")) {
    return "A cell is a tiny living piece. The parts inside each do one job, and together they keep the cell alive.";
  }
  if (topic.includes("earth") || topic.includes("layer")) {
    return "Earth is built in layers. A thin crust is on the outside, and the inside gets hotter and denser toward the center.";
  }
  if (topic.includes("map")) {
    return "A map shows places next to each other, so you can tell where one area ends and the next begins.";
  }
  return "";
}

function listParts(map: RegionMap): string {
  const names = (map.regions ?? []).map((region) => region.name.trim()).filter(Boolean);
  if (names.length === 0) return "There are no named parts yet.";
  const summary = map.summary?.trim();
  return joinSentences([summary ?? "", `The parts are ${names.join(", ")}.`]);
}

function rankedMatches(text: string, regions: readonly Region[]): { region: Region; score: number }[] {
  return regions
    .map((region) => ({ region, score: nameScore(text, region.name.toLowerCase()) }))
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score);
}

function nameScore(query: string, name: string): number {
  if (!name) return 0;
  if (query.includes(name)) return 1;
  const queryWords = query.split(/\W+/).filter((word) => word.length > 2 && !STOP_WORDS.has(word));
  const nameWords = name.split(/\W+/).filter((word) => word.length > 2);
  if (queryWords.length === 0 || nameWords.length === 0) return 0;
  const hits = nameWords.filter((word) => queryWords.some((token) => wordsClose(token, word))).length;
  return hits / nameWords.length;
}

function wordsClose(queryWord: string, nameWord: string): boolean {
  if (queryWord === nameWord) return true;
  if (queryWord.length >= 4 && nameWord.length >= 4 && (queryWord.includes(nameWord) || nameWord.includes(queryWord))) {
    return true;
  }
  const longest = Math.max(queryWord.length, nameWord.length);
  if (longest < 5) return false;
  return editDistance(queryWord, nameWord) <= 2;
}

function editDistance(left: string, right: string): number {
  const rows = left.length + 1;
  const cols = right.length + 1;
  const grid: number[][] = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
  for (let row = 0; row < rows; row += 1) grid[row][0] = row;
  for (let col = 0; col < cols; col += 1) grid[0][col] = col;
  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const cost = left[row - 1] === right[col - 1] ? 0 : 1;
      let best = Math.min(grid[row - 1][col] + 1, grid[row][col - 1] + 1, grid[row - 1][col - 1] + cost);
      if (row > 1 && col > 1 && left[row - 1] === right[col - 2] && left[row - 2] === right[col - 1]) {
        best = Math.min(best, grid[row - 2][col - 2] + 1);
      }
      grid[row][col] = best;
    }
  }
  return grid[left.length][right.length];
}

function asksWhy(question: string): boolean {
  return /\bwhy\b|\bwhat for\b|\bpurpose\b|\bneed\b|\bimportant\b/.test(question);
}

function asksHow(question: string): boolean {
  return /\bhow\b|\bwork\b/.test(question);
}

function normalize(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/[?.!,]/g, "");
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

function joinSentences(parts: string[]): string {
  return parts
    .map((part) => part.trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}
