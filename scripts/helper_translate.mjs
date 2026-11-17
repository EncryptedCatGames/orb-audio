// Run the site's keyword helper (site/app.js `translate`) outside the browser.
// stdin: one JSON string per line; stdout: one JSON {caption, notes} | null per line.
import { readFileSync } from "node:fs";
const src = readFileSync(new URL("../site/app.js", import.meta.url), "utf8");
const a = src.indexOf("const PAN_TOKENS");
const b = src.indexOf("function wavBlob");
const translate = new Function("const MAX_EVENTS = 8; const clamp = (x, a, b) => Math.min(b, Math.max(a, x)); const round2 = (x) => Math.round(x * 100) / 100; const words = (s) => String(s).replace(/_/g, \" \");\n" + src.slice(a, b) + "\nreturn translate;")();
const lines = readFileSync(0, "utf8").split("\n").filter(Boolean);
for (const l of lines) {
  const t = translate(JSON.parse(l));
  console.log(JSON.stringify(t ? { caption: t.caption, notes: t.notes } : null));
}
