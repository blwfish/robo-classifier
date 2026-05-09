/**
 * COVA design tokens — TypeScript module
 * Source of truth: ./tokens.json. Mirror the JSON when changing values.
 *
 * Two themes, same component code. Apply by setting `data-theme` on the
 * app shell root, OR consume `tokens.modern` / `tokens.deco` directly
 * for runtime-computed styles (charts, canvas, native components).
 */

export type ThemeName = "modern" | "deco";

export interface ThemeColors {
  bg: string;
  neutral: {
    "00": string;
    "05": string;
    "10": string;
    "15": string;
    "20": string;
    "25": string;
    "40": string;
    "70": string;
    "95": string;
  };
  line: string;
  lineStrong: string;
  accent: string;
  accentSoft: string;
  accentQuiet: string;
  accentInk: string;
  status: {
    pos: string;
    neg: string;
    info: string;
  };
  text: string;
  text2: string;
  text3: string;
}

export interface Theme {
  name: ThemeName;
  color: ThemeColors;
  fontDisplay: string;
  radius: { 1: string; 2: string };
}

// ---------- Color ----------------------------------------------------------

export const modernColor: ThemeColors = {
  bg: "oklch(0.14 0.005 250)",
  neutral: {
    "00": "oklch(0.10 0.004 250)",
    "05": "oklch(0.14 0.005 250)",
    "10": "oklch(0.17 0.005 250)",
    "15": "oklch(0.20 0.005 250)",
    "20": "oklch(0.24 0.005 250)",
    "25": "oklch(0.28 0.005 250)",
    "40": "oklch(0.40 0.005 250)",
    "70": "oklch(0.70 0.005 250)",
    "95": "oklch(0.95 0.004 250)",
  },
  line:        "oklch(1 0 0 / 0.07)",
  lineStrong:  "oklch(1 0 0 / 0.12)",
  accent:      "oklch(0.80 0.14 75)",
  accentSoft:  "oklch(0.80 0.14 75 / 0.18)",
  accentQuiet: "oklch(0.80 0.14 75 / 0.08)",
  accentInk:   "oklch(0.18 0.05 75)",
  status: {
    pos:  "oklch(0.74 0.14 145)",
    neg:  "oklch(0.66 0.18 25)",
    info: "oklch(0.72 0.10 235)",
  },
  text:  "oklch(0.95 0.004 250)",
  text2: "oklch(0.70 0.005 250)",
  text3: "oklch(0.55 0.005 250)",
};

export const decoColor: ThemeColors = {
  bg: "oklch(0.12 0.005 65)",
  neutral: {
    "00": "oklch(0.08 0.004 65)",
    "05": "oklch(0.12 0.005 65)",
    "10": "oklch(0.16 0.005 65)",
    "15": "oklch(0.19 0.006 65)",
    "20": "oklch(0.23 0.006 65)",
    "25": "oklch(0.28 0.006 65)",
    "40": "oklch(0.40 0.006 65)",
    "70": "oklch(0.72 0.008 65)",
    "95": "oklch(0.94 0.010 65)",
  },
  line:        "oklch(0.78 0.06 80 / 0.14)",
  lineStrong:  "oklch(0.78 0.06 80 / 0.32)",
  accent:      "oklch(0.78 0.07 80)",
  accentSoft:  "oklch(0.78 0.07 80 / 0.16)",
  accentQuiet: "oklch(0.78 0.07 80 / 0.08)",
  accentInk:   "oklch(0.16 0.03 60)",
  status: {
    pos:  "oklch(0.74 0.10 145)",
    neg:  "oklch(0.66 0.14 25)",
    info: "oklch(0.72 0.07 235)",
  },
  text:  "oklch(0.94 0.010 65)",
  text2: "oklch(0.72 0.008 65)",
  text3: "oklch(0.55 0.008 65)",
};

// ---------- Theme objects --------------------------------------------------

export const modern: Theme = {
  name: "modern",
  color: modernColor,
  fontDisplay: '"Inter Tight", system-ui, sans-serif',
  radius: { 1: "2px", 2: "4px" },
};

export const deco: Theme = {
  name: "deco",
  color: decoColor,
  fontDisplay: '"Poiret One", "Italiana", serif',
  radius: { 1: "0px", 2: "0px" },
};

export const themes = { modern, deco } as const;

// ---------- Theme-independent ---------------------------------------------

export const font = {
  ui:   '"Inter Tight", system-ui, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, monospace',
} as const;

export const fontSize = {
  micro:    "11px",
  small:    "12px",
  body:     "13px",
  bodyLg:   "14px",
  title:    "18px",
  display1: "28px",
  display2: "38px",
  display3: "48px",
} as const;

export const fontWeight = {
  regular:  400,
  medium:   500,
  semibold: 600,
} as const;

export const letterSpacing = {
  ui:         "-0.005em",
  capsModern: "0.08em",
  capsDeco:   "0.16em",
} as const;

export const spacing = {
  1:  "4px",
  2:  "8px",
  3:  "12px",
  4:  "16px",
  5:  "20px",
  6:  "24px",
  7:  "32px",
  8:  "40px",
  9:  "56px",
  10: "72px",
} as const;

export const duration = {
  fast:    "80ms",
  default: "160ms",
  slow:    "240ms",
} as const;

export const easing = {
  default: "cubic-bezier(0.2, 0.7, 0.2, 1)",
} as const;

export const layout = {
  topBar:      "56px",
  statusBar:   "28px",
  filmstrip:   "140px",
  sidebar:     "240px",
  inspector:   "340px",
  gridCellMin: "220px",
} as const;

// ---------- Helpers --------------------------------------------------------

/**
 * Get the active theme by name. Useful for code that needs to compute
 * styles outside the CSS cascade — canvas drawing, chart libraries,
 * native bridge, etc.
 */
export function getTheme(name: ThemeName): Theme {
  return themes[name];
}

/**
 * Resolve `data-theme` from a DOM node. Returns "modern" if not set.
 */
export function readActiveTheme(root: HTMLElement = document.documentElement): ThemeName {
  const v = root.getAttribute("data-theme");
  return v === "deco" ? "deco" : "modern";
}

/**
 * Set the theme on a root element. Persists to localStorage by default —
 * pass `{ persist: false }` to skip.
 */
export function setActiveTheme(
  name: ThemeName,
  root: HTMLElement = document.documentElement,
  opts: { persist?: boolean } = {},
): void {
  root.setAttribute("data-theme", name);
  if (opts.persist !== false) {
    try { localStorage.setItem("cova.theme", name); } catch {}
  }
}
