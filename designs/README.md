# Handoff: COVA — Photographer's Catalog & Develop App

## Overview

COVA is a desktop-class photo catalog + develop app for working photographers — think Capture One / Photo Mechanic territory. The app is built around a **library** (folder tree + smart collections), a **contact-sheet grid** of photo cells, an **inspector** stack of develop adjustments, and a **filmstrip** for in-context navigation.

The design comes in **two interchangeable personalities** driven by the *same* structural code:

- **Modern** — cool-neutral monochrome chrome with a single amber accent. Sharp 0–4px radii. The default, intended for daily use. Spiritually adjacent to Capture One Pro.
- **Tasteful Deco** — warm bone + ebony palette, champagne-brass accent, Poiret One display type, double-rule section breaks, stepped corners on toolbars and panel headers. Same components, same layout — only tokens and a handful of chrome flourishes change.

A third extreme **Full Deco** showcase is included for reference (sunbursts, French copy, ornament-heavy) but is **not** the shipping target — it's there to anchor the high end of the personality spectrum.

## About the Design Files

The HTML files in `designs/` are **design references** — high-fidelity prototypes built in plain HTML/CSS + a tiny bit of React-via-Babel for component composition. They show the intended look, layout, copy, and interaction states.

**They are not production code to lift directly.** Your job is to **recreate these designs in the target codebase's environment** using its established framework, component library, and styling conventions. If no environment exists yet, pick the most appropriate stack — for a desktop-class photo app this likely means Tauri / Electron + React, or a native shell (SwiftUI / Qt) — and implement the designs there.

The CSS tokens in `tokens.css` are extracted cleanly and **are** intended to be used directly: drop them into the codebase as the source of truth for color, type, spacing, and motion.

## Fidelity

**High-fidelity.** All colors, type sizes, spacing, border treatments, and component states are intentional and final-ish. The contact-sheet grid, photo cell anatomy, slider behavior, and inspector stack should be implemented pixel-faithful to the prototype.

The only intentionally rough parts:
- **Photo content** — the grids use `picsum.photos` placeholders. The real app will display the user's catalog.
- **Iconography** — toolbar icons are simple inline SVGs. The real app should use a proper icon set (Lucide, Phosphor, or a custom set) at the same visual weight (~1.2–1.4px stroke at 14×14).
- **Empty / loading / error states** — not yet designed. Flag for follow-up.

## Architecture: One Codebase, Two Personalities

This is the core insight to preserve in implementation: **the personality is not a fork — it's a token swap.**

In the prototype this is achieved with a single attribute on a wrapping element:

```html
<div data-style="modern"> ...identical component tree... </div>
<div data-style="deco">   ...identical component tree... </div>
```

In the production app this should be:

```html
<html data-theme="modern"> <!-- or "deco" -->
```

The `data-theme` attribute switches a block of CSS custom properties (see `tokens.css`). Every component reads tokens via `var(--*)` and **never** hard-codes a color, font, or radius. The handful of chrome differences that are not pure-token (Deco's double rules, stepped corners, sunburst brand mark) are gated by `[data-theme="deco"]` selectors, not by branching component code.

This means:
- One `<Button>` component, one `<Slider>` component, one `<PhotoCell>` component.
- Adding a third personality later (e.g., a "Print" theme for export-focused workflows) is a tokens-file change, not a component rewrite.
- A user-facing theme toggle is a one-line attribute set.

## Screens / Views

The prototypes cover four scopes, each shown side-by-side in both personalities:

### 1. Identity
- **Purpose** — establish the personality at a glance: brand mark, color ramp, type specimens.
- **Layout** — header with brand mark + eyebrow + display title; horizontal color ramp (10 swatches); type specimen stack (display / body / mono / micro); footer line listing fonts and accent value.
- **Components** — `<Mark>` (brand mark, two variants), color swatch row, type specimen.

### 2. Components (chrome)
- **Purpose** — toolbar, buttons, inputs, chips, sliders, icon buttons in their canonical resting states.
- **Layout** — full-width toolbar at top; two-column body: chrome controls on left, an adjustments panel on right (320px fixed).
- **Components** — `<Toolbar>`, `<Button>` (primary / default / ghost), `<InputGroup>` (icon + input + addon), `<Chip>` (default / accent / dashed-add), `<IconButton>`, `<Slider>` with center-origin and "edited" amber/brass state, `<Panel>` with grouped header.

### 3. Photo Cell + Grid
- **Purpose** — the most-pixel-touched atom in the app, in all its states.
- **Layout** — top row: 4 cells showing default / selected+pick / rejected / processing states. Below: a 4-column contact sheet with 8 cells in mixed states.
- **Components** — `<PhotoCell>` is the workhorse. Anatomy:
  - 16:9 image fill (cover)
  - **Selection** — Modern: 2px amber inset outline; Deco: hairline brass outline + brass corner brackets
  - **Color label** — Modern: 3px stripe along the top edge in label color; Deco: 4px stripe with a hairline brass underline
  - **Pick / Reject flags** — small icon top-right corner: ◆ for pick (accent color), ✕ for reject (cell goes 40% opacity + desaturate filter)
  - **Star rating** — bottom-left, 1–5 ★ glyphs in accent color, dimmed to text-3 when empty
  - **Badge** — top-left, processing/queued status as a small pill
  - **Hover** — full cell brightness lifts +4%, picks/rejects/rating glyphs reveal at full opacity (when not always-shown)

### 4. App Shell (full integration)
- **Purpose** — show all components composed into the working app at correct proportions.
- **Layout** — CSS grid:
  ```
  ┌─────────────────────────────────────┐  56px  top bar
  ├──────┬──────────────────────┬───────┤
  │      │                      │       │
  │ 240  │        flex-1        │  340  │  flex-1 height
  │ side │     contact sheet    │ insp  │
  │      │                      │       │
  ├──────┴──────────────────────┴───────┤  140px filmstrip
  ├─────────────────────────────────────┤  28px status bar
  └─────────────────────────────────────┘
  ```
- **Components** — sidebar tree (collapsible folders + smart collections with counts), contact sheet (`repeat(auto-fill, minmax(220px, 1fr))`), inspector (file metadata + adjustments + keywords + export CTA), filmstrip (horizontal-scroll thin row of cells, 80px tall), status bar (mono-font pipeline of catalog stats).

## Interactions & Behavior

### Photo cell
- **Click** — select. Click + Shift = range select. Click + ⌘/Ctrl = toggle.
- **P** — toggle pick. **X** — toggle reject. **0–5** — rating. **6–9** — color label.
- **Double-click** — open in Loupe view (single-image review; not in current prototype scope).
- Picks/rejects update **immediately** in the UI; persistence is async with optimistic state.

### Adjustments slider
- Bipolar, center origin. Dragging shows live numeric value to the right.
- "Edited" state — when value ≠ 0, the value text and fill turn accent color, and a small dot appears next to the slider label. Reset on double-click.
- ⌘/Ctrl + drag → fine-grained (×0.1).

### Toolbar tabs
- `aria-pressed="true"` on active tab. Library / Loupe / Compare. Hotkeys G / E / C.

### Theme toggle
- Setting `<html data-theme="modern|deco">` swaps the entire app. Animate via `transition: background-color var(--dur-3) var(--ease), border-color var(--dur-3) var(--ease)` on root tokens — but **not** on every leaf, or it will flash.

### Search
- ⌘F focuses. Live filter on filename / keyword / lens / camera as the user types. Filters compose with chips (every active chip is an AND).

## State Management

State is split by lifetime:

| Scope | Examples | Suggested store |
|---|---|---|
| **Catalog** | Folder tree, photo metadata, ratings, picks, rejects, color labels, keywords | Local SQLite (server-of-truth), in-memory cache, IndexedDB mirror for renderer |
| **Develop** | Per-photo adjustment stack — exposure, contrast, highlights, shadows, whites, blacks, etc. | Versioned per photo; undo/redo stack of last N edits |
| **Session** | Selection, active panel, sort order, filter chips, current view (Library/Loupe/Compare) | Zustand / Redux / Pinia / Riverpod — whatever the codebase uses |
| **App** | Theme (`modern`/`deco`), density, last-opened catalog, panel widths | localStorage / app-data dir |

The design assumes **selection** is a multi-photo concept (`Set<photoId>`) — every action that operates on "the photo" should operate on the selection.

## Design Tokens

All values live in `tokens.css`. Highlights:

### Color (oklch)

**Modern**
- bg: `oklch(0.14 0.005 250)` — cool near-black
- ramp: 10-step neutral from `oklch(0.10 …)` to `oklch(0.95 …)`, hue 250 (cool)
- accent: `oklch(0.80 0.14 75)` — amber
- semantic (status only): pos `oklch(0.74 0.14 145)`, neg `oklch(0.66 0.18 25)`, info `oklch(0.72 0.10 235)`
- text: 95% / 70% / 55% lightness for primary / secondary / tertiary

**Deco**
- bg: `oklch(0.12 0.005 65)` — warm near-black at hue 65
- ramp: 10-step warm-leaning
- accent: `oklch(0.78 0.07 80)` — champagne brass (lower chroma than amber — important: brass should feel restrained, not gold)
- text: warmer near-white at 94% / 72% / 55%

**Both** use the same semantic R/G/B status triplet with slightly desaturated chroma in Deco for cohesion.

### Typography
- **UI** — Inter Tight 400/500/600 (both themes)
- **Mono** — JetBrains Mono 400/500 (both themes — used for EXIF, filenames, numeric values, status bar)
- **Display** — Modern: Inter Tight (no separate display face). Deco: Poiret One with Italiana fallback for elegant, low-contrast strokes.

Sizes (theme-independent):
- micro/mono labels — 11–12px
- body/UI — 13–14px
- panel titles / eyebrows — 11px tracked-caps (`letter-spacing: 0.08em` modern, `0.16em` deco)
- display headers — 28–48px depending on context

### Spacing
4px grid. Tokens `--sp-1` … `--sp-10` mapped to 4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 56 / 72 px.

### Border radius
- Modern: `--r-1: 2px; --r-2: 4px;` — sharp but not severe.
- Deco: `--r-1: 0px; --r-2: 0px;` — fully orthogonal. Deco compensates for the lack of softening with hairline brass borders and double rules.

### Motion
- `--dur-1: 80ms` — hover, press
- `--dur-2: 160ms` — panel reveal, dropdown
- `--dur-3: 240ms` — heavier transitions (theme swap, large panel slide)
- `--ease: cubic-bezier(0.2, 0.7, 0.2, 1)` — universal "snap" easing

## Assets

- **Photos in mockups** — `picsum.photos/seed/<id>/<w>/<h>` placeholders. Replace with real catalog thumbnails in production.
- **Brand mark** — drawn in CSS for the prototype (Modern: solid square; Deco: square with sunburst halo). Production should use SVG; both variants are simple enough to recreate.
- **Icons** — recommend [Lucide](https://lucide.dev/) for the modern theme. For Deco, the same icon set works at 1.2px stroke; consider sourcing/commissioning a small set of Deco-styled flag icons (chevron-pick, etc.) if a more bespoke feel is wanted.
- **Fonts** — Google Fonts: Inter Tight, JetBrains Mono, Poiret One, Italiana. Self-host in production for performance + offline use.

## Files

In `designs/`:

- **`COVA - Modern vs Deco.html`** — primary handoff. Side-by-side comparison of both personalities across Identity / Components / Photo cell / App shell. **Use this as the visual source of truth.**
- **`COVA Design System.html`** — long-scroll reference doc of the Modern direction in full (color scales, every component variant, photo-specific atoms, full app shell). Useful when you need to see *all* states of a single component, not just the side-by-side highlights.
- **`COVA - Full Deco.html`** — extreme Deco showcase with sunbursts, French copy, heavy ornament. **Reference only — do not implement.** Useful for understanding the high end of the personality dial; the actual ship target is Tasteful Deco as shown in the side-by-side.

In bundle root:

- **`tokens.css`** — clean, drop-in design tokens for both themes. Production code should consume this.
- **`README.md`** — this file.

## Open questions for follow-up

These are intentionally not yet designed and should come back to the design conversation:

1. **Loupe view** (single-image review) — keyboard nav, before/after split, zoom + pan.
2. **Compare view** — N-up comparison with synced zoom.
3. **Mask / radial / linear gradient** overlays in the inspector.
4. **Tone curve** + **HSL wheels** — currently only sliders exist.
5. **Export dialog** — destination, format, sizing, watermark, naming pattern.
6. **Empty / loading / error states** — first-launch, importing, GPU/disk stalls, offline catalog.
7. **Light theme** — present in the v1 design system doc as a secondary mode, but not yet refined for parity with the dark default.
