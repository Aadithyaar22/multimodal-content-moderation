---
name: Sentinel Noir
colors:
  surface: '#141313'
  surface-dim: '#141313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353434'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c4c7c8'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#8e9192'
  outline-variant: '#444748'
  surface-tint: '#c6c6c7'
  primary: '#ffffff'
  on-primary: '#2f3131'
  primary-container: '#e2e2e2'
  on-primary-container: '#636565'
  inverse-primary: '#5d5f5f'
  secondary: '#c4c7c8'
  on-secondary: '#2d3132'
  secondary-container: '#464a4b'
  on-secondary-container: '#b6b9ba'
  tertiary: '#ffffff'
  on-tertiary: '#2f3131'
  tertiary-container: '#e2e2e2'
  on-tertiary-container: '#636565'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c7'
  on-primary-fixed: '#1a1c1c'
  on-primary-fixed-variant: '#454747'
  secondary-fixed: '#e0e3e4'
  secondary-fixed-dim: '#c4c7c8'
  on-secondary-fixed: '#191c1d'
  on-secondary-fixed-variant: '#444748'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c7'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#454747'
  background: '#141313'
  on-background: '#e5e2e1'
  surface-variant: '#353434'
  pure-black: '#000000'
  deep-grey: '#121212'
  glass-border: rgba(255, 255, 255, 0.12)
  surface-gradient-start: rgba(255, 255, 255, 0.05)
  surface-gradient-end: rgba(255, 255, 255, 0.01)
  anomaly-highlight: '#ffffff'
  negative-adjustment: '#444748'
typography:
  display-lg:
    fontFamily: Anybody
    fontSize: 84px
    fontWeight: '900'
    lineHeight: 80px
    letterSpacing: -0.04em
  display-md:
    fontFamily: Anybody
    fontSize: 48px
    fontWeight: '800'
    lineHeight: 48px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Anybody
    fontSize: 48px
    fontWeight: '900'
    lineHeight: 44px
  headline-lg:
    fontFamily: Noto Serif
    fontSize: 32px
    fontWeight: '400'
    lineHeight: 40px
  headline-md:
    fontFamily: Noto Serif
    fontSize: 24px
    fontWeight: '400'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-lg:
    fontFamily: Anybody
    fontSize: 14px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.08em
  label-md:
    fontFamily: Anybody
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 14px
    letterSpacing: 0.1em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  container-max: 1440px
  margin: 3rem
  gutter: 2rem
  stack-xl: 80px
  stack-lg: 48px
  stack-md: 24px
  stack-sm: 12px
  stack-xs: 4px
---

## Brand & Style

Sentinel Noir is a high-stakes, technical aesthetic designed for critical AI forensics and cybersecurity analysis. The brand personality is authoritative, cold, and precise, evoking the atmosphere of a high-end surveillance terminal. 

The design style is a sophisticated blend of **Minimalism** and **Glassmorphism**, set against a **Pure Black** foundation to maximize contrast and reduce eye strain during deep focus. It utilizes "Spectral Surfaces"—semi-transparent, blurred layers that suggest depth without physical weight. Visual interest is driven by monochromatic textures, subtle "active" shaders, and high-fidelity typography rather than vibrant color. The emotional goal is to provide the user with a sense of absolute clarity and technical mastery over complex data.

## Colors

The palette is strictly monochromatic to maintain a "zero-distraction" environment. 

- **Foundation**: The background is `pure-black` (#000000), providing an infinite canvas that makes white text and glass elements pop.
- **Primary**: White (#ffffff) is reserved for essential data, primary actions, and critical alerts.
- **Secondary/Neutral**: A range of greys (`outline`, `on-surface-variant`) handles metadata and secondary information.
- **Functional Glass**: Backgrounds are not solid; they use low-opacity white fills with heavy backdrop blurs (24px) to create a layered "heads-up display" (HUD) feel.
- **Accents**: Color is avoided. Meaning is conveyed through luminosity (brightness) and stroke weight rather than hue.

## Typography

The system uses a tri-font hierarchy to distinguish between data types:

1.  **Anybody (Labels & Display)**: Used for high-impact numbers and technical labels. Its variable width and bold weights give the UI a modern, sporty, and aggressive technical edge.
2.  **Noto Serif (Headlines)**: Introduces a layer of "editorial authority." Using a serif for report titles and headers makes the AI's findings feel like a formal document or a definitive record.
3.  **Inter (Body)**: The workhorse for long-form reasoning and UI controls. Chosen for its extreme legibility and neutral character.

All labels should be forced uppercase with increased letter-spacing to reinforce the "technical terminal" look.

## Layout & Spacing

The system uses a **Bento Grid** philosophy contained within a fixed-width `1440px` max container.

- **Grid**: A 12-column layout on desktop, transitioning to a single-column stack on mobile. Elements should be grouped into logical "glass panels" that span various column widths (e.g., 5-col for evidence, 7-col for analysis).
- **Rhythm**: A strict vertical stack scale ensures consistent breathing room. `stack-md` (24px) is the standard gap between related items, while `stack-lg` (48px) separates major sections.
- **Safe Areas**: Generous outer margins (3rem) ensure the UI feels expansive and premium, rather than cramped.

## Elevation & Depth

Depth is achieved through **translucency and refraction** rather than shadows. 

- **Level 0 (Floor)**: Pure Black (#000000) with a subtle, monochromatic animated shader (smoke/fluid motion) to add "life" to the void.
- **Level 1 (Panels)**: The `glass-panel` uses a 135-degree linear gradient from 5% to 1% white opacity. A `24px` backdrop blur is mandatory to separate the panel from the background shader.
- **Level 2 (Active/Hover)**: On hover, panel opacity increases slightly (up to 12%), and the border brightness doubles.
- **Borders**: Every panel must have a `1px` solid border (`glass-border`) to define its silhouette against the black.

## Shapes

The shape language is **Technical-Soft**. 

- **Default**: `0.25rem` (4px) for buttons and small containers to maintain a sharp, precise look.
- **Panels**: `0.5rem` (8px) for major bento-grid cards.
- **Interactive Circles**: Full rounding (9999px) is reserved exclusively for icon-only action buttons (like the back button) to make them stand out from the rectangular layout.
- **Indicators**: Use sharp, 0px-radius left borders for "impact factors" or status bars to imply a "data-slice" aesthetic.

## Components

- **Buttons**:
    - *Primary*: Solid white background, black text, no border. Heavy label font.
    - *Secondary/Glass*: Transparent with a 1px `outline` border. White text. Subtle hover state that fills the background with `surface-container-highest`.
- **Glass Panels**: The core container. Must include a header area with a `1px` bottom border and `uppercase label-lg` text.
- **Data Bars**: Use high-contrast ratios. The background of a bar is `surface-container-highest`, and the fill is `primary` (white).
- **Impact Factors (SHAP)**: High-luminosity fills (`rgba(255,255,255,0.15)`) with a `2px` solid white left-accent border for positive factors; darkened backgrounds for negative factors.
- **Imagery**: All evidence images should have a grayscale filter applied by default, reverting to color only on hover to maintain the monochromatic aesthetic of the dashboard.
- **Decision Bar**: A fixed footer with a heavy glass blur, providing a persistent "Action Zone" at the bottom of the viewport.