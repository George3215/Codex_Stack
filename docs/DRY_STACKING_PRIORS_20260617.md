# Dry Stacking Priors for Height Experiments

This note converts dry-stone walling practice and local research papers into priors for the lunar rock-stacking simulator.

## Sources Reviewed

- The Stone Trust, dry stone wall resource and specification pages:
  - `https://thestonetrust.org/resource-information/`
  - `https://thestonetrust.org/stone-wall-design-and-specifications/`
- UNESCO intangible heritage page for dry stone walling:
  - `https://ich.unesco.org/en/RL/art-of-dry-stone-walling-knowledge-and-techniques-01393`
- General dry-stone construction terminology and layout reference:
  - `https://en.wikipedia.org/wiki/Dry_stone`
- Local papers in `D:\MoonStack\Asset\Papers`, especially Furrer 2017, Johns 2020, Liu 2018/2021, Liu and Napp 2023, and Menezes 2021.

## Construction Priors

Dry stacking should not be modeled as a vertical line of rocks. A high wall needs:

- Foundation/base stones:
  - largest, broadest, most stable stones;
  - low spike score;
  - high compactness;
  - enough mass and support footprint to carry upper courses.
- Two faces or thickness:
  - even a "single wall segment" should have front and back faces;
  - the wall should taper inward with height.
- Bonding and broken joints:
  - upper stones should bridge joints below;
  - repeated vertical seams should be avoided.
- Through/tie stones:
  - some stones should connect front and back faces;
  - moderate elongation is useful here, but extreme elongation and spike-like geometry should still be penalized.
- Hearting/fill:
  - small stones can fill voids, but simulation currently lacks a dedicated small-hearting generator;
  - the present proxy is the `tie` slot role and small middle stones.
- Coping/cap stones:
  - top stones should be stable, not too heavy, and should lock upper stones rather than maximize height alone.
- Batter:
  - target slots should move inward with course height;
  - this reduces overturning risk and improves high-wall stability.

## Position Priors

| Position | Preferred stones | Penalize |
|---|---|---|
| Base | large subangular blocks, broad wedges, compact equant stones | spike score, very high elongation, very high flatness if it creates slab behavior |
| Middle face | compact equant/subangular stones with moderate size | extreme elongation, sharp protrusions, low support overlap |
| Tie/through | moderate elongated or wedge stones spanning wall depth | spike score, very thin/flat stones, extreme elongation |
| Cap/coping | smaller compact stones with low residual motion | large heavy stones, unstable elongated stones, high target error |
| Column lower courses | multi-stone ring/tripod support | one-stone vertical line, high slenderness |
| Column upper courses | compact stones with small offsets | rigid exact centerline when contact geometry is bad |

## Simulation Translation

Implemented after this note:

- `tall_wall_thick_v1`:
  - 7 courses;
  - front and back faces;
  - inward batter;
  - tie slots on lower/middle courses;
  - still a single wall segment, not a four-wall enclosure.
- `tall_pillar_v3`:
  - 8 visible courses;
  - multi-stone lower support courses;
  - tapered toward single cap stones.
- `dry_wall` strategy:
  - strong preference for compact/supportive base stones;
  - moderate allowance for elongated tie stones;
  - strong penalty for spike score and excessive flatness/elongation;
  - candidate pose score emphasizes support overlap and low residual velocity.
- `pillar_bonded` strategy:
  - avoids the failed rigid single-centerline behavior;
  - uses multi-stone support rings in lower courses;
  - prefers compact and height-contributing stones only when support remains good.

## Current Empirical Lessons

From `20260617_height_push_v3_smoke`:

- `tall_wall_v3` reached 7 visible courses and up to about 0.346 m, but strict success was 0/8.
- The main wall failure mode was `missed_target` in middle/upper courses, not total collapse.
- A single-line wall is the wrong abstraction for height; adding thickness and tie stones is the next required step.
- `single_column_v3` reached 8 visible courses in several runs, but true stack height stayed around 0.17-0.23 m.
- For columns, a one-rock-per-course vertical line is too contact-sensitive; a multi-stone lower support column is more defensible.

## Next Experiment

Run a low-candidate smoke test:

- `tall_wall_thick_v1`
- `tall_pillar_v3`
- strategies: `dry_wall,pillar_bonded,random_order`
- Earth and Moon gravity
- record success, height, visible courses, failure role, source rock kind, and cluster label.
