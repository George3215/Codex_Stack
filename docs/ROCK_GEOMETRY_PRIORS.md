# Rock Geometry Priors For Lunar Landmark Stacking

This note records the geometry prior used by the procedural rock generator. It replaces the earlier "spiky angular polyhedron" generator, which produced unrealistic isolated protrusions.

## Literature Basis

- Zingg-style form classes use the long, intermediate, and short axes of a convex envelope to distinguish equant, prolate, oblate, and bladed particles. For this project we keep the same axis-ratio logic but avoid extremely oblate slabs because the user explicitly rejected very flat stones.
- Sedimentology separates particle form, roundness/angularity, and surface texture. This is useful for generation because a rock can be equant but angular, elongated but subangular, or fractured but still free of needle-like spikes.
- Wadell/Krumbein/Powers-style roundness concepts describe corner sharpness, not isolated needle protrusions. Natural angular clasts may have sharp corners and broad fracture faces, but a single-vertex spike is not a good stone prior.
- Lunar regolith and rock fragments are mechanically fragmented by impacts. Fine lunar dust may be sharp and jagged, but stackable cobble/boulder-scale stones should be modeled as fractured clasts with broad facets and limited local protrusion.
- Abrasion and transport studies show that rock particles tend toward convexity as edges round; even when angular, the dominant shape is controlled by broad axis ratios and fracture faces rather than thin spikes.

Useful sources:

- Zingg axis classes: https://en.wikipedia.org/wiki/Equidimensional_%28geology%29
- Sediment form/roundness/texture distinction and Krumbein formulas: https://en.wikipedia.org/wiki/Sediment
- Roundness/angularity categories: https://en.wikipedia.org/wiki/Roundness_%28geology%29
- Domokos et al. 2013/2014, curvature-driven river-rock rounding: https://arxiv.org/abs/1311.6574
- Lunar regolith formation and sharp fine particles: https://en.wikipedia.org/wiki/Lunar_regolith
- Regolith, fractured bedrock, and jagged agglutinates: https://en.wikipedia.org/wiki/Regolith

## Generator Requirements

Allowed:

- Low- to medium-resolution faceted polyhedra.
- Broad convex lobes.
- Broad fracture-plane chips.
- Moderate angular corners.
- Equant, subangular block, wedge, fractured, and elongated clast families.

Rejected:

- Isolated point spikes.
- Very flat slab stones.
- Smooth latitude-longitude spheres.
- High-frequency radial noise that creates needle-like protrusions.
- Any cluster with high `spike_score`.

## Current Procedural Classes

The current generator emits these source kinds:

- `equant_clast`: compact, near-equal axes, low roughness, mild chips.
- `subangular_block`: block-like, moderate axis variation, several broad chips.
- `wedge_clast`: broad wedge tendency, useful as a base/support candidate.
- `fractured_clast`: more fracture planes and rougher broad facets.
- `elongated_clast`: moderately prolate, not a needle, useful only when placement is well supported.

## Quantitative Checks

Every generated rock stores:

- Axis ratios: `elongation = longest / middle`, `flatness = middle / shortest`.
- `sphericity`: surface area versus equal-volume sphere proxy.
- `roughness`: radial standard deviation proxy.
- `angularity`: mean adjacent-face normal angle.
- `spike_score`: maximum local radial excess over neighboring vertices.
- `compactness`: volume / bounding-box volume.

Operational constraints:

- `shortest / middle >= 0.62` to avoid very flat stones.
- `spike_score < 0.18` is expected; values above this should be classified as `spiky_reject` and excluded before formal simulation.
- Shape perturbations are broad and smoothed across mesh neighbors.
- Fracture operations move broad caps inward along a plane; they never push single vertices outward.

## Classification Policy

Cluster names are assigned from measured geometry, not just the procedural source label:

- `spiky_reject`: high local spike score.
- `elongated_clast`: high elongation.
- `wedge_or_broad_clast`: broad/wedge-like axis ratio with adequate compactness.
- `fractured_clast`: low compactness.
- `angular_clast`: high angularity or roughness.
- `equant_clast`: high sphericity and low roughness.
- `subangular_block`: default blocky class.

The stacking planner should prefer `wedge_or_broad_clast` and `subangular_block` early, use `equant_clast` as filler, delay `fractured_clast` and `elongated_clast`, and reject or heavily penalize `spiky_reject`.
