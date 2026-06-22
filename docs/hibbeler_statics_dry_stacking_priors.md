# Hibbeler Statics and Mechanics Priors for Dry Stacking

Source file:

`D:\MoonStack\Asset\Statics and Mechanics of Materials SI (Russell C. Hibbeler) (z-lib.org).pdf`

Important limitation:

The local PDF is a scanned image PDF. `pypdf` reports 792 pages, but direct text extraction returns empty page text. No local OCR executable was available (`tesseract`, `pdftotext`, and ImageMagick were not found). Therefore this note does not quote or claim OCR-verified passages from the PDF. It maps the standard Hibbeler statics and mechanics topics in that text to dry-stone stacking heuristics and treats them as engineering hypotheses to test.

## Relevant Mechanics Principles

### 1. Static equilibrium

For every placed stone and for the whole wall segment, the net force and net moment should be close to zero after settling. In dry stacking this means a stone is not acceptable merely because its center reaches a target slot; it must settle with low residual velocity, low post-hold drift, and a load path into the stones below.

Dry-stacking heuristic:

- Prefer candidates with low final velocity.
- Reject or penalize stones that disturb already placed stones.
- Score final placement after MuJoCo settling, not just initial pose.

### 2. Resultant and center-of-gravity projection

A body is stable against tipping when its weight resultant projects inside the effective support region. For irregular stones, we only approximate this with mesh bounding features and settled contact layout.

Dry-stacking heuristic:

- For upper courses, minimize `support_balance_error_m`: the horizontal distance between the stone center and the local support centroid.
- Do not chase target x-y location if it moves the center outside the local support region.
- Tie stones and cap stones require stricter balance margins because they control the stability of adjacent courses.

### 3. Moment and overturning

Overturning happens when the destabilizing moment from eccentric weight exceeds the restoring moment from support reactions. A high wall needs a wider base, batter, and interlocking/tie stones to increase resisting moment.

Dry-stacking heuristic:

- Use a battered two-face wall for tall structures rather than a one-plane thin wall.
- Widen and complete the lower courses before adding high courses.
- Penalize high-course candidates with large support-balance eccentricity even if target error is small.

### 4. Friction and sliding

Dry stacking has no adhesive bond; sliding resistance is controlled by normal force and friction. In the simulation this is approximated through post-settle drift, velocity, and contact/support metrics.

Dry-stacking heuristic:

- Penalize horizontal drift after placement and after final hold.
- Prefer broad, angular, moderately rough stones with multiple contact opportunities.
- Avoid elongated or spiky stones that concentrate contact and induce rolling/sliding.

### 5. Bearing and distributed loading

Load transfer through a tiny contact area produces high local bearing stress and unstable rocking. We do not yet compute true contact area, so use support overlap and support contact count as proxies.

Dry-stacking heuristic:

- Prefer candidates with higher `support_overlap` and `support_contact_count`.
- Add `bearing_pressure_proxy = mass / support_proxy` and penalize high values.
- Use compact/blocky stones in lower courses; save tall stones for positions with adequate support.

### 6. Factor of safety

The planner should not be a binary accept/reject rule only. It should rank candidates by stability margin and choose the one with the largest practical margin.

Dry-stacking heuristic:

- Use a weighted score combining target error, support overlap, support balance, post-placement velocity, disturbance, and bearing proxy.
- Record both successes and failures; failed candidates are useful data for improving the safety margins.

## Implemented Strategy

The strategy `statics_wall` was added as an experimental variant of `literature_wall`.

It adds or emphasizes:

- online stone selection;
- low-clearance quasi-static placement;
- deterministic wall-oriented yaw candidates;
- support centroid tracking in upper courses;
- `support_balance_error_m` in candidate metrics;
- `bearing_pressure_proxy` in candidate metrics;
- stronger support/contact weighting;
- stronger disturbance and residual velocity penalties;
- stone selection that favors compact, blocky, moderate-height stones and penalizes high spike score.

## Not Yet Solved

This is still a heuristic approximation, not a true rigid-body stability proof.

Missing pieces:

- true contact patch geometry;
- actual support polygon extraction from MuJoCo contacts;
- explicit friction cone margin;
- course-repair planner that redoes failed lower-course slots before moving upward;
- load-path analysis for multi-course walls;
- learning from accumulated failed candidates.

## Experimental Rule Going Forward

For high-wall claims, a case must not be counted as success unless it satisfies:

- enough placed stones;
- enough visible courses;
- height threshold;
- wall footprint threshold;
- low y spread and low outlier count;
- low drift and velocity after final hold;
- front/top RGB and depth images showing wall-like structure rather than a mound or column.

If a result is tall but column-like, it must be logged as a column/core failure case, not as a wall.
