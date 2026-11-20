# The LucidFace method

LucidFace restores faces that are **out of focus** in interview and talking-head video: the camera
focused on the wall, the subject is soft for the whole take. It is a method more than a model:
the pieces are existing, credited components, put together around five rules we learned the hard way
on real client footage. Each rule says what we observed, what LucidFace does about it, and how the
effect is measured.

## Rule 1: diagnose the blur before choosing a tool

Defocus is not low resolution. A defocused face at 1080p has plenty of pixels; what it lacks is
high-frequency content, removed by the lens's disk-shaped point-spread function. Video
super-resolution models are trained to undo downscaling and compression, and on defocus they mostly
leave the image as it is.

*What we saw:* on a 27-minute defocused interview, SeedVR2 (3B, 7B and 7B-sharp, with the input
pre-shrunk 1x to 4x) only slightly crisped the eyes. A face prior (CodeFormer) rebuilt the eyes, brows
and lips.

*What LucidFace does:* face-prior restoration for the face, and, optionally, a model trained on real
lens defocus (Restormer, DPDD) for hair, hands and clothes. Pass 1 gives every frame a
scale-independent **focus score** (high- to mid-frequency energy on a 256 px aligned face), so shots
that are already sharp can be left untouched (`--gate`).

## Rule 2: the camera is the ground truth for everything it can see (the consistency lock)

Face priors invent detail. That is how they work, and also how they fail: on our footage the model
moved the far eye's gaze toward the camera and turned a brown iris grey. Both errors are at **low
spatial frequencies**, and low frequencies are exactly what survives defocus. The blurry frame
therefore still tells us where the iris is, what colour it is, and how the face is lit.

LucidFace keeps only what the camera could not record:

```
r' = r + G_s * (a - r)
```

`a` is the aligned observed face, `r` the model output, and `G_s` a gaussian low-pass whose cutoff
`s` is a fixed fraction of the inter-ocular distance (`--lock 0.066`, so behaviour does not depend on
shot size). Below the cutoff the result equals the observation; above it, the model's detail (lashes,
lid edges, iris texture) is kept. This is one step of classic back-projection (Irani and Peleg, 1991),
the same idea as data-consistency steps in modern restoration (e.g. DDNM, DPS), applied as a
model-agnostic post-step.

*Measured:* on two eye test clips, eye fidelity error (re-blurred output vs observation, Lab) went from
2.26 to 1.81, and iris colour error from 4.98 to 3.62 (dE). The same lock is applied to the body
deblur.

*Choosing the model setting:* with the lock in place, use the most faithful setting of the face prior
(CodeFormer `w = 1.0`). The common default `w = 0.7` invents more and departs further from the
observation (fidelity error 2.88 vs 2.26 before the lock).

## Rule 3: judge eyes by eyes

Viewers look at eyes first, and whole-face metrics hide eye failures. LucidFace's QC and benchmark
report eye-specific numbers:

- **eye departure:** re-blur the restored eye region with the estimated defocus and compare it with the
  observed eye. No reference is needed, so this works on real footage. It catches moved gaze and
  changed iris colour.
- **iris colour error:** Lab distance of the darkest quarter of the eye box.
- **eye flicker:** frame-to-frame change of the restored eyes, compared with the original's own eye
  motion.

In our model comparison these numbers put the options in the same order a person did:
RestoreFormer turned irises blue, GFPGAN gave "doll eyes", and CodeFormer 0.7 greyed the far eye.

## Rule 4: steady the face before and after the model

Per-frame restoration shimmers. Two sources, two fixes:

1. **Alignment jitter.** Landmarks jump by a pixel or two per frame, so the face is re-aligned
   differently each time and the model's detail moves with it. Pass 1 smooths landmarks over time (a
   gaussian over about 5 frames), fills short detection gaps, and never smooths across a cut.
2. **Detail flicker.** Pass 2 blends each frame's detail layer `d = r' - G_s(r')` with the previous
   frame's, motion-compensated by optical flow (RAFT-small), and only where the flow is trustworthy
   (`conf = exp(-|warp(a_t-1) - a_t| / tau)`). Blinks and fast mouth movement get confidence near
   zero, so they are never ghosted. Only the detail layer is blended: low frequencies stay locked to
   the observation, so motion is never lagged.

## Rule 5: a person signs off, and the tool makes that fast

No automatic metric replaces a person looking at the result, so LucidFace makes the look quick:

- **Flags:** frames where the model departed strongly from what was filmed, where eyes shimmer, where
  identity drifts from the reference photos, or where tracking is unreliable.
- **Fallback:** flagged frames fade back to the original over a short ramp; there are no hard switches.
- **Report:** an HTML page with timelines, flagged ranges as timestamps, and before/after thumbnails.
- **Reference lock (optional):** with sharp photos of the person, each frame is restored several ways
  and the candidate closest to the reference identity is kept, with hysteresis so the choice does not
  flicker.

## Production details that matter

- **Upload only the face region.** `lucidface crop` finds the smallest box that holds the face in every
  frame. On our 27-minute 10-bit file this was 815 MB instead of 10.6 GB.
- **Composite onto the original.** Only that box is replaced, with a feathered edge. Everything else is
  the untouched camera original, at the source bit depth (10-bit 4:2:2 stays 10-bit 4:2:2), with the
  original audio and timecode.
- **GPU end to end.** Alignment, the lock, the temporal blend and paste-back are torch operations on
  the GPU. The reference script used the CPU for these and capped at 1.4 fps per worker.

## Limits

- Hair, hands and clothes need `--body`, which is slow (about 0.3 to 1 s per frame).
- The face prior can still invent features the camera never captured (moles, exact eye shape). The
  reference lock reduces this, it does not remove it. Always check the report.
- Profile views beyond about 60 degrees and heavily occluded faces fall back to the original.
- Licences: the best face prior (CodeFormer) is **non-commercial**. See NOTICE.md.
