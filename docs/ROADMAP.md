# Roadmap

Ordered by expected payoff. Each item says what we would measure to know it worked.

1. **Measured blur model for the lock.** The consistency lock uses a gaussian low-pass. Real defocus
   is closer to a disk. Estimating the actual point-spread function per shot (from edges in the face
   region) and back-projecting through it would let the lock pin more of the true structure without
   blurring the restored detail. *Measure:* eye PSNR at levels 2 and 3 of the benchmark.
2. **Better temporal steadying.** Today the steadier trades sharpness for stability (see the benchmark
   at `--steady 0.5`). Options: estimate flow on the restored faces instead of the blurry inputs, or use
   a video face model with temporal attention (KEEP, PGTFormer) as the backend under the same lock.
   *Measure:* `t_err_eyes` down without `psnr_eyes` going down.
3. **A true reference-based backend.** InstantRestore (CC-BY-NC weights, no code licence) or DMDNet/
   ASFFNet (CC BY-NC-SA) as optional, clearly non-commercial backends. A permissively licensed one
   would need training on an Apache base model. *Measure:* `id_cos` against ground truth.
4. **Commercial-clean default.** A face prior trained on permissively licensed data, so LucidFace can
   be used commercially without per-model permission.
5. **Faster body deblur.** NRKNet (MIT, 6 M parameters) as a lighter alternative to Restormer, or
   running the body model at half resolution under the lock. *Measure:* fps with `--body`.
6. **Multiple faces.** Track and restore every face in a two-person interview, not just the centre one.
7. **NLE plug-in.** A DaVinci Resolve or Premiere workflow that sends only the face region out and
   brings the composite back on the timeline.
