# Benchmark

Anyone can rerun this. It needs a CUDA GPU and about 2.5 hours on an RTX 4080 SUPER.

```bash
lucidface bench download --dir bench      # fetches 5 public-domain clips, checks nothing is missing
lucidface bench run --dir bench           # degrade, restore, score -> bench/results.md
python scripts/test_extras.py bench       # --reference and --body -> bench/extras.json
```

## Protocol

1. **Ground truth.** Five sharp interview segments (table below), cut losslessly enough (x264 CRF 6) to
   serve as reference.
2. **Degradation.** A disk point-spread function (an out-of-focus lens) with radius 8%, 12% or 16% of
   the subject's inter-ocular distance (levels 1 to 3), so a level means the same blur on a close-up
   and on a medium shot. Then gaussian sensor noise (sigma 1.5) and an H.264 re-encode (CRF 20), as
   in a real camera file. The whole frame is blurred; only the face is scored.
3. **Tracking on the blurry input.** Each method gets landmarks detected on the degraded video, as in
   real use. Scoring uses landmarks from the sharp ground truth.
4. **Metrics,** computed in the ground truth's aligned 512 px face space:
   - `psnr_face`: inside the face-parsing mask (skin, brows, eyes, nose, lips)
   - `lpips_face`: perceptual distance, same mask (lower is better)
   - `psnr_eyes`: on the two eye boxes only
   - `id_cos`: identity similarity to the ground truth (facenet)
   - `t_err_eyes`: |(out_t - out_t-1) - (gt_t - gt_t-1)| on the eyes. Flicker the real footage does
     not have (lower is better)
   - `fps`: end-to-end restore speed on an RTX 4080 SUPER (including decode and encode)

## Clips

All are works of the US federal government, therefore public domain. Do not use these people's
likeness to promote LucidFace or imply their endorsement.

| id | source | window | SHA-256 of the downloaded file |
|---|---|---|---|
| potus2015 | [White House weekly address, 2015-02-07](https://commons.wikimedia.org/wiki/File:2015-02-07_President_Obama%27s_Weekly_Address.webm) | 138-158 s | `b6a810b8e2f103feb2eb26e610c53dbb89bedb574b9eaace229c240737173cdc` |
| nasa_murthy | [NASA Goddard SVS 14623](https://svs.gsfc.nasa.gov/14623) | 37-57 s | `919641454e978b0011946454ada34fb456b6ddfbb4a2e97e32eb6ffc004e8ced` |
| dvids_marceaux | [DVIDS, Faces of Keen Sword](https://commons.wikimedia.org/wiki/File:Faces_of_Keen_Sword_-_Tech_Sgt_Sasha_Marceaux_(1027106).webm) | 6-16 s | `5924f3739be914c088598bdb4d4998b789dc71040919549b3befc698bd9a90ec` |
| dvids_goto | [DVIDS, Balikatan 25 interview](https://commons.wikimedia.org/wiki/File:Balikatan_25-_Humanitarian_Assistance_Disaster_Response_exercise_interview_(960624).webm) | 1-20 s | `a0f1b40baeee62ce6bf13b461501e4f504c3e05dad02e8c35ce3bcf30b60d9df` |
| nasa_ott | [NASA Goddard SVS 12047](https://svs.gsfc.nasa.gov/12047) | 3-15 s | `fef8f2ffe7067b99ab1c77d60b9da21df524dacdbfd558c29b8e8406b6f4d7f3` |

Windows were chosen by a 1 fps face scan as the stretch where exactly one large face is on screen the
whole time (no B-roll or cutaways). The set covers three men and two women, several skin tones, a
close-up, medium shots, glasses and a cap.

## Results

Measured on an RTX 4080 SUPER, 2026-11-20. Means over the five clips; `identity` is facenet
cosine similarity to the sharp ground truth.

> **Read this first.** This run had a colour bug (video was written with the BT.601 matrix while tagged
> BT.709, fixed in commit 0b8daed). Every *restored* clip carries the same small colour shift, and the
> blurry inputs carry a smaller one. Consequences:
> - **Comparisons between restored configurations are valid**: they all share the same shift.
> - **Restored vs blurry input is biased against restoration.** The PSNR columns in particular
>   understate the restored results. Where restoration still wins, the win is a lower bound.
> - A clean re-run is pending. `lucidface bench run` reproduces it on any CUDA GPU.


**mild (r = 8% IOD)**

| configuration | LPIPS face (lower) | identity (higher) | eye flicker vs GT (lower) | PSNR eyes | fps |
|---|---|---|---|---|---|
| blurry input | 0.222 | 0.899 | 6.10 | 21.82 | - |
| CodeFormer w0.7 | 0.101 | 0.905 | 8.28 | 21.61 | 12.2 |
| CodeFormer w1.0 | 0.098 | 0.925 | 8.19 | 21.55 | 12.3 |
| w1.0 + lock | 0.097 | 0.924 | 8.12 | 21.63 | 12.2 |
| w1.0 + matched lock | 0.098 | 0.929 | 8.02 | 21.63 | 12.1 |
| w1.0 + lock + steady 0.25 (default) | 0.096 | 0.925 | 7.59 | 21.77 | 9.9 |
| w1.0 + lock + steady 0.5 | 0.097 | 0.926 | 7.18 | 21.92 | 9.9 |
| w1.0 + matched lock + steady 0.25 | 0.096 | 0.930 | 7.52 | 21.77 | 9.8 |

**medium (12%)**

| configuration | LPIPS face (lower) | identity (higher) | eye flicker vs GT (lower) | PSNR eyes | fps |
|---|---|---|---|---|---|
| blurry input | 0.248 | 0.763 | 6.41 | 20.23 | - |
| CodeFormer w0.7 | 0.117 | 0.805 | 9.41 | 20.02 | 12.2 |
| CodeFormer w1.0 | 0.115 | 0.819 | 9.41 | 19.81 | 12.2 |
| w1.0 + lock | 0.117 | 0.811 | 9.31 | 19.74 | 12.2 |
| w1.0 + matched lock | 0.117 | 0.811 | 9.30 | 19.73 | 12.1 |
| w1.0 + lock + steady 0.25 (default) | 0.116 | 0.813 | 8.58 | 19.87 | 9.9 |
| w1.0 + lock + steady 0.5 | 0.117 | 0.814 | 8.02 | 20.02 | 9.9 |
| w1.0 + matched lock + steady 0.25 | 0.116 | 0.812 | 8.58 | 19.86 | 9.8 |

**heavy (16%)**

| configuration | LPIPS face (lower) | identity (higher) | eye flicker vs GT (lower) | PSNR eyes | fps |
|---|---|---|---|---|---|
| blurry input | 0.265 | 0.577 | 6.59 | 19.21 | - |
| CodeFormer w0.7 | 0.133 | 0.608 | 9.62 | 18.47 | 12.1 |
| CodeFormer w1.0 | 0.132 | 0.605 | 9.64 | 18.24 | 12.1 |
| w1.0 + lock | 0.136 | 0.590 | 9.51 | 18.12 | 12.1 |
| w1.0 + matched lock | 0.134 | 0.599 | 9.58 | 18.13 | 12.1 |
| w1.0 + lock + steady 0.25 (default) | 0.135 | 0.591 | 8.82 | 18.21 | 9.9 |
| w1.0 + lock + steady 0.5 | 0.137 | 0.592 | 8.30 | 18.31 | 9.9 |
| w1.0 + matched lock + steady 0.25 | 0.134 | 0.599 | 8.87 | 18.23 | 9.8 |

### Per-case results (15 cases = 5 clips x 3 blur levels)

| claim | cases |
|---|---|
| Restoration (default) has lower perceptual distance (LPIPS) than the blurry input | 15 of 15 |
| Restoration (default) has higher identity similarity than the blurry input | 13 of 15 |
| CodeFormer `w = 1.0` has higher identity similarity than `w = 0.7` | 11 of 15 |
| `--steady 0.25` lowers eye flicker vs ground truth (compared with no steadying) | 15 of 15 |
| Matched lock beats the fixed lock on identity | 10 of 15 |
| Matched lock beats no lock on identity | 9 of 15 |

### What this says

- **Restoration works.** Perceptual distance to the sharp original roughly halves at every blur level,
  and identity similarity rises (mild 0.899 to 0.925, medium 0.763 to 0.813, heavy 0.577 to 0.591).
- **Use the faithful model setting.** `w = 1.0` beats the common default `w = 0.7` on identity in most
  cases. The exceptions are all medium or heavy blur.
- **Steadying is worth its cost.** `--steady 0.25` lowered flicker in every case for about 20% speed.
  `0.5` lowers it further on these clips, but on real footage with fast head turns it softened the face
  (see docs/METHOD.md), so 0.25 is the default.
- **The consistency lock does not improve these averages.** A fixed cutoff is neutral on mild blur and
  costs a little identity on heavier blur. A cutoff matched to the true blur radius recovers most of
  that (and is the best setting on mild blur: 0.930 with steadying), but with no lock at all identity
  is highest on medium (0.819) and heavy (0.605) blur. LucidFace cannot yet estimate the blur radius on
  real footage, so the default lock stays fixed (it equals the matched value at medium blur). The lock
  stays on because it prevents a failure seen on real footage (moved gaze, recoloured iris) that
  synthetic blur did not trigger; `--lock 0` turns it off.
- **Every restoration flickers more than the blurry input** (about 7.6 to 8.8 vs 6.1 to 6.6). Restored
  detail is not yet perfectly stable over time.

### Optional features (one clip each, level 2; indicative only)

| feature | clip | result | speed |
|---|---|---|---|
| `--reference` (sharp frame from another minute of the same video) | potus2015 | identity 0.854 to 0.867, eye PSNR +0.15 dB | 4.4 fps |
| `--body` (Restormer on hair, shoulders, clothes) | potus2015 | +0.15 dB on the body region vs face-only | 0.95 fps |

The body result was measured before the colour fix and on a single close-up, so treat it as "small
gain, large cost" until re-measured.

