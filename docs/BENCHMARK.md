# Benchmark

Anyone can rerun this. It needs a CUDA GPU and about 3 hours on an RTX 4080 SUPER.

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

RESULTS_PLACEHOLDER
