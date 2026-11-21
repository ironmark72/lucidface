# LucidFace

**Restore out-of-focus faces in video, locked to what the camera actually filmed.**

Interview footage where the camera focused on the wall instead of the person is common, and usually
unusable. LucidFace rebuilds the face (eyes, brows, lips, skin detail) while guaranteeing that
everything the blurry footage *can* still tell us, such as where the person is looking, the colour of
their eyes and how the face is lit, stays exactly as filmed. The rest of the frame is the untouched
camera original.

It is a method as much as a tool. Each rule in [docs/METHOD.md](docs/METHOD.md) came from a failure on
real client footage, and each is backed by a benchmark anyone can rerun on public-domain video
([docs/BENCHMARK.md](docs/BENCHMARK.md)).

## What it does

1. **Tracks the face** through the whole video, with temporally smoothed landmarks (less shimmer),
   cut detection, and a focus score per frame.
2. **Restores the face** with a face prior (CodeFormer by default, GFPGAN optional), batched on the GPU.
3. **Locks it to the observation**: below a cutoff frequency the result equals the camera's own picture,
   so the model cannot move the gaze, recolour the eyes or reshape the face; above it, the new detail stays.
4. **Steadies it over time** with motion-compensated blending of the detail layer that switches off
   wherever the motion is unreliable.
5. **Optionally**: matches identity to reference photos you supply (`--reference`), deblurs hair, hands
   and clothes with a lens-defocus model (`--body`), and leaves already-sharp shots alone (`--gate`).
6. **Checks itself**: flags frames that look wrong, fades them back to the original, and writes an HTML
   QC report with timestamps and thumbnails.
7. **Composites onto your original file**: same bit depth and chroma (10-bit 4:2:2 stays 10-bit 4:2:2),
   original audio, original timecode. Only a feathered box around the face changes.

## Quick start

```bash
git clone https://github.com/ironmark72/lucidface && cd lucidface
bash scripts/pod_setup.sh ~/models          # installs LucidFace, fetches CodeFormer + Restormer
export LUCIDFACE_CODEFORMER_DIR=~/models/CodeFormer
lucidface run interview.mp4 -o interview_fixed.mov
```

You get `interview_fixed.mov` and `interview_fixed.qc.html`. Open the report first.

### Long videos, slow uploads

Upload only the face region, restore it on a GPU machine, and composite at home:

```bash
# local
lucidface crop interview.mov -o face.mp4 --box-out box.json --stride 25
# GPU machine
lucidface restore face.mp4 -o face_restored.mp4
# local
lucidface composite interview.mov face_restored.mp4 -o interview_fixed.mov --box box.json --qc face_restored.qc.npz
```

On a 27-minute 10-bit interview the upload went from 10.6 GB to 815 MB.

### Useful options

| option | default | what it does |
|---|---|---|
| `--backend` | codeformer | `codeformer`, `gfpgan`, or `passthrough` (pipeline test) |
| `--fidelity` | 1.0 | CodeFormer `w`. Keep 1.0: lower values invent more |
| `--lock` | 0.066 | consistency-lock cutoff, as a fraction of the eye distance. 0 = off |
| `--steady` | 0.25 | temporal blend strength. 0 = off. Higher = less shimmer, softer motion |
| `--reference a.jpg b.jpg` | | sharp photos of the person: identity lock + identity QC |
| `--body` | off | also deblur hair, hands and clothes (slow) |
| `--gate T` | off | only restore frames whose focus score is below T |

## Requirements

A CUDA GPU with 16 GB or more, Python 3.10+, PyTorch 2.1+, and ffmpeg. Measured on an RTX 4080 SUPER:
about 10 fps for 1080p with CodeFormer and the default settings (12 fps without steadying), everything included. Results: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Licences: read before commercial use

LucidFace's code is MIT. It ships **no model weights**. The default face model, CodeFormer, is
**non-commercial** (S-Lab License 1.0). See [NOTICE.md](NOTICE.md) for every component and what each
licence allows.

## Ethics

LucidFace is meant to recover footage of people who were filmed with their knowledge. It reconstructs
detail the camera did not record, so a restored face is an estimate, not evidence: do not use it for
identification, forensics or anything where a mistaken detail could harm someone. Keep the original
footage, and say that restoration was applied when it matters.

## Credits

Face priors: CodeFormer (Zhou et al., 2022), GFPGAN (Wang et al., 2021). Detection and parsing:
facexlib (RetinaFace, ParseNet). Defocus: Restormer (Zamir et al., 2022). Flow: RAFT (Teed and Deng,
2020). Identity: facenet-pytorch. The consistency lock is classic iterative back-projection
(Irani and Peleg, 1991). Benchmark footage: US government works, public domain.
