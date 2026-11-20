# Third-party components and their licences

LucidFace's own code is MIT (see LICENSE). LucidFace **does not ship any model weights**. Each model
below is downloaded by the user and is governed by its own licence. Using LucidFace commercially
requires that every model you enable allows commercial use. Licences as checked on 2026-11-20; not
legal advice; verify before use.

| Component | Used for | Code licence | Weights | Commercial use |
|---|---|---|---|---|
| [CodeFormer](https://github.com/sczhou/CodeFormer) | face restoration (default backend) | NTU S-Lab License 1.0 | same | **No** without written permission from the authors |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) | face restoration (alternative backend) | Apache-2.0 | trained on FFHQ (CC BY-NC-SA 4.0 dataset) | code yes; weights unclear |
| [facexlib](https://github.com/xinntao/facexlib) RetinaFace + ParseNet | face detection, landmarks, face parsing | MIT | bundled downloads | check the facexlib release terms |
| [Restormer](https://github.com/swz30/Restormer) single-image defocus | `--body` (hair, hands, clothes) | MIT | DPDD-trained, no separate terms stated | likely, check |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) InceptionResnetV1 | `--reference` identity lock, benchmark identity score | MIT | trained on VGGFace2 | check the dataset terms |
| [RAFT](https://pytorch.org/vision/stable/models/raft.html) (torchvision) | temporal steadying (optical flow) | BSD-3-Clause | torchvision weights | yes |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | benchmark only | BSD-2-Clause | bundled | yes |

## Benchmark footage
The benchmark downloads five interviews that are works of the US federal government and therefore in
the public domain: a White House weekly address (2015), NASA Goddard Scientific Visualization Studio
14623, two DVIDS interviews, and a CDC message. Public-domain status does not waive personality
rights: do not use these people's faces to promote LucidFace or imply their endorsement. See
docs/BENCHMARK.md for exact URLs and SHA-256 hashes.

## Models reviewed and not used by default
- Reference-based face restoration (InstantRestore, DMDNet, ASFFNet512, RestorerID, ReF-LDM,
  RefSTAR): all have non-commercial or missing licences, or non-commercial dependencies, as of 2026-11.
- Defocus deblurring (LaKDNet, DRBNet, IFAN): AGPL-3.0 or unclear terms. LaKDNet (LFDOF+DPDD)
  generalised best in published numbers and is a good choice where AGPL is acceptable.
- DaBiT (video defocus): S-Lab License 1.0, non-commercial.
