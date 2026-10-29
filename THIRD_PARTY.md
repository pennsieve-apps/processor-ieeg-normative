# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI as binary wheels at the versions pinned in `requirements.txt`.
All are permissively licensed; nothing GPL is linked in-process.

| Component | Version | What it does here | Licence | Source |
|---|---|---|---|---|
| NumPy | 2.5.4 | Arrays, FFT of the Welch windows, distances, z-scores | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | Reads the atlas .mat (`scipy.io.loadmat`), Hamming window, filtering/resampling in the shared spectral module | BSD-3-Clause | https://scipy.org |
| awslambdaric | 4.2.3 | Lambda runtime interface client (only when run as a Lambda function) | Apache-2.0 | https://github.com/aws/aws-lambda-python-runtime-interface-client |
| simplejson, snapshot-restore-py | 4.2.0, 1.0.0 | awslambdaric dependencies | MIT OR AFL-2.1; Apache-2.0 | PyPI |
| python:3.12.15-slim-bookworm | — | Base image (Debian 12 + CPython 3.12, pip) | PSF-2.0 (Python); Debian packages under their own free licences; pip MIT | https://hub.docker.com/_/python |

`processor/spectral.py` is a byte-identical copy of the module in processor-ieeg-bandpower
(same repository owner, Apache-2.0); `make check-vendored` compares the two.

The normative atlas is **not** in the image: it is an input (port `reference`). It is the HUP
Interictal iEEG Atlas, Pennsieve Discover 179 (Bernabei J, Litt B; doi:10.26275/lzhy-ox20),
released under the Open Data Commons Attribution License 1.0 (ODC-By): use requires attribution,
which `CITATION.cff` and the README give.

The method follows Bernabei et al. 2022 (Brain, doi:10.1093/brain/awab480) and Taylor et al.
2022 (Brain, doi:10.1093/brain/awab380) and was reimplemented from the papers; no code was taken
from jbernabei/iEEG_atlas (no licence). cnnp-lab/2023Wang_Temporal-stability (CC0-1.0) was read
as a reference only; no code was copied from it either.
