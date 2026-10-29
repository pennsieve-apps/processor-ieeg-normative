# processor-ieeg-normative

A step of the **Interictal Biomarker Mapping** workflow
(`ieeg-prep → {ieeg-spikes, ieeg-bandpower → ieeg-normative, ieeg-hfo} → ieeg-report`).
It asks, for every intracranial EEG channel: *is this channel's spectrum unusual for its
location in the brain?* The answer is a z-score per frequency band against the normative
channels of the **HUP Interictal iEEG Atlas** (Bernabei et al. 2022; Pennsieve Discover 179)
that lie near the channel in MNI space, and a per-channel maximum |z|.

- **For the epileptologist:** a normative map flags channels whose interictal activity departs
  from what healthy tissue in the same region shows across 60 other patients. It is one line of
  evidence, to be read with spikes, HFOs and seizure onset. On the atlas itself, the maximum |z|
  over bands **did not** separate seizure-onset-zone or resected channels from the others (AUC
  ≈ 0.48; see *Validation*). The direction of change carries more signal than its size: SOZ
  channels have more delta (signed delta z, AUC 0.58).
- **For the researcher:** the atlas's reference spectra are recomputed from its raw 60-s wake
  clips with the same module ieeg-bandpower uses (`processor/spectral.py`, byte-identical), with
  the spectral settings read from ieeg-bandpower's `provenance.json`. Reference sets are
  coordinate neighbourhoods (`RADIUS_MM`) instead of atlas regions, and atlas patients are left
  out of their own reference.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `features` | input | subfolder | `bandpower.csv` from ieeg-bandpower with its `provenance.json` beside it (the spectral settings). Several may fan in. |
| `reference` | input | subfolder | `HUP_atlas_final.mat` (Discover 179, MATLAB 5 file, read with `scipy.io`), or another `.mat` with the same fields. |
| `ieeg` | input (optional) | subfolder | The ieeg-prep derivative: `_channels.tsv` (`x, y, z, space`, `soz`, `resected`), `_electrodes.tsv` (`atlas_patient`), `recordings.csv` (`subject`). Without it every channel is `unknown`. |
| `abnormality` | output | `OUTPUT_DIR` (or `OUTPUT_DIR/<LABEL>/`) | `abnormality.csv`, `channel_abnormality.csv`, `validation.csv` (when labels are known), `provenance.json`. |

Without links (everything merged at the root of `INPUT_DIR`) the files are found by name.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `RADIUS_MM` | `20` | Reference = the atlas's normative channels whose MNI centroid is within this distance of the test channel. |
| `MIN_REFERENCE` | `10` | Fewer reference channels than this: `region = insufficient`, no z. |
| `FEATURE` | `norm_median` | What is z-scored: `norm_median` (Bernabei 2022: median of the sum-normalised PSD per band), `taylor2022` (log10 band power normalised to sum to one over the bands, Taylor et al. 2022), `log_rel_power`, `rel_power`, `log_abs_power`. |
| `LEAVE_PATIENT_OUT` | `true` | A subject that is an atlas patient (`sub-HUPatlas<NNN>`, or an `atlas_patient` value in its `_electrodes.tsv`) is compared only with the other 59 patients. |
| `EXCLUDE_IRRITATIVE` | `false` | Also drop atlas channels with `spike_24h > 24` (the atlas's own definition of the irritative zone) from the reference, as Bernabei et al. did; leaves 854 of 2626 normative channels. |
| `LABEL` | (empty) | When set, outputs go to `OUTPUT_DIR/<LABEL>/`. |

## What it does

1. **Reference atlas.** `HUP_atlas_final.mat` is a MATLAB 5 file (not v7.3/HDF5), read with
   `scipy.io.loadmat`: `wake_clip` (12000 samples = 60 s at 200 Hz × 3431 bipolar channels,
   filtered 0.5–80 Hz), `mni_coords` (MNI-152 bipolar centroids), `patient_no` (60 patients),
   `soz_ch` (462), `resected_ch` (609), `spike_24h`. A v7.3 file is read with h5py if it is
   installed (it is not in this image) and otherwise refused with a clear message.
2. **Normative channels:** `soz_ch == 0 and resected_ch == 0` (2626 of 3431), optionally also
   `spike_24h <= 24`.
3. **Reference spectra** of all 3431 clips with `processor/spectral.py` and the settings of
   the bandpower run (window, overlap, resample rate, band edges; from its `provenance.json`).
   The clips are already 0.5–80 Hz at 200 Hz, so they are not filtered or resampled again.
   Computed once per run and setting (~2 s), kept in memory only.
4. **Per channel** of each recording in `bandpower.csv`: coordinates from `_channels.tsv`
   (`x, y, z` with `space` starting with `MNI`, `IXI549Space`, `ICBM` or `fsaverage`; empty
   `space` is taken as MNI with a note; any other space, e.g. ieeg-prep's `native`, counts as
   no coordinates). Reference = normative channels within `RADIUS_MM`, minus the tested
   patient's own channels when it is an atlas patient. `z = (x − mean) / sd` (sd with ddof 1)
   per band, `abs_z = |z|`; `max_abs_z` = the largest `abs_z` over the five bands (not
   broadband), `max_band` the band it came from.
5. **Validation:** when `_channels.tsv` has `soz` / `resected` labels, the AUC of `max_abs_z`
   for each label, per recording and pooled (`validation.csv`).

### Region assignment: coordinate neighbourhoods, not atlas regions

The atlas has MNI coordinates but no region labels. Bernabei et al. 2022 assigned channels to
the **AAL** atlas (automated anatomical labelling), merged sparsely sampled neighbouring gyri to
20 regions per hemisphere, dropped cerebellum and basal ganglia, and z-scored per region (the
merge table is in their supplement). We found no clear open licence for the AAL atlas files,
and the merged-region table is not machine-readable, so this app uses **coordinate
neighbourhoods** instead: every test channel gets its own reference set, the normative channels
within `RADIUS_MM` (default 20 mm, roughly the size of a merged AAL region) in MNI space.

Effect, on the atlas (leave-one-patient-out, `norm_median`, `MIN_REFERENCE=10`):

| `RADIUS_MM` | channels scored | reference channels per scored channel, median (IQR) | AUC SOZ | AUC resected |
|---|---|---|---|---|
| 10 | 2047 / 3431 (60 %) | 18 (13–24) | 0.443 | 0.456 |
| 20 | 3367 / 3431 (98 %) | 70 (44–101) | 0.480 | 0.474 |
| 30 | 3425 / 3431 (99.8 %) | 179 (120–233) | 0.483 | 0.474 |

Smaller neighbourhoods are more anatomically specific but leave channels in sparsely sampled
areas without a reference and make the mean and sd noisy; beyond 20 mm little changes. The
neighbourhood ignores grey/white matter and depth vs. surface contacts (matching depth to depth
and strip/grid to strip/grid with the atlas's `depth_elecs` did not change the AUC: 0.460 vs
0.480 at 20 mm). `region` in the outputs therefore says how the reference was chosen
(`within-20mm`), not an anatomical label.

## Outputs

**`abnormality.csv`**: one row per channel and band (5 bands).

| Column | |
|---|---|
| `recording_id`, `channel` | as in `bandpower.csv` |
| `band` | `delta`, `theta`, `alpha`, `beta`, `gamma` |
| `z` | (value − reference mean) / reference sd; empty when not scored |
| `abs_z` | \|z\| |
| `region` | `within-<R>mm` (scored), `insufficient` (fewer than `MIN_REFERENCE` reference channels), `unknown` (no MNI coordinates) |
| `n_reference` | normative atlas channels in the reference set |

**`channel_abnormality.csv`**: one row per channel: `recording_id, channel, max_abs_z, max_band,
region, n_reference` (`max_band` is an extra column beyond `conventions.md`).

**`validation.csv`** (when labels are known): `scope` (a recording id or `all`), `label`
(`soz`, `resected`), `n_positive`, `n_negative`, `n_unscored`, `auc_max_abs_z` (Mann–Whitney;
> 0.5 means the labelled channels are more abnormal).

**`provenance.json`**: app and version, tool versions, parameters, the reference (path, channels,
patients, normative channels and the rule, seconds), inputs found and per recording: subject,
atlas patient left out, channels scored/insufficient/unknown, notes.

## Validation

### Leave-one-patient-out over the whole atlas (`make validate-atlas`)

Every one of the 60 atlas patients scored against the other 59 (`scripts/validate_atlas.py`,
the app's own code). AUC of `max_abs_z`, pooled over 3431 channels (per-patient median in
brackets), `RADIUS_MM=20`, `MIN_REFERENCE=10`:

| Feature | Bands | Irritative zone in reference | AUC SOZ | AUC resected |
|---|---|---|---|---|
| `norm_median` (default) | conventions | kept | **0.480** (0.434) | **0.474** (0.416) |
| `norm_median` | bernabei2022 | kept | 0.459 (0.400) | 0.447 (0.422) |
| `norm_median` | conventions | excluded | 0.471 (0.456) | 0.483 (0.472) |
| `taylor2022` | conventions | kept | 0.451 (0.389) | 0.482 (0.478) |
| `log_rel_power` | conventions | kept | 0.479 (0.431) | 0.453 (0.441) |

Per band, signed z (higher z = more of that band than the neighbourhood), `norm_median`, 20 mm:
SOZ AUC delta **0.576**, theta 0.510, alpha 0.451, beta 0.456, gamma 0.455; resected delta
0.583. SOZ channels carry relatively more delta and less alpha/beta/gamma, but the maximum
|z| over bands mixes these small shifts with the ordinary spread of normal channels (median
max |z|: SOZ 1.43, normative channels of other patients 1.47).

### Through the pipeline, 5 atlas patients (ieeg-bandpower → ieeg-normative, Docker)

ieeg-prep does not yet emit the atlas patients, so `scripts/atlas_subjects.py` writes patients
1–5 as an ieeg-prep-style derivative (`sub-HUPatlas001`…`005`, 200 Hz EDF 16-bit, MNI152
centroids, soz/resected labels, `atlas_patient` in `_electrodes.tsv`). Both images were run on
it with default parameters:

| Patient | Channels | SOZ | Resected | AUC SOZ | AUC resected |
|---|---|---|---|---|---|
| HUPatlas001 | 22 | 2 | 3 | 0.300 | 0.246 |
| HUPatlas002 | 74 (73 scored) | 4 | 24 | 0.192 | 0.351 |
| HUPatlas003 | 55 | 15 | 14 | 0.757 | 0.822 |
| HUPatlas004 | 74 (73 scored) | 24 | 26 | 0.400 | 0.421 |
| HUPatlas005 | 59 (57 scored) | 13 | 28 | 0.521 | 0.349 |
| **pooled** | 284 (280 scored) | 58 | 95 | **0.497** | **0.394** |

The direct atlas computation for the same five patients gives the same AUCs to three decimals,
so the EDF round trip and the two-app split change nothing.

### Compared with Bernabei et al. 2022

Bernabei et al. did not report a univariate AUC for spectral z-scores. Their channel-level
classifier (a random forest on |z| of band power and coherence, patient-level 10-fold
cross-validation, target: resected SOZ) reached AUC 0.77 with both feature sets, **0.65 with
spectral power alone** and 0.68 with connectivity alone. An unsupervised max |z| cannot weight
bands or use their sign, and the reference here is different (all 60 atlas patients' non-SOZ,
non-resected channels; theirs was the MNI Open iEEG Atlas plus 532 channels of good-outcome HUP
patients outside the SOZ, irritative and resection zones). Taylor et al. 2022 report
region-level, not channel-level, distinguishability (D_RS, an AUC); in their good-outcome
patients D_RS did not differ significantly from 0.5. Use `max_abs_z` as a screen, look at the
signed per-band z in `abnormality.csv`, and do not read a low |z| as evidence of healthy tissue.

### Checks that rule out a bug (`scripts/atlas_checks.py`)

- **The arrays line up.** `whosmat` shows plain double matrices, not structs: `wake_clip` is
  12000 × 3431 and every per-channel field is 3431 × 1, all read without transposing. Three
  checks show that signals, coordinates and labels describe the same channels:
  - signal correlation between channel pairs of a patient falls with their MNI distance
    (Spearman ρ = −0.50 for consecutive channels, −0.39 for random pairs);
  - `spike_24h` tracks the spectrum: log spike rate vs delta share ρ = 0.34, vs beta share
    ρ = −0.31. Irritative channels (spike_24h > 24) have a median delta share of 0.47 against
    0.32 for the others (rank-sum p = 2e-40);
  - SOZ channels cluster in space: the nearest other SOZ channel lies at a median 6.2 mm, the
    nearest non-SOZ channel at 12.6 mm.
- **Units and filtering are the same on both sides.** The clips are in µV (median channel SD
  33 µV). The reference and atlas patients run through ieeg-bandpower both take the "already
  0.5–80 Hz" path, so neither is filtered twice. The pipeline AUCs equal the direct ones to
  three decimals.
- **Leave-one-patient-out works.** The test patient's channels are removed by `patient_no`
  (n_reference drops; tested). The z-scores are calibrated: the median |z| of normative
  channels is 0.73, close to the 0.67 expected for N(0, 1).

**Replication grid** (AUC with max |z| / mean |z| over the five bands):

| R (mm) | LOPO | Irritative zone excluded | SOZ vs all others | SOZ vs normative channels | resected SOZ vs all others |
|---|---|---|---|---|---|
| 20 | yes | no (default) | 0.480 / 0.488 | 0.480 / 0.489 | 0.442 / 0.435 |
| 20 | yes | yes | 0.471 / 0.492 | 0.452 / 0.495 | 0.435 / 0.438 |
| 30 | yes | yes | 0.496 / 0.513 | 0.480 / 0.520 | 0.457 / 0.464 |
| 40 | yes | yes | 0.501 / 0.519 | 0.486 / 0.530 | 0.459 / 0.473 |
| 20 | **no** | yes | 0.523 / 0.534 | **0.592 / 0.608** | 0.491 / 0.486 |
| 30 | **no** | yes | 0.524 / 0.536 | 0.561 / 0.589 | 0.491 / 0.491 |

Without leave-one-patient-out, a patient's own normative channels sit in their reference,
which is circular. Bernabei et al. do not say whether they excluded the patient; doing so here
moves the AUCs from about 0.5 to about 0.6. Coarser radii (30–40 mm, closer to the merged AAL
regions) and the mean of |z| instead of the maximum each add 0.01–0.04.

**Per band |z|, SOZ vs normative channels** (R = 20 mm, LOPO, two-sided rank-sum):
- *Irritative zone excluded from the reference, as in the paper:* delta is the only band that
  differs (median |z| 0.90 vs 0.72, p = 0.0009); alpha 0.90 vs 0.76 (p = 0.10); theta, beta and
  gamma p ≥ 0.46.
- *Irritative zone kept:* no band differs (p ≥ 0.11).
- *Signed z of SOZ channels* (medians): delta +0.61, alpha −0.67, beta −0.60, gamma −0.58. Signed
  delta z gives AUC 0.58, better than any |z| summary here. The paper reports the opposite in
  its supplement (absolute scores beat directional ones).

**How Bernabei et al. compared:** their univariate result is regional. A rank-sum test of the
median normalised band power per AAL region and band, Bonferroni-corrected, found SOZ and
irritative channels different from normative tissue in some neocortical regions (fusiform,
inferior temporal, medial frontal) and **not** in mesial temporal structures. They give no
per-band |z| statistics in the text, and say absolute band power did not discriminate. The
strong effects they report are for coherence; that is not in this app.

Conclusion: no bug was found. In this atlas, under leave-one-patient-out, single-channel
spectral |z| is near chance. That agrees with the paper's own regional and largely negative
univariate spectral results. The defaults stay as they are.

On the demo seizure patients (Discover 530 via ieeg-prep) every channel is `unknown`: their
electrode coordinates are in each patient's native space, not MNI (see *Limitations*).

## Runtime

`computeTypes: [standard, lambda]`, 1 vCPU / 2 GiB, 900 s timeout. Measured in Docker
(linux/amd64 **emulated on an M-series Mac**, `--cpus 2 --memory 2g`), the real atlas as
reference:

| Input | Wall time | cgroup `memory.peak` |
|---|---|---|
| 5 atlas patients (284 channels) | 7.9 s (3.6 s in the app: atlas load 2.1 s, reference spectra 1.4 s) | 1.18 GB |
| 24 demo recordings + 5 atlas patients (3166 channels, fan-in) | 10.3 s | 0.76 GB |

The peak includes the page cache of the 317 MB atlas file; the process itself holds the atlas
(~330 MB as float64) and the reference tables. The image is 91 MB compressed (179 MB unpacked).

## Running it locally

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python             # pytest: synthetic atlas, features and channels; Lambda handler
make run PYTHON=.venv/bin/python              # Docker over data/fixture/linked/, into data/output/
make validate-atlas PYTHON=.venv/bin/python   # LOPO AUC grid over the real atlas -> data/validation-atlas.csv
make atlas-subjects PYTHON=.venv/bin/python   # atlas patients 1-5 as a derivative, for the pipeline check
make run-demo                                 # WORK=../workflow-epilepsy-ieeg/data/work (bandpower + prep/ieeg)
make check-vendored                           # spectral.py identical to ieeg-bandpower's
```

`tests/make_fixture.py` writes a synthetic atlas (6 patients × 24 channels in the Discover 179
layout, SOZ channels with extra delta), the `_channels.tsv` of an atlas patient and of a new
patient (two normal channels, two with a delta excess, one without coordinates, one far from
any atlas channel) and their `bandpower.csv` computed with `spectral.py`.

As a Lambda function: `processor/entry.py` hands over to `awslambdaric` when
`AWS_LAMBDA_RUNTIME_API` is set; `processor/handler.py` takes `inputDir`, `outputDir` and the
parameters from the payload and runs the same `main()`; a failed run raises.

## Limitations

- **Coordinates must be in MNI space.** Channels in a patient's native (CT/MRI) space cannot be
  placed in the atlas and are `unknown`. The demo seizure dataset (Discover 530) only has native
  coordinates, so ieeg-prep writes `space = native` and nothing is scored there. Registering
  electrodes to MNI is outside this step.
- **Weak discrimination.** On the atlas, max |z| is at chance for SOZ and resected channels
  (above). A 60-s wake clip per patient is a small sample of interictal activity, and abnormality
  maps vary over time (Wang et al. 2023).
- **Reference ≠ healthy brain.** Normative channels are those outside the SOZ and the resection
  in epilepsy patients; they include irritative-zone channels unless `EXCLUDE_IRRITATIVE=true`,
  and outcome is not in the atlas file, so poor-outcome patients' "normal" channels are included.
- **State.** The atlas is wake, interictal, ≥ 2 h from seizures. A pre-ictal baseline of a
  seizure clip (ieeg-bandpower's default on seizure recordings) is not the same state.
- **Montage and sampling.** The atlas is bipolar, 200 Hz, 0.5–80 Hz; inputs should be bipolar
  too (ieeg-prep's default) and go through ieeg-bandpower's resampling.
- Coordinate spaces are matched by name only (`MNI…` etc.); differences between MNI templates
  (a few mm) are ignored.

## Licences

Apache-2.0 for this app's code (`LICENSE`). The image bundles NumPy and SciPy (BSD-3-Clause)
and awslambdaric (Apache-2.0); see `THIRD_PARTY.md`. The atlas (input, not bundled) is ODC-By
1.0: cite it (below) when you use results computed with it. Nothing was copied from
jbernabei/iEEG_atlas (no licence).

## Citations

Normative atlas, method and features:

> Bernabei JM, Sinha N, Arnold TC, Conrad E, Ong I, Pattnaik AR, Stein JM, Shinohara RT, Lucas TH,
> Bassett DS, Davis KA, Litt B. **Normative intracranial EEG maps epileptogenic tissues in focal
> epilepsy.** Brain 145(6):1949–1961 (2022). doi:10.1093/brain/awab480, PMID 35640886.

> Bernabei J, Litt B. **HUP Interictal iEEG Atlas** (version 1). Pennsieve Discover (2021).
> doi:10.26275/lzhy-ox20. ODC-By 1.0.

Normative mapping with max |z| over bands, `taylor2022` feature:

> Taylor PN, Papasavvas CA, Owen TW, Schroeder GM, Hutchings FE, Chowdhury FA, Diehl B, Duncan JS,
> McEvoy AW, Miserocchi A, de Tisi J, Vos SB, Walker MC, Wang Y. **Normative brain mapping of
> interictal intracranial EEG to localize epileptogenic tissue.** Brain 145(3):939–949 (2022).
> doi:10.1093/brain/awab380, PMID 35075485.

Temporal stability of abnormality maps:

> Wang Y, Schroeder GM, Horsley JJ, Panagiotopoulou M, Chowdhury FA, Diehl B, Duncan JS, McEvoy AW,
> Miserocchi A, de Tisi J, Taylor PN. **Temporal stability of intracranial electroencephalographic
> abnormality maps for localizing epileptogenic tissue.** Epilepsia 64(8):2070–2080 (2023).
> doi:10.1111/epi.17663, PMID 37226553.

Software:

> Virtanen P, Gommers R, Oliphant TE, et al. (SciPy 1.0 Contributors). **SciPy 1.0: fundamental
> algorithms for scientific computing in Python.** Nat Methods 17(3):261–272 (2020).
> doi:10.1038/s41592-019-0686-2, PMID 32015543.

DOIs were checked against Crossref (articles) and DataCite (the dataset): title, authors, year,
volume, pages; PMIDs against PubMed E-utilities. `CITATION.cff` lists all authors in
Crossref/DataCite order.
