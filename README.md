# processor-ieeg-normative

A step of the **Interictal Biomarker Mapping** workflow
(`ieeg-prep → {ieeg-spikes, ieeg-bandpower → ieeg-normative, ieeg-hfo} → ieeg-report`).
It asks, for every intracranial EEG channel: *are this channel's features unusual for its
location in the brain?* It z-scores **every feature** in its `features` input (band power,
coherence strength, or any other per-channel value in the shared `features.csv` shape) against
the normative channels of a **reference supplied as input**, in the same region. With coherence
between channel pairs it also scores **edges** (Bernabei et al. 2022's connectivity
abnormality). The reference is either a table built from any normative cohort (Format A) or the
**HUP Interictal iEEG Atlas** `.mat` (Discover 179; Format B).

- **For the epileptologist:** a normative map flags channels whose interictal activity departs
  from what normative tissue in the same region shows in other patients. It is one line of
  evidence, to be read with spikes, HFOs and seizure onset. On the atlas, single-channel
  band-power |z| **did not** separate seizure-onset-zone or resected channels from the others
  (AUC ≈ 0.48; see *Validation*). The paper's strongest signal is in coherence, which this app
  now scores too; **its leave-one-patient-out validation over the 60 atlas patients is still
  pending**, so do not read the connectivity score as validated.
- **For the researcher:** test and reference features come from the same code
  (`processor/spectral.py`, byte-identical with ieeg-bandpower's). The atlas is converted by
  running that code on its 60-s wake clips with the spectral settings of the bandpower run being
  scored. Regions are MNI neighbourhoods (`RADIUS_MM`) or `roi` labels; reference patients equal
  to the tested patient are left out of the reference.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `features` | input | subfolder | `features.csv` (`recording_id, channel, feature, value`) from ieeg-bandpower, or any app writing that shape, with `coherence.csv` (channel pairs, for edge-level connectivity) and ieeg-bandpower's `provenance.json` (the spectral settings) beside it. An older ieeg-bandpower output with only `bandpower.csv` is read too (spectral features only). Several may fan in. |
| `reference` | input | subfolder | **Format A**: a folder with `reference_channels.csv` + `features.csv` (+ `coherence.csv`, `provenance.json`), e.g. written by `scripts/build_reference.py`. **Format B**: `HUP_atlas_final.mat` (Discover 179, MATLAB 5 file), or another `.mat` with the same fields. |
| `ieeg` | input (optional) | subfolder | The ieeg-prep derivative: `_channels.tsv` (`x, y, z, space`, `roi`, `soz`, `resected`), `_electrodes.tsv` (`atlas_patient`), `recordings.csv` (`subject`). Without it every channel is `unknown`. |
| `abnormality` | output | `OUTPUT_DIR` (or `OUTPUT_DIR/<LABEL>/`) | `abnormality.csv`, `channel_abnormality.csv`, `edge_abnormality.csv`, `validation.csv` (when labels are known), `provenance.json`. |

Without links (everything merged at the root of `INPUT_DIR`) the files are found by name; a
`features.csv` beside a `reference_channels.csv` belongs to the reference.

### Reference Format A (a reference feature table)

One folder, any normative cohort run through ieeg-prep and ieeg-bandpower (a site's own controls,
or its patients' channels outside the SOZ and resection):

| File | Columns |
|---|---|
| `reference_channels.csv` | `recording_id, channel` (the key into `features.csv`), `x, y, z, space` (MNI for `REGION=radius`), `roi` (optional), `patient` (the unit left out; default the BIDS subject), `normative` (true/false), and optionally `spike_24h` (> 24 = irritative zone) or `irritative` (true/false) |
| `features.csv` | the cohort's `features.csv` rows |
| `coherence.csv` | optional: the cohort's `coherence.csv` rows, for edge-level connectivity |
| `provenance.json` | optional: ieeg-bandpower's, so the spectral settings can be checked |

The brief named the key `channel_id`; it is `recording_id` + `channel` here, because that is the
key of `features.csv`. Build it with:

```sh
python -m scripts.build_reference --ieeg <ieeg-prep output>/ieeg --features <ieeg-bandpower output> --out reference/
python -m scripts.build_reference ... --all-normative       # a cohort of controls: every channel normative
python -m scripts.build_reference --from-atlas HUP_atlas_final.mat --out reference/   # the atlas as Format A
```

A channel is normative unless its `_channels.tsv` marks it `soz` or `resected` (or always, with
`--all-normative`); `spike_24h` comes from `_channels.tsv` or `_electrodes.tsv` when present.
`--from-atlas` writes the atlas with its default spectral settings; the test checks that Format A
built this way gives the same z-scores as Format B.

### Reference Format B (the HUP atlas `.mat`)

`HUP_atlas_final.mat` is a MATLAB 5 file (not v7.3/HDF5), read with `scipy.io.loadmat`:
`wake_clip` (12000 samples = 60 s at 200 Hz × 3431 bipolar channels, filtered 0.5–80 Hz),
`mni_coords` (MNI-152 bipolar centroids), `patient_no` (60 patients), `soz_ch` (462),
`resected_ch` (609), `spike_24h`. A v7.3 file is read with h5py if it is installed (it is not in
this image) and otherwise refused with a clear message. It is converted per patient with the
same feature code as ieeg-bandpower (band power and, within each patient, coherence of every
channel pair), once per spectral setting (~1.5 s), and kept in memory. Its channels are named
`ch0001`… (as ieeg-prep names them), its patients `HUPatlas001`…, so ieeg-prep's
`sub-HUPatlas<NNN>` subjects are left out of their own reference.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `REGION` | `auto` | How a channel's reference set is chosen. `radius`: normative channels whose MNI centroid is within `RADIUS_MM`. `roi`: normative channels with the same `roi` label (test `_channels.tsv` and reference). `auto`: `roi` when the test channel has a `roi` and the reference has `roi` labels, else `radius`. |
| `RADIUS_MM` | `20` | Neighbourhood radius for `REGION=radius`. |
| `MIN_REFERENCE` | `10` | Fewer reference channels than this: `region = insufficient`, no z. Also the minimum number of reference edges for an edge z. |
| `LEAVE_PATIENT_OUT` | `true` | Reference channels of the tested patient are left out: the reference `patient` equal to the test subject, or the atlas patient (`sub-HUPatlas<NNN>`, or `atlas_patient` in `_electrodes.tsv`). |
| `EXCLUDE_IRRITATIVE` | `true` | Drop irritative-zone channels (`spike_24h > 24`, i.e. ≥ 1 spike per hour, the paper's definition; or `irritative = true` in Format A) from the reference, as Bernabei et al. did. On the atlas this leaves 854 of 2626 normative channels. |
| `SPECTRAL_FEATURES` | `normpow` | Which spectral features make up the spectral family summaries (`max_abs_z`, `spectral_*`, `combined`): comma-separated from `normpow` (Bernabei 2022), `relpow`, `logpow`. Every feature is z-scored and written to `abnormality.csv` regardless. |
| `EDGES` | `true` | Edge-level connectivity from `coherence.csv`. `false`: the connectivity score falls back to coherence node strength (`coh_<band>`). |
| `LABEL` | (empty) | When set, outputs go to `OUTPUT_DIR/<LABEL>/`. |

`FEATURE` (one feature chosen for z-scoring) is gone: every feature is z-scored now. The old
`norm_median` is `normpow_<band>`, `rel_power` is `relpow_<band>`, `log_abs_power` is
`logpow_<band>`; `taylor2022` and `log_rel_power` are not computed any more.

## What it does

1. **Reference** (Format A or B, above). The pool is the normative channels (`normative`; for
   the atlas `soz_ch == 0 and resected_ch == 0`), minus the irritative zone by default.
2. **Region per test channel** (`REGION`): coordinates from `_channels.tsv` (`x, y, z` with
   `space` starting with `MNI`, `IXI549Space`, `ICBM` or `fsaverage`; empty `space` is taken as
   MNI with a note; any other space, e.g. ieeg-prep's `native`, counts as no coordinates), or its
   `roi`. Reference channels of the tested patient are removed (`LEAVE_PATIENT_OUT`).
3. **Node z-scores**, for every feature present in both the input and the reference:
   `z = (x − mean) / sd` over the region's reference channels (sd with ddof 1), `abs_z = |z|`.
4. **Edge z-scores** (Bernabei 2022, "Bivariate features": each edge "standardized against the
   distribution of normative atlas edges between the same pair of brain regions"): the coherence
   of each test channel pair (i, j) per band against every reference edge (a, b) between two
   normative channels of one reference recording with a in the region of i and b in the region
   of j, or the other way round (each reference edge counted once). With `REGION=roi` that is
   exactly the paper's region pair. With the default radius regions, a "region pair" is the pair
   of 20 mm neighbourhoods around the two test channels; this keeps the reference edges at about
   the test edge's length, which matters because coherence falls with distance. We chose this
   over a coarse parcellation because the atlas has no region labels and no openly licensed
   parcellation is bundled (see below); it needs no extra data and degrades gracefully where
   the reference is sparse.
5. **Connectivity per channel and band** (`conn_<band>`): the **75th percentile of |z| over the
   channel's scored edges**, the paper's node feature ("captured a high level of abnormality but
   was not as sensitive to outliers"; MATLAB `prctile`, here numpy `method="hazen"`, which is the
   same interpolation), from 3 scored edges up. Without edges (`EDGES=false`, no
   `coherence.csv`, or no scored edge) the connectivity score falls back to the |z| of coherence
   node strength `coh_<band>`.
6. **Summaries per channel and family** (`channel_abnormality.csv`, below), and **validation**
   when `_channels.tsv` has `soz` / `resected` labels: the AUC of each family score per
   recording and pooled (`validation.csv`).

### Region assignment: coordinate neighbourhoods, not atlas regions

The atlas has MNI coordinates but no region labels. Bernabei et al. 2022 assigned channels to
the **AAL** atlas (automated anatomical labelling), merged sparsely sampled neighbouring gyri to
20 regions per hemisphere, dropped cerebellum and basal ganglia, and z-scored per region (the
merge table is in their supplement). We found no clear open licence for the AAL atlas files,
and the merged-region table is not machine-readable, so by default this app uses **coordinate
neighbourhoods**: every test channel gets its own reference set, the normative channels within
`RADIUS_MM` (default 20 mm, roughly the size of a merged AAL region) in MNI space. A site with
its own labels (a `roi` column in both `_channels.tsv` and a Format A reference) gets
region-based z-scores with `REGION=auto` or `roi`.

Effect of the radius on the atlas (band power, leave-one-patient-out, `normpow`, then called
`norm_median`, `MIN_REFERENCE=10`, irritative zone kept):

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
(`within-20mm` or `roi:<label>`), not an anatomical label. With the irritative zone excluded
(now the default), fewer channels have 10 reference channels within 20 mm: 37 of 240 channels
of the five patients in the quick check below were `insufficient`.

## Outputs

**`abnormality.csv`**: one row per channel and feature.

| Column | |
|---|---|
| `recording_id`, `channel` | as in `features.csv` |
| `feature` | `relpow_<band>`, `normpow_<band>`, `logpow_<band>`, `coh_<band>` (node z), any other feature of the input, and `conn_<band>` (edge-based connectivity) |
| `z` | (value − reference mean) / reference sd; empty when not scored, and always empty for `conn_<band>` (unsigned) |
| `abs_z` | \|z\|; for `conn_<band>` the 75th percentile of the channel's edge \|z\| |
| `region` | `within-<R>mm` or `roi:<label>` (scored), `insufficient` (fewer than `MIN_REFERENCE` reference channels), `unknown` (no MNI coordinates / no roi) |
| `n_reference` | reference values used for this z; for `conn_<band>` the number of scored edges |
| `family` | `spectral`, `coherence` (node strength), `connectivity` (edge-based), `other` |

**`channel_abnormality.csv`**: one row per channel.

| Column | |
|---|---|
| `recording_id`, `channel` | |
| `max_abs_z`, `max_band` | **kept for compatibility**: the spectral family's largest \|z\| (the `SPECTRAL_FEATURES` features, default `normpow_*`) and its band |
| `region`, `n_reference` | region, and the number of reference channels in it |
| `spectral_max_abs_z`, `spectral_mean_abs_z`, `spectral_max_feature` | max and mean \|z\| over the spectral family's features (= `max_abs_z`) |
| `coherence_max_abs_z`, `coherence_mean_abs_z` | over `coh_<band>` (node strength) |
| `connectivity_max`, `connectivity_mean` | over the five `conn_<band>` values (the paper's node feature); `connectivity_mean` is the connectivity score |
| `connectivity_source` | `edges`, or `node-strength` when it fell back to the coherence family |
| `combined` | mean of the paper's ten features: the spectral family's \|z\| per band and the connectivity per band |
| `other_max_abs_z` | max \|z\| over features of no known family (empty when there are none) |

**`edge_abnormality.csv`**: one row per channel pair and band: `recording_id, channel_a, channel_b,
band, coherence, z, abs_z, n_reference` (reference edges). Grows with the square of the channel
count.

**`validation.csv`** (when labels are known): `scope` (a recording id or `all`), `label` (`soz`,
`resected`), `score` (`spectral_max_abs_z`, `spectral_mean_abs_z`, `coherence_mean_abs_z`,
`connectivity_mean`, `combined`, `other_max_abs_z`), `n_positive`, `n_negative`, `n_unscored`,
`auc` (Mann–Whitney; > 0.5 means the labelled channels are more abnormal). Univariate and
unsupervised: no score is trained on the labels.

**`provenance.json`**: app and version, tool versions, parameters, the reference (kind, path,
channels, patients, normative and irritative channels, features, normative edges, the rule,
seconds), inputs found and per recording: subject, reference patients left out, channels
scored/insufficient/unknown, features, pairs, notes.

## Validation

**Status.** The band-power numbers below are the leave-one-patient-out validation over all 60
atlas patients, computed before this app was made generic (then `FEATURE=norm_median`, now the
`normpow_*` features; same code, irritative zone kept, the default then). **The full
leave-one-patient-out validation of the coherence features (node strength, edge-based
connectivity, combined) is pending**; only the quick end-to-end check on five patients below has
been run. `scripts/validate_atlas.py` still covers the band-power grid only.

### Quick check: coherence and connectivity on five atlas patients (pipeline, Docker)

Atlas patients 1, 15, 29, 44 and 60 as delivered by ieeg-prep
(`processor-ieeg-prep/data/atlas-out`), through ieeg-bandpower and ieeg-normative with default
parameters (irritative zone excluded, `REGION=auto` → radius 20 mm, edges on). 240 channels: 203 with a
spectral score, 217 with an edge-based connectivity score (7 more through the node-strength
fallback). Pooled AUC:

| Score | AUC SOZ (n = 27–29) | AUC resected (n = 33–34) |
|---|---|---|
| `spectral_max_abs_z` | 0.416 | 0.410 |
| `spectral_mean_abs_z` | 0.387 | 0.372 |
| `coherence_mean_abs_z` (node strength) | 0.408 | 0.437 |
| `connectivity_mean` (edges, 75th percentile) | 0.365 | 0.410 |
| `combined` | 0.345 | 0.352 |

Five patients with 27 SOZ channels say nothing about performance; this check is about whether
the numbers are sane. They are. The z-scores are calibrated: on these patients' normative
channels the median |z| per feature is 0.55–0.92 (0.67 expected for N(0, 1)), the median signed
z is within ±0.3, the median of the max over five bands is 1.66 (1.68 expected) and the median
connectivity score 1.03 (the 75th percentile of |N(0, 1)| is 1.15). Scoring the same patients
directly from the atlas clips gives the same AUCs to ±0.01 (0.407, 0.387, 0.406, 0.356, 0.341 for
SOZ). The small differences come from ieeg-prep, which filters the atlas clips again (a 4th-order
0.5 Hz high-pass and a 60 Hz notch), while the reference is computed from the clips as stored
(see *Limitations*). Bernabei et al. reached AUC 0.68 with connectivity alone and 0.77 combined,
but with a **random forest** trained on the ten |z| features (patient-level 10-fold
cross-validation, target: resected SOZ). An unsupervised univariate score cannot weight or
combine features and will be lower.

### Band power: leave-one-patient-out over the whole atlas (`make validate-atlas`)

Every one of the 60 atlas patients scored against the other 59 (`scripts/validate_atlas.py`,
the app's own code). AUC of `max_abs_z`, pooled over 3431 channels (per-patient median in
brackets), `RADIUS_MM=20`, `MIN_REFERENCE=10`:

| Feature | Bands | Irritative zone in reference | AUC SOZ | AUC resected |
|---|---|---|---|---|
| `norm_median` (now `normpow`) | conventions | kept | **0.480** (0.434) | **0.474** (0.416) |
| `norm_median` | bernabei2022 | kept | 0.459 (0.400) | 0.447 (0.422) |
| `norm_median` | conventions | excluded (now the default) | 0.471 (0.456) | 0.483 (0.472) |
| `taylor2022` | conventions | kept | 0.451 (0.389) | 0.482 (0.478) |
| `log_rel_power` | conventions | kept | 0.479 (0.431) | 0.453 (0.441) |

Per band, signed z (higher z = more of that band than the neighbourhood), `norm_median`, 20 mm:
SOZ AUC delta **0.576**, theta 0.510, alpha 0.451, beta 0.456, gamma 0.455; resected delta
0.583. SOZ channels carry relatively more delta and less alpha/beta/gamma, but the maximum
|z| over bands mixes these small shifts with the ordinary spread of normal channels (median
max |z|: SOZ 1.43, normative channels of other patients 1.47).

### Band power through the pipeline, 5 atlas patients (ieeg-bandpower → ieeg-normative, Docker; before the restructure)

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
strong effects they report are for coherence (e.g. mesial temporal SOZ: mean coherence |z|
1.87 ± 0.44 against 1.24 ± 0.39 in uninvolved tissue); the app scores coherence now, and its
validation is pending (above).

Conclusion: no bug was found. In this atlas, under leave-one-patient-out, single-channel
spectral |z| is near chance. That agrees with the paper's own regional and largely negative
univariate spectral results. That finding is why the app now also scores coherence.

On the demo seizure patients (Discover 530 via ieeg-prep) every channel is `unknown`: their
electrode coordinates are in each patient's native space, not MNI (see *Limitations*).

## Runtime

`computeTypes: [standard, lambda]`, 1 vCPU / 2 GiB, 900 s timeout. Measured in Docker
(linux/amd64 **emulated on an M-series Mac**, `--cpus 2 --memory 2g`), the real atlas as
reference:

| Input | Wall time | cgroup `memory.peak` |
|---|---|---|
| 5 atlas patients via ieeg-prep (240 channels, 6837 pairs), features + coherence | 6.3 s (3.5 s in the app: atlas load 1.6 s, atlas features with coherence 1.4 s, scoring 0.5 s) | 0.93 GB |
| 24 demo recordings (950,000 coherence rows, 80 MB) + the 5 atlas patients, fan-in | 16.7 s (13.4 s in the app) | 1.26 GB |
| before the restructure, band power only: 5 atlas patients (284 channels) | 7.9 s | 1.18 GB |

The peak includes the page cache of the 317 MB atlas file and the input CSVs; the process holds
the atlas (~330 MB as float64), the reference features and 109,616 atlas edges, and the input
coherence tables (parsed into compact tuples; a dict per row took the fan-in run to 1.5 GB).
`computeTypes` stays `[standard, lambda]`. The image pins BLAS to one thread
(`OPENBLAS_NUM_THREADS=1`): with OpenBLAS's default pool, sized from the host's CPUs, the atlas
conversion took 50 s under a 2-CPU quota instead of 1.4 s. No other parallelism is used. Very
large inputs (many recordings with 200+ channels) grow `coherence.csv` and `edge_abnormality.csv`
with the square of the channel count; run those as `standard`. The image is 91 MB compressed
(179 MB unpacked).

## Running it locally

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python             # pytest: synthetic atlas, features and channels; Lambda handler
make run PYTHON=.venv/bin/python              # Docker over data/fixture/linked/, into data/output/
make validate-atlas PYTHON=.venv/bin/python   # band-power LOPO AUC grid over the real atlas -> data/validation-atlas.csv
.venv/bin/python -m scripts.build_reference --help   # Format A reference from a normative cohort (or the atlas)
make atlas-subjects PYTHON=.venv/bin/python   # atlas patients 1-5 as a derivative, for the pipeline check
make run-demo                                 # WORK=../workflow-epilepsy-ieeg/data/work (bandpower + prep/ieeg)
make check-vendored                           # spectral.py identical to ieeg-bandpower's
```

`tests/make_fixture.py` writes a synthetic atlas (6 patients × 24 channels in the Discover 179
layout, SOZ channels with extra delta), the `_channels.tsv` of an atlas patient and of a new
patient (two normal channels, two with a delta excess, one without coordinates, one far from
any atlas channel) and their `features.csv`, `coherence.csv` and `bandpower.csv` computed with
`spectral.py`. The tests also check edge z-scores against a brute-force computation, the 75th
percentile, ROI regions, the node-strength fallback, Format A (the atlas exported by
`build_reference.py`) against Format B, a reference built from a cohort, and an old
`bandpower.csv`-only input.

As a Lambda function: `processor/entry.py` hands over to `awslambdaric` when
`AWS_LAMBDA_RUNTIME_API` is set; `processor/handler.py` takes `inputDir`, `outputDir` and the
parameters from the payload and runs the same `main()`; a failed run raises.

## Limitations

- **Coordinates must be in MNI space.** Channels in a patient's native (CT/MRI) space cannot be
  placed in the atlas and are `unknown`. The demo seizure dataset (Discover 530) only has native
  coordinates, so ieeg-prep writes `space = native` and nothing is scored there. Registering
  electrodes to MNI is outside this step.
- **Weak discrimination, and connectivity not yet validated.** On the atlas, band-power max |z|
  is at chance for SOZ and resected channels (above). The coherence and connectivity scores
  have only been run on five patients; their 60-patient validation is pending. All scores are
  unsupervised and univariate; the paper's 0.68/0.77 came from a trained random forest. A 60-s
  wake clip per patient is a small sample of interictal activity, and abnormality maps vary over
  time (Wang et al. 2023).
- **ieeg-prep filters the atlas clips again.** Run through ieeg-prep, the atlas clips get a
  4th-order 0.5 Hz high-pass and a 60 Hz notch (Q 30) on top of their 0.5–80 Hz filtering; the
  Format B reference is computed from the clips as stored. Power at 58–62 Hz (down to 3 % at
  60 Hz) and below 1 Hz drops: `normpow_*` differs by a median 3 % (up to 32 %),
  `logpow_gamma` by up to 0.9 (log10), coherence by ≤ 0.03. The z-scores of normative channels
  stay calibrated and the AUCs move by ≤ 0.01 on five patients, but test and reference are not
  filtered identically. A fix belongs in ieeg-prep (skip the notch and high-pass for data whose
  `_channels.tsv` cutoffs already say 0.5–80 Hz) or in a Format A reference built through the
  same ieeg-prep.
- **Edge regions under the radius method** are pairs of 20 mm neighbourhoods, not anatomical
  region pairs; with the irritative zone excluded, a channel needs normative channels of the
  same reference patient within 20 mm of both ends of its edges, so sparsely sampled areas get no
  edge score (then the node-strength fallback is used).
- **Bipolar neighbours** that share a contact have inflated coherence; the reference has the
  same inflation only where its montage matches (the atlas is bipolar).
- **Reference ≠ healthy brain.** Normative channels are those outside the SOZ and the resection
  in epilepsy patients; irritative-zone channels are excluded by default (`EXCLUDE_IRRITATIVE`),
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

Normative mapping with max |z| over bands (and the former `taylor2022` feature, still in `scripts/validate_atlas.py`):

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
