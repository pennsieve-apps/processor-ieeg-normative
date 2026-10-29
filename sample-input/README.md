# sample-input

This app needs three inputs:

- `features`: `features.csv` + `coherence.csv` + `provenance.json` from **processor-ieeg-bandpower**;
- `reference`: a Format A reference folder (`python -m scripts.build_reference`, from any
  normative cohort run through ieeg-prep and ieeg-bandpower), or the HUP Interictal iEEG Atlas,
  `HUP_atlas_final.mat` (317 MB):

  > Bernabei J, Litt B. HUP Interictal iEEG Atlas. Pennsieve Discover 179 (2021),
  > doi:10.26275/lzhy-ox20 (ODC-By 1.0). Paper: Bernabei et al., Brain 2022,
  > doi:10.1093/brain/awab480.

  `workflow-epilepsy-ieeg/scripts/fetch-sample-data.sh` downloads it to `data/atlas/`.
- `ieeg` (optional): the **processor-ieeg-prep** derivative, for MNI coordinates and labels.

```sh
cd ../workflow-epilepsy-ieeg && scripts/fetch-sample-data.sh data
# run ieeg-prep into data/work/prep/ and ieeg-bandpower into data/work/bandpower/, then:
make run-demo        # WORK=../workflow-epilepsy-ieeg/data/work ATLAS=.../HUP_atlas_final.mat
```

The demo seizure patients have native-space coordinates only, so they come out `unknown`. To
see scores, use atlas patients: `make atlas-subjects` writes patients 1-5 as an ieeg-prep-style
derivative (`data/atlas-subjects/ieeg/`); run ieeg-bandpower on it and this app on the result
(leave-one-patient-out applies automatically). `make validate-atlas` runs the
band-power leave-one-patient-out validation over all 60 atlas patients without Docker (the
coherence features' 60-patient validation is pending).

For tests and `make run`, `tests/make_fixture.py` writes a small synthetic atlas, features and
channel tables instead (`make fixture` -> `data/fixture/linked/`).
