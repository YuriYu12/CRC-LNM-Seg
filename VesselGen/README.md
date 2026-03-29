# Vessel Generation

## prepare images
CT, its organ segmentations (from TotalSegmentator), its colon scribbles (from PRNet), a pseudo image that erases the original vessels from the CT.

- save CT to folder `f1`, `f1` folder structure `<person>/<series>/<phase>.nii.gz`
- obtain its organ segmentations with TotalSegmentator and save to folder `f1` as `<person>/<series>/<phase>_totalseg.nii.gz`
- obtain its colon segment scribbles and save to folder `f3`
- run `preprocess_cmu_v2()` in `process/preprocess.py` with `BASE_PATH=f1`, `SCRIBBLE_PATH=f3` and set output path `MOD_PATH`. After data loading and selecting the CE image for each person, it will output a `people_dict` that has structured data and feed this dict to `preprocessor()` to dump a CT with *no mensenteric vessels* under `MOD_PATH`

## generate vessels
run `trace/augmentor.py`, call `PipelineAugmentor(mode=-1, fname=os.path.basename(ct_path), dirname=os.path.dirname(ct_path))`
- `ct_path` is the processed CT path in `MOD_PATH`.
- `save_root` is the output path.