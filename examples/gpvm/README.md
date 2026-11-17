# Building tables and running the evaluation at Fermilab (gpvm and FermiGrid)

The inference samples (the Fermilab productions and the data) can only be read at Fermilab, so their tables are built
there with the same `tables/build_tables.py` as the training tables. This folder holds the commands, with placeholders
for everything that belongs to a site or a person.

| placeholder | meaning |
|---|---|
| `<REPO>` | your clone of this repository |
| `<LISTS>` | a folder with one `<sample>.txt` per sample, one merged CAF per line |
| `<SAMPLE_ROOT>` | the folder of one sample: `<SAMPLE_ROOT>/<unit id>/<name>.wc.flat.caf.root` |
| `<OUT>` | where the tables go |
| `<GROUP>` | your experiment's jobsub group |
| `<MOUNT_PREFIX>`, `<XROOTD_PREFIX>` | the same storage as a mounted path and as an xrootd URL |

## What you need

- An EL9 interactive node. Its system Python is 3.9; the builder is written for it.
- numpy, pandas, pyarrow, uproot. The node does not have all of them, and a grid worker has none, so they are
  installed into a folder and used through `PYTHONPATH`:

  ```bash
  python3 -m pip install --target <PYDEPS> numpy pandas pyarrow uproot awkward
  export PYTHONPATH=<PYDEPS> PYTHONNOUSERSITE=1
  ```

  The versions the tables were checked with: Python 3.9.25, uproot 5.6.9, numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0.
  The same builder gives identical tables under Python 3.12 with uproot 5.7.7 (`tables/compare_tables.py`).

## File ids

`file_id` is half of every key, and the lists of excluded files (hold-out, earlier training rounds) are matched on it.
In the Fermilab productions each merged CAF sits in its own folder named after the production unit, so pass
`--file-id-from dirname` there. The builder's default (`basename`, the file name) is meant for productions whose
folders do not identify the file. Use one rule per sample, and the same rule for the lists you match against.

## On the interactive node (a few hundred files)

```bash
B=<REPO>/tables/build_tables.py
python3 $B fg1-mc-nuecc <OUT> --list <LISTS>/fg1-mc-nuecc.txt --slice 0:150 --nproc 3 \
        --file-id-from dirname --root <SAMPLE_ROOT>
```

Measured: about 2.7 core-seconds per MC file; the node has 4 cores. A read of all tagger branches with
`library='ak'` is some 30 times slower than the `library='np'` the builder uses. Anything beyond a few thousand files
belongs on the grid.

The builder writes `cands_`, `truth_`, `files_<sample>[_<tag>].parquet` and a `build_<sample>[_<tag>].json` summary.
Look at `errors`, `label_unreadable_files` and `partner_missing` in it before using the tables.

## On FermiGrid

A worker sees only `/cvmfs` and its scratch disk. The job copies each input with `xrdcp`, builds, and copies the
tables out. The lists in the tarball therefore hold xrootd URLs:

```bash
sed 's|^<MOUNT_PREFIX>|<XROOTD_PREFIX>|' <mounted list> > <LISTS>/<sample>.txt
```

1. Build the tarball (the repository scripts, the Python packages, the lists):

   ```bash
   bash <REPO>/examples/gpvm/make_bundle.sh <LISTS> <BUILD> <BUILD>/tables.tar.gz
   ```

2. Emulate one job on the interactive node, in the worker image, before submitting. For this step the lists hold
   plain paths and the job copies with `cp`; the token and `xrdcp` part is the one thing only a real job tests.

   ```bash
   bash <REPO>/examples/gpvm/emulate.sh <BUILD>/tables.tar.gz <sample> 4 <WORK> <data mount> 0
   ```

3. Submit, 100 files per job:

   ```bash
   htgettoken -a <vault server> -i <GROUP>
   GROUP=<GROUP> bash <REPO>/examples/gpvm/submit_tables.sh <BUILD>/tables.tar.gz <sample> 100 <number of files> \
         <XROOTD_PREFIX>/<your scratch area>/<sample>          # add --dry-run to print the command only
   ```

   A slice is done when its `build_<sample>_<a>_<b>.json` is in the output folder. `copyfail_<sample>_<a>_<b>.txt`
   lists inputs that could not be copied; they are absent from that slice's tables. To rerun a slice, remove what it
   left in the output folder first: the storage does not overwrite.

4. Merge the slices into one parquet per table:

   ```bash
   python3 <REPO>/tables/merge_tables.py <folder with the slices> <sample> <OUT>/merged
   ```

Status of these scripts: `table_job.sh` was run through `emulate.sh` in the worker image and its tables are identical
to a direct build of the same files. `submit_tables.sh` has not been used for a production run yet; the same job
pattern (tarball, worker image, `xrdcp` with the job's token) ran on FermiGrid in a separate test.

## The evaluation standard

```bash
python3 <REPO>/evaluation/eval_standard.py <OUT>/merged <BNB sample> <nueCC-filtered sample> \
        --drop-repeated-keys --out standard.json
```

It reads the exposure from the `files_` tables and prints the progressive efficiency / purity table and the per-bin
curves (`docs/evaluation_standard.md`). `--score <column>` puts a retrained model's score through the same code;
`--file-ids <list>` restricts both samples to the file ids in a list, for example the list without the hold-out files.

## Things that cost time here

- uproot cannot read some branches of a small fraction of files (a basket stored inside the branch header); ROOT
  reads them. The builder flags such files (`label_unreadable`, `truth_pf_unreadable`) instead of failing; count them.
- (run, subrun, event) is not unique across the files of a filtered MC sample. Always key on `file_id` as well.
- Data files have no truth match; an older data production has no `T_match` tree at all. Both mean "no partner".
- Check a new builder version on a few files against tables you trust with `tables/compare_tables.py` before a full run.
