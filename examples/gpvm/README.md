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

A worker sees only `/cvmfs` and its scratch disk. The job copies each input with `xrdcp` (three tries), builds, and
copies the tables out, creating the output folder. The lists in the tarball therefore hold xrootd URLs:

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
   # with the per-object tables of the small models: put BUILD_ARGS=--objs in front of the command
   ```

   A slice is done when its `build_<sample>_<a>_<b>.json` is in the output folder. `copyfail_<sample>_<a>_<b>.txt`
   lists inputs that could not be copied; they are absent from that slice's tables. Each job also leaves
   `timing_<sample>_<a>_<b>.json`: seconds for the copy-in and for the build, bytes copied, the worker and its CPU.

4. Follow the jobs and keep the time record. Run this in `tmux` right after submitting; it polls the queue once a
   minute until it is empty and writes the queue history and every state change of every job:

   ```bash
   tmux new -s tables "bash <REPO>/examples/gpvm/monitor_jobs.sh <GROUP> <output folder as mounted> <MONITOR>"
   # afterwards, with the slices copied to a local folder:
   python3 <REPO>/examples/gpvm/timing_summary.py <MONITOR> <folder with the slices> submit_time.txt
   ```

5. Merge the slices into one parquet per table:

   ```bash
   python3 <REPO>/tables/merge_tables.py <folder with the slices> <sample> <OUT>/merged
   ```

Status of these scripts: this job pattern built 4,000 merged CAFs on FermiGrid in 40 jobs of 100 files (no held or
failed job, no failed copy), and the tables are identical to a build of the same files on the interactive node. That
run used a submit script written for it; `submit_tables.sh` here carries the same options but has itself only been
used with `--dry-run`.

What that run cost, to plan with: all jobs started within about 3 minutes and the last one was done 21 minutes after
submission. Per file, the copy-in took 1.0 to 1.3 s and the build 2.7 to 8.5 s depending on the worker's CPU (4.7 s
on average), so allow about 6 s per file and a factor of two between workers. Merging and scoring 4,000 files' tables
afterwards took under two minutes.

## The evaluation standard

```bash
python3 <REPO>/evaluation/eval_standard.py <OUT>/merged <BNB sample> <nueCC-filtered sample> \
        --drop-repeated-keys --out standard.json
```

It reads the exposure from the `files_` tables and prints the progressive efficiency / purity table and the per-bin
curves (`docs/evaluation_standard.md`). `--score <column>` puts a retrained model's score through the same code;
`--file-ids <list>` restricts both samples to the file ids in a list, for example the list without the hold-out files.

## Things that cost time here

- Do not build by reading the merged CAFs through the storage mount. A read through the mount hung on a file that
  was online and that `xrdcp` copied in a fraction of a second; the builder has no per-file time limit and writes
  only at the end, so one stuck read blocked and lost a whole run. Copy with `xrdcp` first, also on the interactive
  node (`table_job.sh` does it; run it slice by slice with `INPUT_TAR_DIR_LOCAL`, `_CONDOR_SCRATCH_DIR` and `PROCESS`
  set by hand).
- uproot cannot read some branches of a small fraction of files (a basket stored inside the branch header); ROOT
  reads them. The builder flags such files (`label_unreadable`, `truth_pf_unreadable`) instead of failing; count them.
- (run, subrun, event) is not unique across the files of a filtered MC sample. Always key on `file_id` as well.
- Data files have no truth match; an older data production has no `T_match` tree at all. Both mean "no partner".
- Check a new builder version on a few files against tables you trust with `tables/compare_tables.py` before a full run.
