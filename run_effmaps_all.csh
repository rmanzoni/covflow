#!/bin/tcsh
#
# run_effmaps_all.csh -- pixel hit-efficiency maps for every covflow epoch,
# one after the other. Meant to run inside screen on the UI (t3ui07), where
# /pnfs is mounted.
#
#   screen -S effmaps
#   tcsh run_effmaps_all.csh                      # all 7 epochs
#   tcsh run_effmaps_all.csh 2024 2025            # only these
#   (detach: Ctrl-a d   reattach: screen -r effmaps)
#
# For each epoch:  output in   <REPO>/effmaps_<epoch>/
#                  log in      <REPO>/effmaps_logs/effmaps_<epoch>.log
# A summary table (epoch, OK/FAILED, minutes) is printed at the end and
# written to effmaps_logs/summary_<timestamp>.txt.
#
# An epoch counts as OK only if pixel_eff_maps.py wrote its .json in this
# run. The exit status cannot be used: the output goes through tee, and in
# tcsh $status is then the status of tee.

# ---------------------------------------------------------------- settings
set REPO   = /work/manzoni/correct_track_covariance/covflow
set CONDA  = /work/manzoni/miniconda3
set ENV    = covflow
set EPOCHS = (2022_preEE 2022_postEE 2023_preBPix 2023_postBPix 2024 2025 2026)

# options shared by every epoch (same as the 2026 test run)
set OPTS = (--branch vx=pv_x --branch vy=pv_y --z0-from dz_pv --png)

# ------------------------------------------------------------------ checks
if ( $#argv > 0 ) set EPOCHS = ($argv)

if ( ! -d $REPO ) then
    echo "ERROR: repository $REPO not found"
    exit 1
endif
cd $REPO

if ( ! -f pixel_eff_maps.py ) then
    echo "ERROR: $REPO/pixel_eff_maps.py not found"
    exit 1
endif

# the polar-plot fix (full disks) must be in
grep -q "set_thetalim" pixel_eff_maps.py
if ( $status != 0 ) then
    echo "ERROR: pixel_eff_maps.py predates the polar-plot fix (disks would show"
    echo "       only half). Copy the latest version into $REPO first."
    exit 1
endif

# disk acceptance at the measured outer edge (~15 cm), not the nominal 16.1
grep -q "FPIX_R=(4.5, 16.1)" pixel_eff_maps.py
if ( $status == 0 ) then
    echo "Setting the disk acceptance to 4.5 < r < 14.8 cm in pixel_eff_maps.py"
    sed -i 's/FPIX_R=(4.5, 16.1)/FPIX_R=(4.5, 14.8)/' pixel_eff_maps.py
endif
grep -n "FPIX_R=" pixel_eff_maps.py

# conda env (a script does not inherit 'conda activate' from the login shell)
if ( ! $?CONDA_DEFAULT_ENV ) setenv CONDA_DEFAULT_ENV none
if ( "$CONDA_DEFAULT_ENV" != "$ENV" ) then
    if ( ! -f $CONDA/etc/profile.d/conda.csh ) then
        echo "ERROR: $CONDA/etc/profile.d/conda.csh not found; activate $ENV yourself"
        exit 1
    endif
    source $CONDA/etc/profile.d/conda.csh
    conda activate $ENV
endif
python -c "import numpy, uproot, matplotlib" >& /dev/null
if ( $status != 0 ) then
    echo "ERROR: python of env '$ENV' lacks numpy/uproot/matplotlib"
    exit 1
endif

# unknown epoch names stop the run here, not after hours
foreach ep ($EPOCHS)
    grep -q "'$ep'" configs/run3_epochs.py
    if ( $status != 0 ) then
        echo "ERROR: epoch '$ep' is not in configs/run3_epochs.py"
        exit 1
    endif
end

mkdir -p effmaps_logs
set STAMP   = `date +%Y%m%d_%H%M%S`
set SUMMARY = effmaps_logs/summary_$STAMP.txt
set T_ALL   = `date +%s`
set N       = $#EPOCHS
set I       = 0

echo "============================================================"
echo " pixel_eff_maps.py over $N epoch(s): $EPOCHS"
echo " repo    : $REPO"
echo " python  : `which python`"
echo " options : $OPTS"
echo " started : `date`"
echo "============================================================"
printf "%-15s %-8s %8s\n" epoch result minutes > $SUMMARY

# -------------------------------------------------------------------- loop
foreach ep ($EPOCHS)
    @ I = $I + 1
    set LOG  = effmaps_logs/effmaps_$ep.log
    set JSON = effmaps_$ep/effmaps_$ep.json
    set T0   = `date +%s`

    echo ""
    echo "=== [$I/$N] $ep   started `date '+%H:%M:%S'`   log: $LOG"
    rm -f $JSON
    python pixel_eff_maps.py --epoch $ep $OPTS --out effmaps_$ep |& tee $LOG

    @ MIN = ( `date +%s` - $T0 ) / 60
    if ( -f $JSON ) then
        set RES = OK
    else
        set RES = FAILED
    endif
    echo "=== [$I/$N] $ep   $RES after $MIN min"
    printf "%-15s %-8s %8d\n" $ep $RES $MIN >> $SUMMARY
end

# ----------------------------------------------------------------- summary
@ MIN_ALL = ( `date +%s` - $T_ALL ) / 60
echo ""
echo "============================================================"
echo " finished `date`, $MIN_ALL min in total"
echo "============================================================"
cat $SUMMARY
echo ""
echo "summary: $REPO/$SUMMARY"
grep -q FAILED $SUMMARY
if ( $status == 0 ) then
    echo "Some epochs FAILED: see the last lines of their logs in effmaps_logs/."
    exit 2
endif
exit 0
