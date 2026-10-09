"""
covflow Run 3 trainings, one per data-taking epoch and per muon.

Read by batch/submit_run3_epochs.py, which checks every input on /pnfs,
renders one self-contained bash config per (epoch, muon) in the format
batch/train_gpu.sh already reads, and submits each with batch/submit_gpu.sh.
Nothing in this file is read by the jobs themselves: what a run actually used
is the rendered job.conf copied into its task directory.

An epoch is a set of data eras matched to the MC campaign simulated for them,
following the PPD summary tables (PdmV Run 3 analysis twiki):

    2022 C D        <-> Run3Summer22        2023 C <-> Run3Summer23
    2022 E F G      <-> Run3Summer22EE      2023 D <-> Run3Summer23BPix
    2024 C-I        <-> RunIII2024Summer24
    2025 B-G, 2026 B D <-> RunIII2024Summer24 (no 2025/2026 MC; PPD: "use
                        Summer24 for now"). Each year is its OWN training: the
                        point of training per epoch is to follow the detector
                        in time, so sharing the MC is no reason to merge years.
    2026 C (low PU / special runs) is excluded on purpose.

Prompt data: every processing version is used (PPD rule), hence e.g. 2022D v1
and v2, 2025C v1 and v2. Reprocessings with several processing-string versions
(2023C 22Sep2023_v1..v4, 2024I MINIv6NANOv15_v2/_v3) cover different run
ranges and are all used.

Placeholders, filled per job: {mu} -> mu1 / mu2, {pu_year} -> the epoch's.
"""

# ---------------------------------------------------------------------------
# where things are
# ---------------------------------------------------------------------------
# REPO: the covflow checkout the JOBS run from. None = the checkout this file
# lives in (the usual case). The submitter refuses a REPO whose
# train_covflow.py does not know --context-edges.
REPO = None
CONDA_ENV = 'covflow'

INPUT_DIR = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/covflow_dimuon_ntuples_02oct26'
SE_HOST   = 't3dcachedb03.psi.ch:1094'     # same door as the Bmmm ntuple jobs

OUT_ROOT = '/work/manzoni/correct_track_covariance/covflow-runs'
TAG      = 'run3_epochs_05oct26_trgmatch_e1200'  # runs go to OUT_ROOT/TAG/<epoch>/<mu>

MUONS = ('mu1', 'mu2')

# ---------------------------------------------------------------------------
# the MC <-> data map
# ---------------------------------------------------------------------------
# Explicit file names, no globs: a file added to INPUT_DIR later cannot slip
# into a training. The submitter also checks the converse -- every data file
# in INPUT_DIR belongs to exactly one epoch.
EPOCHS = {
    '2022_preEE': dict(
        mc      = ['hb_Summer22.root'],
        pu_year = '2022',
        data    = ['data2022C_PromptReco_v1.root',
                   'data2022D_PromptReco_v1.root',
                   'data2022D_PromptReco_v2.root'],
    ),
    '2022_postEE': dict(
        mc      = ['hb_Summer22EE.root'],
        pu_year = '2022',
        data    = ['data2022E_PromptReco_v1.root',
                   'data2022F_22Sep2023_v1.root',
                   'data2022G_22Sep2023_v1.root'],
    ),
    '2023_preBPix': dict(
        mc      = ['hb_Summer23.root'],
        pu_year = '2023',
        data    = ['data2023C_22Sep2023_v1_v2.root',
                   'data2023C_22Sep2023_v2_v1.root',
                   'data2023C_22Sep2023_v3_v1.root',
                   'data2023C_22Sep2023_v4_v1.root'],
    ),
    '2023_postBPix': dict(
        mc      = ['hb_Summer23BPix.root'],
        pu_year = '2023',
        data    = ['data2023D_22Sep2023_v1_v1.root',
                   'data2023D_22Sep2023_v2_v1.root'],
    ),
    '2024': dict(
        mc      = ['hb_Summer24.root'],
        pu_year = '2024',
        data    = ['data2024C_MINIv6NANOv15_v1.root',
                   'data2024D_MINIv6NANOv15_v1.root',
                   'data2024E_MINIv6NANOv15_v1.root',
                   'data2024F_MINIv6NANOv15_v3.root',
                   'data2024G_MINIv6NANOv15_v1.root',
                   'data2024H_MINIv6NANOv15_v1.root',
                   'data2024I_MINIv6NANOv15_v2_v2.root',
                   'data2024I_MINIv6NANOv15_v3.root'],
    ),
    '2025': dict(
        mc      = ['hb_Summer24.root'],
        pu_year = '2025',
        # 2025B is not in the PPD 2025 table; kept on purpose (~0.3% of 2025)
        data    = ['data2025B_PromptReco_v1.root',
                   'data2025C_PromptReco_v1.root',
                   'data2025C_PromptReco_v2.root',
                   'data2025D_PromptReco_v1.root',
                   'data2025E_PromptReco_v1.root',
                   'data2025F_PromptReco_v1.root',
                   'data2025F_PromptReco_v2.root',
                   'data2025G_PromptReco_v1.root'],
    ),
    '2026': dict(
        mc      = ['hb_Summer24.root'],
        pu_year = '2026',
        data    = ['data2026B_PromptReco_v1.root',
                   'data2026D_PromptReco_v1.root'],
    ),
}

# In INPUT_DIR but deliberately NOT trained on: the Bc samples are kept as an
# independent check of the Hb-trained correction.
VALIDATION_ONLY = ['bc_Summer22.root', 'bc_Summer22EE.root', 'bc_Summer23.root',
                   'bc_Summer23BPix.root', 'bc_Summer24.root']

# ---------------------------------------------------------------------------
# training settings, identical for every epoch and muon
# ---------------------------------------------------------------------------
# From train_gpu_mu2_2Mevents_fourconditional_fullrun3.conf, with four
# deliberate changes: --max-events 1M (was 2M), the 5-D context below, the MC
# pileup weight, and the trigger requirement in the selection. Any difference between two epoch trainings can therefore
# only come from the inputs.
TRAINING = dict(
    tree       = 'tree',
    cov_prefix = '{mu}_cov_',

    # pt and eta of bestTrack() (the track whose covariance is corrected, and
    # what Bmmm's CovFlowCorrector evaluates at application), the pixel hit
    # count, and where the first pixel hit is -- as TWO axes, barrel layer and
    # endcap disk, never the merged pix_first_layer code (see --context-edges).
    context = ['{mu}_best_trk_pt',
               '{mu}_best_trk_eta',
               '{mu}_n_pix_hit',
               '{mu}_pix_first_b_layer',     # 0 = no BPix hit, 1-4 = layer
               '{mu}_pix_first_e_disk'],     # 0 = no FPix hit, 1-3 = disk
    log_pt  = '{mu}_best_trk_pt',

    # categorical cells for the two discrete axes, in every validation binning
    #   b_layer: {0} {1} {2} {3,4}     e_disk: {0} {1} {2,3}
    context_edges = {'{mu}_pix_first_b_layer': [0.5, 1.5, 2.5],
                     '{mu}_pix_first_e_disk'  : [0.5, 1.5]},

    data_weight = '',                        # no sWeights
    mc_weight   = 'pu_weight_{pu_year}',     # NaN in the other years' branches

    # 1200, not 800: in the first round (2022_preEE, 800 epochs) both flows
    # took their best snapshot at epoch 791-798 and val NLL was still falling
    # steeply between epochs 400 and 600 -- the end plateau was the cosine LR
    # reaching zero, not convergence. 800 epochs took ~63 min there; from the
    # 2026 pilot (~6.8 s/epoch per flow at the 1M cap) 1200 epochs is ~4.5 h of
    # training in the largest epochs, inside the 8 h wall time. Do not go
    # further without a time budget in the training loop: products are only
    # written after training.
    epochs     = 1200,
    batch_size = 8192,
    lr         = '2e-3',
    transforms = 4,
    hidden     = '128 128',
    bins       = 8,
    seeds      = [0],

    # Same events for mu1 and mu2 trainings: the selection is event-level.
    #
    # Trigger: HLT_DoubleMu4_3_LowMass, the path of the ParkingDoubleMuonLowMass
    # data, required for BOTH samples, with BOTH muons matched to its dimuon
    # filter (hltDisplacedmumuFilterDoubleMu43LowMass, dR < 0.15, the
    # mu<N>_<path>_tag branches). The data ntuples were produced without
    # --savenontrig, so the path decision is already applied there and the cut
    # is a no-op; the MC was produced with --savenontrig and needs it. The
    # muon-level match was applied in neither. It matters for covflow beyond
    # efficiency: the filter is a displaced-vertex filter, so it can select on
    # the online vertex quality, which is correlated with the track errors
    # themselves -- applied to data only, that is a data/MC covariance
    # difference the flow would learn and then "correct" in the analysis.
    selection    = ('(HLT_DoubleMu4_3_LowMass > 0.5) & '
                    '(mu1_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
                    '(mu2_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
                    '(mu1_pt > 4.5) & (abs(mu1_eta) < 2.4) & (mu1_id_medium>0.5) & '
                    '(mu2_id_medium>0.5) & (abs(mass-3.0969)<0.1) & '
                    '((lxy * cos2d /pt * 3.0969)>0.008)'),
    mc_selection = '(abs(mu1_gen_pdgid)==13) & (abs(mu2_gen_pdgid)==13)',

    # everything train_gpu.sh does not already put on the command line.
    # bins per context dim: pt, eta, n_pix_hit, b_layer, e_disk; the last two
    # entries are superseded by context_edges and only document the cell count.
    # Closure: 3,3,2 x 4 x 3 = 216 cells with >= 200 tracks each. With
    # 4,4,3 x 4 x 3 = 384 cells and >= 400, only 53-55 cells survived in
    # 2022_preEE (239k data tracks), too few for a meaningful binned closure,
    # and the trigger requirement shrinks the samples further.
    extra_args = [
        '--features', 'block',
        '--patience', '50',
        '--no-distill',
        '--max-events', '1000000',
        '--closure-bins', '3,3,2,4,3',
        '--closure-min-count', '200',
        '--plot-bins', '4,3,2,4,3',
    ],
)

# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------
# 8 h: longer requests wait much longer for a gpu node. Note that
# train_covflow.py writes its products only at the end, so a run cut by the
# wall clock leaves nothing behind -- check Elapsed on the first ones.
SLURM = dict(
    time    = '8:00:00',
    mem     = '64G',
    min_mem = 16384,            # MB, floor checked by train_gpu.sh
    proxy_min_valid = '48:00',  # HH:MM, checked at submission
)
