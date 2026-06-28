"""
Global configuration. All tunables in one place.
"""

# ── LLM ──────────────────────────────────────────────────────────
LLM_PROVIDER = "openai"
LLM_API_KEY_ENV = "OPENAI_API_KEY_UMICH_DYIMOD"  # key name in .env
LLM_MODEL_ANNOTATION = "gpt-5.4-mini"  # offline annotation
LLM_MODEL_PROFILE = "gpt-5.4-mini"  # online profile updates
ANNOTATION_CONCURRENCY = 30  # parallel requests for annotation

# ── Embedding (must match what produced embeddings.npy) ──────────
# Original: BAAI/bge-m3 (1024-dim, normalized)
# We reuse the existing embeddings.npy — no re-embedding needed
EMBEDDING_MODEL = "BAAI/bge-m3"

# ── Data paths ───────────────────────────────────────────────────
CORPUS_FILE = "../topics_result/2550_posts_final.jsonl"
EMBEDDINGS_FILE = "../embeddings.npy"
POST_IDS_FILE = "../post_ids.npy"
ANNOTATED_CORPUS_FILE = "data/annotated_posts.parquet"
SUBCLUSTERS_FILE = "data/subclusters.parquet"
PAIRS_FILE = "data/contrastive_pairs.parquet"
CROSS_TOPIC_PAIRS_FILE = "data/cross_topic_pairs.parquet"
PROFILES_DIR = "data/profiles"

# ── Subclustering ────────────────────────────────────────────────
N_SUBCLUSTERS = 4  # K-means clusters within each topic

# ── Contrastive pair scoring ─────────────────────────────────────
PAIR_LAMBDA = 0.5  # penalty for other-axis gaps
PAIR_MU = 0.5  # penalty for embedding distance
PAIRS_PER_SUBCLUSTER_AXIS = 5  # top pairs to keep
FALLBACK_TO_TOPIC = True  # if subcluster has < 3 pairs, search whole topic

# ── Seeding session ──────────────────────────────────────────────
MIN_TOPICS_CONFIDENT = 2
CONFIDENCE_THRESHOLD = 0.65
MAX_STEPS = 12
SHELF_SIZE = 8  # posts shown in Mode A

# ── SAE (Option 3: within-subcluster residuals on BGE-M3) ────────
# Discovery instrument — finds candidate delivery axes beyond the 7 LLM ones.
# Not used in the serving loop. Trained offline by m4_sae.py.
SAE_DIR = "data/sae"
SAE_RESIDUALS_FILE = "data/sae/residuals.npy"
SAE_MODEL_FILE = "data/sae/sae_model.pt"
SAE_ACTIVATIONS_FILE = "data/sae/feature_activations.npy"
SAE_INTERPRETATION_FILE = "data/sae/interpretation.json"
SAE_REPORT_FILE = "data/sae/report.md"

SAE_INPUT_DIM = 1024          # BGE-M3
SAE_N_FEATURES = 32           # small to match data scale (~2550 vectors)
SAE_L1_COEF = 0.05            # L1 sparsity penalty (variable per-sample density ~30-40%)
SAE_LR = 1e-3
SAE_EPOCHS = 300
SAE_BATCH_SIZE = 64
SAE_SEED = 42

# Interpretation
SAE_TOP_K_POSTS = 20          # posts per (feature, direction) shown in report
SAE_DEAD_DENSITY = 0.01       # density below this → flagged dead
# Axis-alignment thresholds (using axis lift, not Pearson r):
#   lift = mean(axis_score | feature active) − mean(axis_score | feature inactive)
SAE_CONFIRM_LIFT = 0.20       # |lift| >= this → "confirms_axis"
SAE_PARTIAL_LIFT = 0.10       # |lift| in [0.10, 0.20) → "partial_overlap"; below → "novel_candidate"

# ── SAE (Option 2: LLM hidden states from Qwen2.5-7B) ────────────
# Same Path 1.5 architecture as Option 3, but the SAE input is no longer the
# BGE-M3 residual — it's the residual-stream activation of a base LLM read of
# each post, captured at a mid-to-late layer and mean-pooled across tokens.
# The hypothesis (Anthropic / OpenAI SAE convention): LLM internal states
# encode style/register much more richly than retrieval embeddings.
SAE_LLM_DIR = "data/sae/llm"
SAE_LLM_ACTIVATIONS_FILE = "data/sae/llm/llm_activations.npy"
SAE_LLM_RESIDUALS_FILE = "data/sae/llm/residuals.npy"
SAE_LLM_MODEL_FILE = "data/sae/llm/sae_model.pt"
SAE_LLM_FEATURE_ACTIVATIONS_FILE = "data/sae/llm/feature_activations.npy"
SAE_LLM_INTERPRETATION_FILE = "data/sae/llm/interpretation.json"
SAE_LLM_REPORT_FILE = "data/sae/llm/report.md"

SAE_LLM_MODEL_NAME = "Qwen/Qwen2.5-7B"   # base (not Instruct); Apache 2.0, no HF gate
SAE_LLM_LAYER = 18                       # 18 of 28 (~64% depth) — middle-to-late residual stream
SAE_LLM_MAX_TOKENS = 128                 # corpus max is ~63 words; 128 tokens is generous headroom
SAE_LLM_EXTRACT_BATCH = 4                # conservative for MPS unified memory
SAE_LLM_DEVICE = "mps"                   # Apple Silicon; falls back to "cpu" if MPS unavailable
SAE_LLM_DTYPE = "bfloat16"               # safer than fp16 for transformer activations

# SAE-on-LLM training hyperparameters
SAE_LLM_INPUT_DIM = 3584                 # Qwen2.5-7B hidden_size
SAE_LLM_N_FEATURES = 128                 # 4× expansion over the F=32 used for BGE-M3 residuals
SAE_LLM_L1_COEF = 0.05                   # same scale as Option 3 (residuals are globally rescaled first)
SAE_LLM_LR = 1e-3
SAE_LLM_EPOCHS = 300
SAE_LLM_BATCH_SIZE = 64
SAE_LLM_SEED = 42
SAE_LLM_NORMALIZE_RESIDUALS = True       # divide by median ||resid|| so L1 scale matches Option 3


# ── SAE2: style fingerprint + rewrite audit (9,500-post curated corpus) ──────
# This block configures the new subsystem documented in
# docs/style_sae_handover.md. It is independent of the SAE_/SAE_LLM_ blocks
# above (those were the 2,550-post Option 2/3 experiments). Dimensions are NOT
# defined here — they come from config/axes.py (the single source of truth).

# Inputs (relative to the seeding_app app root; the curated corpus and the
# global BGE matrix live at the repo root, one level up).
SAE2_CURATED_CORPUS = "../phase3_curation/out/curated_corpus.parquet"
SAE2_GLOBAL_EMBEDDINGS = "../embeddings.npy"   # (2.2M, 1024) BGE-M3, sliced by row_idx
SAE2_GLOBAL_POST_IDS = "../post_ids.npy"        # alignment check: post_ids[row_idx] == post_id

# Outputs
SAE2_DIR = "data/sae2"
SAE2_DATASET_FILE = "data/sae2/dataset.parquet"     # aligned working table (row-order contract)
SAE2_BGE_FILE = "data/sae2/bge.npy"                 # (9500, 1024) sliced BGE matrix
SAE2_QWEN_DIR = "data/sae2/qwen"                    # per-layer Qwen activations: qwen_L{n}.npy
SAE2_VARIANTS_DIR = "data/sae2/variants"            # one subdir per trained SAE variant
SAE2_USERMODELS_DIR = "data/sae2/usermodels"
SAE2_PAIRS_DIR = "data/sae2/pairs"
SAE2_LABELS_FILE = "data/sae2/axis_labels.parquet"  # LLM axis scores on the validation subset
SAE2_REPORT_FILE = "data/sae2/report.md"

# Representation space — Qwen activation extraction
SAE2_QWEN_MODEL = "Qwen/Qwen2.5-7B"   # base; Apache 2.0, no HF gate
SAE2_QWEN_LAYERS = [14, 16, 18, 20, 22, 24, 26]  # candidate layers (of 28) — upper-middle band; swept, then one chosen
SAE2_QWEN_DIM = 3584                  # Qwen2.5-7B hidden_size
SAE2_QWEN_MAX_TOKENS = 128
SAE2_QWEN_BATCH = 4
SAE2_QWEN_DEVICE = "mps"              # falls back to cpu if unavailable
SAE2_QWEN_DTYPE = "bfloat16"
SAE2_BGE_DIM = 1024

# Topic removal
SAE2_KNN_K = 20                       # neighbors averaged to form the residual (fluid topics, no labels)

# SAE training (shared by single-post and difference variants)
SAE2_N_FEATURES = 128
SAE2_L1_COEF = 0.05
SAE2_LR = 1e-3
SAE2_EPOCHS = 300
SAE2_BATCH_SIZE = 64
SAE2_SEED = 42
SAE2_NORMALIZE = True                 # rescale inputs to unit median norm before training

# Validation reference
SAE2_LABEL_SUBSET_N = 2000            # stratified posts annotated on the 9 axes (a reference, not training data)
SAE2_CONFIRM = 0.20                   # max(|r|,|lift|) >= this → axis confidently covered
SAE2_PARTIAL = 0.10                   # in [0.10, 0.20) → partial
SAE2_DEAD_DENSITY = 0.01

# Pairs (see §5 of the handover: natural for training, synthetic for probing)
SAE2_NATURAL_PAIRS_N = 40000          # within-neighborhood differences sampled to TRAIN the difference-SAE
SAE2_SYNTH_PAIRS_PER_AXIS = 400       # LLM single-axis pairs per axis, for naming/grading features
