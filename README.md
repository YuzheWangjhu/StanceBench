# StanceBench

StanceBench is a benchmark and evaluation pipeline for measuring interpersonal stance from speech with audio-language models. It builds evaluation clips from the Seamless Interaction dataset, prompts model judges with stance-specific forced-choice questions, and writes per-speaker stance scores and evidence to CSV.

The benchmark covers nine interpersonal stance dimensions:

- `S0` Interpersonal Warmth
- `S1` Compassion and Empathy
- `S2` Politeness and Respect
- `S3` Assertiveness
- `S4` Sincerity and Honesty
- `S5` Cognitive Attentiveness
- `S6` Social Engagement
- `S7` Power Orientation
- `S8` Conflict Regulation

## Repository Layout

```text
stancebench/
  metadata/
    questions_main.json            # stance question definitions
    category_roles.csv             # role/category mapping
    interactions_role_ABmapped.csv # StanceBench conversation metadata
  scripts/
    filter_roles.py                # select benchmark rows by stance roles
    select_one_question.py         # extract one stance question config
    turnover_extractor_IPU.py      # extract speaker IPUs from dyad audio
    build_eval_inputs.py           # build single/interaction eval clips
  models/
    qwen_omni/                     # Qwen2.5-Omni audio judge
    kimi_audio/                    # Kimi-Audio judge
    granite_speech/                # Granite Speech transcript baseline
    gpt_audio/                     # GPT audio judge
    gemini_audio/                  # Gemini audio judge
    qwen_transcript_ablation/      # Qwen transcript-only ablation
examples/notebooks/
  analyze_all_paper.ipynb          # analysis notebook for generated results
```

Top-level `scripts/`, `models/`, and `metadata/` paths are compatibility
entrypoints for older commands. New usage should prefer the `stancebench` CLI.

## Data

StanceBench uses the improvised subset of the Meta Seamless Interaction dataset. Download the dataset separately from the official sources:

- https://huggingface.co/datasets/facebook/seamless-interaction
- https://github.com/facebookresearch/seamless_interaction

The dataset is distributed by Meta under CC-BY-NC 4.0. This repository includes StanceBench metadata, but does not redistribute Seamless Interaction audio.

Set the dataset paths before running evaluation:

```bash
export SEAMLESS_DATASET_ROOT=/path/to/seamless_interaction/datasets/improvised
export SEAMLESS_DYAD_LOOKUP_CSV=/path/to/seamless_interaction/datasets/assets/dyad_lookup.csv
```

By default, the pipeline uses:

```bash
stancebench/metadata/interactions_role_ABmapped.csv
```

To use a different StanceBench metadata file:

```bash
export STANCEBENCH_INTERACTIONS_ROLE_ABMAPPED_CSV=/path/to/interactions_role_ABmapped.csv
```

## Setup

For model evaluation, create the full conda environment:

```bash
micromamba env create -f environment.yml
micromamba activate stancebench
```

`environment.yml` targets GPU evaluation with PyTorch 2.6 and CUDA 12.4 through conda's `pytorch-cuda` package. It also includes Transformers, Qwen/Granite dependencies, Kimi-Audio, and API client libraries.

For lightweight metadata checks or notebook-only work, `requirements.txt` provides a smaller pip dependency list:

```bash
pip install -r requirements.txt
```

For editable CLI use:

```bash
pip install -e .
stancebench --help
```

Model-specific requirements:

- Qwen: access to `Qwen/Qwen2.5-Omni-7B`
- Kimi: Kimi-Audio inference package and access to `moonshotai/Kimi-Audio-7B-Instruct`
- Granite: access to `ibm-granite/granite-speech-3.3-8b`
- GPT audio: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY` or `GEMINI_API_KEY`

The project license is pending. No `LICENSE` file is included yet.

## Validate Data

Check bundled StanceBench metadata:

```bash
stancebench validate-data
```

Check a local Seamless Interaction layout:

```bash
stancebench validate-data \
  --dataset-root "$SEAMLESS_DATASET_ROOT" \
  --dyad-lookup-csv "$SEAMLESS_DYAD_LOOKUP_CSV"
```

## Run An Evaluation

The CLI is dimension-driven: pass a paper dimension such as `S0`, and StanceBench
selects the matching question, roles, and input mode.

Create a question config for `S0`:

```bash
stancebench select-question \
  --dimension S0 \
  --output question_0.json
```

Run Qwen2.5-Omni on `S0`:

```bash
stancebench run \
  --model qwen-omni \
  --dimension S0 \
  --output-root runs \
  --model-id Qwen/Qwen2.5-Omni-7B
```

Supported model names:

- `qwen-omni`
- `kimi-audio`
- `granite-speech`
- `gpt-audio`
- `gemini-audio`
- `qwen-transcript-ablation`

Use `single` mode for `S0`-`S5` and `interaction` mode for `S6`-`S8`; the CLI
chooses this automatically from the dimension.

You can also run from a YAML config:

```yaml
model: qwen-omni
dimension: S0
output_root: runs
selection_portion: 1.0
seed: 666
dataset_root: /path/to/seamless_interaction/datasets/improvised
dyad_lookup_csv: /path/to/seamless_interaction/datasets/assets/dyad_lookup.csv
model_id: Qwen/Qwen2.5-Omni-7B
```

```bash
stancebench run --config config.yaml
```

## Stance Dimensions

| ID | Input mode | Positive pole | Negative pole |
| --- | --- | --- | --- |
| S0 | single | Warmth | Coldness |
| S1 | single | Compassion | Callousness |
| S2 | single | Politeness | Rudeness |
| S3 | single | Assertiveness | Inhibition |
| S4 | single | Honesty | Deception |
| S5 | single | Focus | Distraction |
| S6 | interaction | Sociability | Withdrawal |
| S7 | interaction | Deference | Dominance |
| S8 | interaction | Calmness/Avoidance | Aggression |

Single-speaker clips use a target active-speech threshold of 30 seconds and a maximum of 45 seconds. Interaction clips evaluate a target response with nearby partner context.

## Outputs

CLI runs write structured output under:

```text
runs/<model>/<dimension>/<run_id>/
  config.resolved.yaml
  question.json
  legacy_filtered_subset.csv
  filtered_rows.csv
  scores.csv
  metrics.json
  logs/run.log
```

`legacy_filtered_subset.csv` preserves the model-runner CSV shape for
reproducibility checks. `filtered_rows.csv` and `scores.csv` are copies of that
legacy-compatible CSV.

Main output columns are:

- `evidence_a`, `avg_score_a`, `flip_rate_a`, `fail_note_a`
- `evidence_b`, `avg_score_b`, `flip_rate_b`, `fail_note_b`

Scores are derived from two balanced prompt-order variants. Each item stores the model choice, probability, mapped score, and short evidence list.

## Compatibility Entrypoints

Model-specific script entrypoints remain available for existing workflows:

```bash
python stancebench/models/qwen_omni/run_turnover_qwen_QA.py ...
python models/qwen_omni/run_turnover_qwen_QA.py ...
```

These wrappers are retained for compatibility. The recommended public interface
is `stancebench run`.

## Analysis

Use `examples/notebooks/analyze_all_paper.ipynb` after generating local model outputs. The notebook reads the CLI layout under `runs/<model>/<dimension>/<run_id>/legacy_filtered_subset.csv` and also supports the compatibility `runs_bpc_evidence/q*/filtered_subset_*.csv` layout. It computes benchmark summaries such as failure rate, item disagreement, pole consistency, EER, AUROC, oracle F1, and human-correlation analyses when human-evaluation outputs are available locally.

To confirm that generated CLI outputs are visible to the analysis workflow:

```bash
stancebench analyze --runs runs
```
