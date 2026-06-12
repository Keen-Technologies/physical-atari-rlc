# Agent scripts

Run commands from the `agent_code/` directory.

Install dependencies with `pip install -r requirements.txt`. **ale-py 0.12+** is required (older versions return action IDs differently and break `sim_env`).

Training runs save checkpoints to local directories (see `checkpoint_dir` in experiment configs).

## Robotroller config

Physical runs need a machine-specific robotroller config (camera and servo settings). Config files live under `agent_code/physical_atari_configs/`:

| File | Purpose |
|------|---------|
| `physical_atari_configs/robotroller.default.json` | Committed template with default values |
| `physical_atari_configs/physical_atari_sample_config.json` | Local config used at runtime (gitignored) |

Copy the template to create your local config, then edit camera and robot settings for your machine:

```bash
cp physical_atari_configs/robotroller.default.json physical_atari_configs/physical_atari_sample_config.json
```

Edit `physical_atari_sample_config.json` (e.g. `camera.device`, `robot.serial_port`, servo positions).

Physical experiment configs (e.g. `experiment_configs/agent_action_input_real.json`) specify the config path:

```json
"robotroller_config_path": ["./physical_atari_configs/physical_atari_sample_config.json"]
```

The resolved path is stored in checkpoint `results.json` so `evaluate_policy.py` and `finetune_policy.py` can reuse it when run with `--env real`. Override with `--robotroller-config` if needed.

If you have an existing home-directory config, copy it once:

```bash
cp ~/robotroller.conf physical_atari_configs/physical_atari_sample_config.json
```

## agent_random

Smoke-test the sim or physical stack with random actions (replaces the old `test_python.py` hardware test). The agent picks a new uniform random action every frame; no training or weight checkpoints are saved (`store_weights: false` in the random experiment configs).

For physical runs, set up your robotroller config first (see **Robotroller config** above).

Sim smoke test:

```bash
python learn_policy.py --config experiment_configs/agent_random_sim.json --run 0
```

Physical smoke test (launch the game on the devbox first, e.g. `PhysicalALE ./games/ pong`):

```bash
python learn_policy.py --config experiment_configs/agent_random_real.json --run 0
```

## learn_policy.py

Trains or benchmarks an agent based on a configuration file.

### Usage

```bash
python learn_policy.py --config experiment_configs/agent_action_input_sim.json --run 0
```

Checkpoints are written to `{checkpoint_dir}/{run_name}/` with `results.json`, `weights.pkl`, and observation state files.

### Arguments

*   `--config`: Path to the JSON configuration file containing hyperparameters.
*   `--run`: Integer index to select a specific configuration from the cross-product of parameters defined in the JSON file.
*   `--gpu`: (Optional) GPU device ID to use.

### Physical environment

Launch the game on the devbox before training:

```bash
# On devbox
PhysicalALE ./games/ pong

# On agent machine
python learn_policy.py --config experiment_configs/agent_action_input_real.json --run 0
```

---

## evaluate_policy.py

Loads a trained model from a checkpoint directory and evaluates its performance.

### Usage

```bash
python evaluate_policy.py --checkpoint-dir ./checkpoints/pong__agents.agent_action_input__0__20260612-120000
```

### Arguments

*   `--checkpoint-dir`: Path to a training run directory (contains `results.json` and `weights.pkl`).
*   `--eval_steps`: Number of steps to run the evaluation (default: 20000).
*   `--device`: Device to run evaluation on (e.g., `cpu`, `cuda`, `mps`).
*   `--env`: Environment type (`sim` or `real`; default: `sim`).
*   `--robotroller-config`: Path to robotroller config when `--env real` (overrides checkpoint).
*   `--comment`: Optional comment logged with evaluation results.

Eval results are saved to `{checkpoint-dir}/eval_results.json`.

---

## finetune_policy.py

Loads a trained model, evaluates it (pre-finetune), fine-tunes it, and evaluates again (post-finetune).

### Usage

```bash
python finetune_policy.py --checkpoint-dir ./checkpoints/pong__...__0__<timestamp> --learning_steps 100000
```

### Arguments

*   `--checkpoint-dir`: Path to source checkpoint run directory.
*   `--eval_before_steps`: Evaluation steps before fine-tuning (default: 18000).
*   `--learning_steps`: Number of fine-tuning steps (required).
*   `--eval_after_steps`: Evaluation steps after fine-tuning (default: 18000).
*   `--device`: Device to run on (e.g., `cpu`, `cuda`, `mps`).
*   `--env`: Environment type (`sim` or `real`; default: `sim`).
*   `--robotroller-config`: Path to robotroller config when `--env real` (overrides checkpoint).
*   `--comment`: Optional comment logged with finetune results.

Fine-tuned weights are saved to a new run directory under `checkpoint_dir`. Summary metrics are written to `{checkpoint-dir}/finetune_results.json`.
