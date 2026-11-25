# System Architecture Guide

This guide documents the production-oriented 7-layer modular architecture, rule-based decision-tree formulations, verifiable `YOLOv8n` training & spatial tracking pipeline, mixed-DPI coordinate projections, concurrency invariants, and native `.pyd` packaging workflow of **Onmyoji Tactical Helper**.

---

## 1. Production-Oriented 7-Layer Modular Architecture

The codebase enforces a strict unidirectional dependency graph (`app / ui` $\to$ `decision / vision / automation` $\to$ `domain / infra`). Core symbolic decision engines, finite-state machines, and spatial trackers remain decoupled from Win32 window handles, GDI surfaces, and Tkinter widgets—enabling deterministic, sub-second CI verification on headless runners.

```mermaid
flowchart LR
    App["app/\nbootstrap · mode_controller · battle_coordinator"]
    UI["ui/\nmain_window · tactical_panel · hud_overlay · demo_sandbox"]
    Dec["decision/\ndraft_fsm · draft_engine · target_selector · llm_engine"]
    Vis["vision/\nspatial_tracker · turn_detector · action_bar · draft_ocr_scanner"]
    Auto["automation/\ncoordinate_mapper · safety_guard · battle_executor"]
    Dom["domain/\nmodels · public_demo_data · shikigami_repository"]
    Inf["infra/\nenv_loader · win32_* · capture_engine · sheets_sync_store · model_lifecycle"]

    App --> UI & Dec & Vis & Auto & Dom & Inf
    UI --> Dec & Dom & Inf
    Dec --> Dom & Inf
    Vis --> Dom
    Auto --> Dom & Inf
```

### Key Architectural Invariants
1. **Zero Import-Time Side Effects**: Process DPI awareness, OpenMP/BLAS thread-pool caps, rotating log handlers, and neural runtimes (`RapidOCR`, `YOLOv8n`, `Gemma-4`) are never initialized at module import time; all runtime wiring is deferred to `bootstrap.run_application()` or lazy per-engine initialization.
2. **Dual-Track Configuration Isolation (`infra/env_loader.py`)**:
   - **Private Cloud Track**: When an untracked `.env` file (`<project_root>/.env` or `<exe_dir>/data/.env`) defines `ONMYOJI_SHEET_*` and `ONMYOJI_AUTH_CSV_URL`, `SheetsSyncStore` synchronizes the full **1,128-entry alias dictionary and 251 shikigami combat tag profiles** from external Google Sheets into a local AppData cache obfuscated via vectorized NumPy XOR (`np.bitwise_xor`, ~50x faster than Python byte loops), while `AuthService` validates the hardware SHA-256 Machine ID (`wmic csproduct get uuid` with `Cryptography\MachineGuid` fallback for Windows 11 24H2+).
   - **Public Sandbox Track**: When `.env` is absent (default after `git clone`) or `--demo` is passed, the application skips UAC elevation and network sync, loading the self-contained sandbox dataset (`domain/public_demo_data.py`: **~25 sample aliases, 13 unit combat profiles, and 2 playbooks**) into `ShikigamiRepository` and `DemoSandboxWindow`.
3. **Bounded Alias & Tag Resolution (`domain/shikigami_repository.py`)**: Resolves exact dictionary keys first, exact `full_to_short_map` aliases second, and length-descending root containment (`len >= 2`, overlap ratio $\ge 0.33$, excluding generic rarity prefixes such as `SP`/`SSR`) third—wrapped in `@lru_cache(maxsize=512)` (`P50 < 0.001 ms`) to prevent short-token false positives.

---

## 2. Rule-Based, Dependency-Aware XMind Decision Tree (`decision/draft_engine.py`)

Competitive PvP drafting playbooks are structured as a rule-based, dependency-aware counter-pick tree partitioned by the opponent's Ban root:
- **Ally Lineup $M = (m_0, m_1, m_2, m_3, m_4)$** is modeled as a **strictly ordered 5-step counter-chain backbone**.
- **Opponent Lineup $O = \{o_1, \dots, o_5\}$** is modeled as an **order-invariant opening formation set**, eliminating combinatorial noise when opponents permute their first three picks.

### 2.1 Global Strict Availability Mask
Let $b_{\text{opp}}$ and $b_{\text{my}}$ denote banned cards, and let $P_{\text{my}}, P_{\text{opp}}$ denote revealed picks. Every candidate recommendation or prediction $c$ is filtered against the hard exclusion mask $U$:

$$
U = \{b_{\text{opp}}, b_{\text{my}}\} \cup P_{\text{my}} \cup P_{\text{opp}}, \qquad c \notin U
$$

### 2.2 Branch Weighting & Dead-Branch Viability Decay
For any playbook branch $(M, O)$ evaluated at draft step $k = |P_{\text{my}}|$, its composite branch weight $W_{\text{branch}}$ combines ordered ally prefix adherence $S_{\text{backbone}}$, unordered opponent precision/recall $S_{\text{opp}}$, Ban-root affinity $\mu_{\text{ban}}$, and an endgame viability decay factor $\gamma_{\text{viability}}$:

$$
W_{\text{branch}} = \left( S_{\text{backbone}}(M, P_{\text{my}}) + S_{\text{opp}}(O, P_{\text{opp}}) \right) \cdot \mu_{\text{ban}} \cdot \gamma_{\text{viability}}
$$

Whenever unpicked core cards in $R_{\text{my}} = M \setminus P_{\text{my}}$ have been banned ($n_{\text{ban}}$) or stolen by the opponent ($n_{\text{stolen}}$), $\gamma_{\text{viability}}$ applies exponential decay so broken branches yield to intact pivot branches:

$$
\gamma_{\text{viability}} = (0.12)^{n_{\text{ban}}} \cdot (0.18)^{n_{\text{stolen}}} \cdot \eta_{\text{next}} \cdot \eta_{\text{opp\_ban}}
$$

where $\eta_{\text{next}} = 0.35$ if the immediate next slot $m_k \in U$ (else $1.0$), and $\eta_{\text{opp\_ban}} = 0.55$ if $O \cap \{b_{\text{opp}}, b_{\text{my}}\} \neq \emptyset$ (else $1.0$).

### 2.3 Top-Tier Quadratic Sharpening & Mirror Detection
Branches within $50\%$ of the dominant weight $W_{\max}$ are aggregated via quadratic sharpening:

$$
S(c) = \sum_{i \in \mathcal{T}_c} W_i \cdot \left(\frac{W_i}{W_{\max}}\right)^2
$$

Simultaneously, a parallel **Mirror / Copycat Detector** evaluates $P_{\text{opp}} \cap \text{set}(M)$ across our own recorded ally backbones (boosted by $+35.0$ per Turn-1/2 opener match). When mirror weight dominates or reaches $\ge 85\%$ of standard opponent-column weight with $\ge 2$ hits, the engine emits a `[鏡像/內戰]` alert and predicts the opponent's remaining cards directly from our own $M$ chain.

---

## 3. Verifiable `YOLOv8n` Detector, 2D Affine Stabilization & Geometric Verification (`vision/`)

### 3.1 `YOLOv8n` Dataset Split, Hyperparameters & Validation Evidence
All training records and evaluation curves are archived in [`docs/model_artifacts/`](model_artifacts/) ([args.yaml](model_artifacts/args.yaml), [results.csv](model_artifacts/results.csv), [labels.jpg](model_artifacts/labels.jpg), [BoxPR_curve.png](model_artifacts/BoxPR_curve.png), [BoxF1_curve.png](model_artifacts/BoxF1_curve.png)). Binary `.pt` weights and raw game frame datasets remain excluded from Git via `.gitignore`.

- **Dataset & Train/Val Partition**: Because *Onmyoji* 5v5 PvP takes place on a fixed-perspective isometric battle stage, transfer learning from `yolov8n.pt` (`5.9 MB`, `imgsz=640`, `batch=4`, `150 epochs`, `110.1s` CUDA wall time) was trained on **16 annotated 5v5 combat scenes (`121` unit bounding boxes: `62 enemy`, `59 ally`)** and evaluated against a held-out **validation split (`20` unit bounding boxes: `10 enemy`, `10 ally`)**.
- **Augmentation Pipeline**: `auto_augment=randaugment`, `mosaic=1.0` (`close_mosaic=10`), `erasing=0.4`, HSV color perturbation (`hsv_h=0.015`, `hsv_s=0.70`, `hsv_v=0.40` for intense ultimate lighting), geometric translation/scaling (`translate=0.10`, `scale=0.20`), and horizontal flip (`fliplr=0.50`).
- **Best Checkpoint Metrics (Epoch 98 in [results.csv](model_artifacts/results.csv))**:
  - **`mAP@50`**: `99.5%` (`0.9950` — `enemy: 0.995`, `ally: 0.995`)
  - **`Precision`**: `89.0%` (`0.88982`)
  - **`Recall`**: `93.5%` (`0.93525`)
  - **`mAP@50-95`**: `60.1%` (`0.60071`)

![YOLOv8n Validation Confusion Matrix](model_artifacts/confusion_matrix.png)

### 3.2 2D Partial Affine Stabilization & `3.5s` Occlusion Continuity (`StateParser`)
During 5v5 combat, ultimate animations trigger camera zoom/pan motion, `±4 px` bounding-box jitter, and multi-second particle blackouts:

1. **Dynamic Baseline Anchoring**: Once 5 units of a camp are localized, `StateParser` locks an initial horizontal baseline $\{p_i^{(0)}\}_{i=1}^5$ (automatically rescaling anchors proportionally if the window is resized mid-match).
2. **2D Partial Affine Camera Solve**: Whenever at least 3 units are visible on frame $t$, `cv2.estimateAffinePartial2D` solves the 4-DoF transformation matrix $\mathbf{H}_t \in \mathbb{R}^{2 \times 3}$ (scale, rotation, translation) mapping baseline anchors to current frame coordinates, projecting occluded unit positions $\hat{p}_k^{(t)}$:

$$
\hat{p}_k^{(t)} = \mathbf{H}_t \begin{bmatrix} p_{k,x}^{(0)} \\ p_{k,y}^{(0)} \\ 1 \end{bmatrix}
$$

3. **Symmetric EMA Trajectories & `3.5s` Grace Recovery**: Both ally and enemy camps maintain exponential moving average positions (`smooth_pos_my`, `smooth_pos_opp`), suppressing frame-to-frame detection jitter by `> 35%`. If a unit vanishes due to ultimate VFX ($\Delta t < 3.5\text{ s}$), its alive state and coordinates recover via $\hat{p}_k^{(t)} \to \text{EMA}_k \to p_k^{(0)}$, maintaining `100%` slot continuity (`< 10 px` mean error under `±4 px` noise + random dropouts in `tests/test_vision_synthetic_regression.py`) before transitioning genuinely defeated units ($\Delta t \ge 3.5\text{ s}$) to `alive = False`.

### 3.3 Circumferential Sobel Turn Verifier & Whitelist-Tiered `RapidOCR`
- **`TurnDetector`**: Combines `CLAHE` (`clipLimit=2.5`), `HoughCircles`, and a **16-point circumferential Sobel gradient check** (`>= 7/16` perimeter samples with $\|\nabla I\| \ge 32.0$ plus pairwise radius/vertical colinearity `r_diff <= 12%`, `y_diff <= 8%`), rejecting full-screen flashbang washouts while distinguishing 3- vs. 4-skill layouts and HSV availability (`V_mean >= 75`, `V_90 >= 110`, `S_mean >= 30 | S_85 >= 45`).
- **`DraftOcrScanner`**: Resizes the opponent ROI (`1.30x` linear upscale, or normalized to `640 px` width on narrow windows), horizontally stitches the aligned ally column (`np.hstack`) for a single `RapidOCR` pass, and applies **two-tier whitelist confidence filtering**: general text boxes require `min_confidence = 0.65`, whereas recognized glyphs matching the known single-character / short-token shikigami whitelist (`季`, `瀧`, `離`, `神`, `連`, `石`, etc.) accept `known_token_min_confidence = 0.35` so isolated single CJK characters are never dropped.

---

## 4. Mixed-DPI Coordinate Projection & Deterministic Fail-Safe Guardrails (`automation/` & `ui/`)

```mermaid
stateDiagram-v2
    [*] --> Green: Default Startup (0 MB VRAM)
    Green --> Yellow: Click Status Dot (Load YOLOv8n + Gemma-4 / Rule Tier)
    Yellow --> Red: Click Status Dot (Arm SafetyGuard + Restore GA_ROOT Focus)
    Red --> Yellow: F12/ESC (50 ms Poll) or >95 px Mouse Step Delta (Keep VRAM & HUD)
    Red --> Green: Click Status Dot (Stop Combat Threads & Release CUDA VRAM)
    Yellow --> Green: Close Tactical Panel or Cycle Dot (Release CUDA VRAM)
```

### 4.1 Unified DWM DPI & Letterbox Physical Projection (`CoordinateMapper`)
On displays with Windows DWM scaling (e.g., `125%` or `150%`), GDI `BitBlt` against a non-DPI-aware application surface captures logical buffer dimensions $(W_{\text{cap}}, H_{\text{cap}})$ before `CaptureEngine` strips black letterbox bars $(x_0, y_0)$, whereas `GetClientRect` under our Per-Monitor DPI v2 process returns physical client pixels $(W_{\text{phys}}, H_{\text{phys}})$.

Both `BattleOverlayUI` and `CoordinateMapper` invoke `project_cropped_to_physical_client`, computing exact DWM scale factors from the uncropped capture frame $W_{\text{cap}} = W_{\text{img}} + 2x_0$:

$$
X_{\text{client}} = (x_0 + x_{\text{img}}) \cdot \frac{W_{\text{phys}}}{W_{\text{img}} + 2x_0}, \qquad
Y_{\text{client}} = (y_0 + y_{\text{img}}) \cdot \frac{H_{\text{phys}}}{H_{\text{img}} + 2y_0}
$$

Screen coordinates are translated via `ClientToScreen` and clamped within the intersection of the client rect and `GetAncestor(hwnd, GA_ROOT)` root window bounds—preserving 1:1 alignment across `16:9`, `16:10` (`1920×1200`), and negative-coordinate multi-monitor setups.

### 4.2 Smooth Cursor Interpolation, UI Hover Stabilization & Human-in-the-Loop Override Split
- **Smooth Cursor Interpolation & UI Hover Stabilization**: DirectX/Messiah UI frameworks update hover hitboxes asynchronously on mouse-move messages. `win32_input.perform_humanized_click` interpolates cursor motion across `4–6` smoothstep slices (`SetCursorPos`) and emits a zero-delta `MOUSEEVENTF_MOVE` hover stabilization pulse (`35–50 ms`) prior to `SendInput` button dispatch so UI hover states register reliably.
- **Unified 2-Step Dispatch & `S1` Fallback**: Single-target skills, enemy AoE (`敵方全體`), and ally AoE (`我方全體`) follow a unified 2-step sequence (**Step 1: Alive Unit Lock** $\to$ **Step 2: Skill Button Dispatch**). When `S2/S3/S4` are unavailable due to insufficient orbs or cooldown, `AutoBattleExecutor` automatically falls back to `[S1]` normal attack on the highest-priority alive enemy unit.
- **Two-Layer Human-in-the-Loop Override Architecture**:
  1. **Background Emergency Hotkey Thread (`SafetyGuard`)**: `RedModeKillSwitch` polls `F12` and `ESC` via `GetAsyncKeyState` every **`50 ms`**, setting `stop_event`, incrementing `ModeController._transition_epoch`, and immediately downgrading to Yellow Mode.
  2. **In-Flight Cursor Takeover Check (`Win32Input`)**: Between consecutive smoothstep interpolation slices inside `perform_humanized_click`, `get_cursor_pos()` compares actual physical cursor position against the previous slice position; if unexpected user mouse movement exceeds **`> 95 px`** (`mouse_takeover_px`), the click aborts before `SendInput` and triggers the same Yellow Mode downgrade callback.
  3. **Single-Action Turn Latch**: Every completed dispatch engages both a `>= 2.2s` cooldown timer and a `True -> False -> True` turn-edge latch (`5.5s` extra-turn escape hatch).

---

## 5. Threading Model, Lock Hierarchy & Resource Symbiosis

| Thread Name | Cadence / Trigger | Synchronization Boundary & Responsibility |
| :--- | :--- | :--- |
| **`MainThread` (Tkinter)** | `50 ms` UI queue drain + `.after(0)` | Sole owner of Tkinter widgets and Canvas primitives; floating panels apply 64-bit safe `WS_EX_NOACTIVATE` post-`deiconify()`. |
| **`ocr_worker`** | `0.9s / 1.3s / 2.5s / 4.5s` | Executes GDI draft capture, motion-diff gating, `RapidOCR` (`0.65 / 0.35` whitelist gate), and `DraftStateMachine`; publishes tuples via `queue.Queue`. |
| **`BattleCoordinatorLoop`** | `0.25s` (ally) / `0.60s` (enemy) | Acquires `yolo_lock` for `YOLOv8n` + affine tracking; guards shared slot dictionaries via re-entrant `_state_lock (RLock)`. |
| **`AsyncGemmaWorker`** | On turn edge (`>= 3.0s` cooldown) | Acquires independent `llm_lock` for Gemma-4 GGUF completion so LLM prefill never blocks `0.25s` HUD tracking. |
| **`RedModeKillSwitch`** | `50 ms` `GetAsyncKeyState` poll | Monitors `F12` / `ESC` override hotkeys; increments `ModeController._transition_epoch` and atomically disarms `auto_battle_enabled`. |

---

## 6. Native Binary Compilation — Source Obfuscation & Packaging (`build_release.py`)

1. **Sequential Low-RAM Cython 3.3 Compilation**: Discovers all **33 modules** under `src/onmyoji_helper/` and compiles each inside an isolated child process (`--workers 1` default) into native Windows `.pyd` C-extensions with `docstrings=False`, `binding=False`, `c_line_in_traceback=False`, `emit_code_comments=False`, MSVC `/O2 /DNDEBUG`, and `/DEBUG:NONE` symbol stripping for source obfuscation and distribution hygiene.
2. **PyInstaller Standalone Packaging**: Packages `OnmyojiHelper.exe` with `--onedir --windowed --uac-admin --contents-directory=data`, embedding the Windows `requireAdministrator` PE `RT_MANIFEST` and copying `best.pt` (plus local `.env` into `data/.env` when present) directly without archive re-compression.
3. **Automated Post-Build Hygiene Audit**: Verifies root directory cleanliness (`['OnmyojiHelper.exe', 'data']`), `requireAdministrator` manifest presence, `33/33` `.pyd` module completeness, and **zero `.py` source files** in the release directory.
