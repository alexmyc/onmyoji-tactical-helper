# Architecture & Core Algorithms Whitepaper

This document details the layered software architecture, mathematical decision models, spatial tracking formulations, mixed-DPI coordinate projections, concurrency invariants, and native binary hardening pipeline of **Onmyoji Tactical Helper**.

---

## 1. Clean 7-Layer Dependency Architecture & Dual-Track Isolation

The codebase enforces a strict unidirectional dependency graph (`app / ui` $\to$ `decision / vision / automation` $\to$ `domain / infra`). Core symbolic decision engines, finite-state machines, and spatial trackers have zero compile-time or runtime coupling to Win32 window handles, GDI surfaces, or Tkinter widgets—enabling deterministic, sub-second CI verification on headless runners.

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
1. **Zero Import-Time Side Effects**: Process DPI awareness, OpenMP/BLAS thread-pool caps, rotating log handlers, and heavy neural runtimes (`RapidOCR`, `YOLOv8`, `Gemma-4`) are never initialized at module import time; all wiring is deferred to `bootstrap.run_application()` or lazy per-engine initialization.
2. **Dual-Track Configuration Isolation (`infra/env_loader.py`)**:
   - **Private Production Track**: When an untracked `.env` file (`<project_root>/.env` or `<exe_dir>/data/.env`) supplies `ONMYOJI_SHEET_*` and `ONMYOJI_AUTH_CSV_URL`, `SheetsSyncStore` synchronizes live 251-unit metadata and playbooks into an AppData cache encrypted via vectorized NumPy XOR (`np.bitwise_xor`, ~50x faster than Python byte loops), while `AuthService` validates the hardware SHA-256 Machine ID (`wmic csproduct get uuid` with `Cryptography\MachineGuid` fallback for Windows 11 24H2+).
   - **Public Sandbox Track**: When `.env` is absent (default after `git clone`) or `--demo` is passed, the system bypasses UAC elevation and network sync, loading `domain/public_demo_data.py` into `ShikigamiRepository` and `DemoSandboxWindow` with zero errors.
3. **$O(1)$ Amortized Metadata Queries**: `ShikigamiRepository` pre-sorts full-to-short aliases and multi-character tokens once by descending string length at startup and wraps tag lookups in `@lru_cache(maxsize=512)`, reducing dozens of per-frame combat queries to $< 0.02\text{ ms}$.

---

## 2. XMind Counter-Tree & Mirror-Match Mathematical Model (`decision/draft_engine.py`)

Competitive *Onmyoji* drafting playbooks form a causal counter-pick tree partitioned by the opponent's Ban root:
- **Ally Lineup $M = (m_0, m_1, m_2, m_3, m_4)$** is modeled as a **strictly ordered 5-step counter-chain backbone**.
- **Opponent Lineup $O = \{o_1, \dots, o_5\}$** is modeled as an **order-invariant opening formation set**, eliminating combinatorial noise when opponents permute their first three picks.

### 2.1 Global Strict Availability Mask
Let $b_{\text{opp}}$ and $b_{\text{my}}$ denote the banned cards, and let $P_{\text{my}}, P_{\text{opp}}$ denote revealed picks. Every candidate recommendation or prediction $c$ is filtered against the hard exclusion mask $U$:

$$
U = \{b_{\text{opp}}, b_{\text{my}}\} \cup P_{\text{my}} \cup P_{\text{opp}}, \qquad c \notin U
$$

### 2.2 Branch Weighting & Dead-Branch Viability Decay
For any playbook branch $(M, O)$ evaluated at draft step $k = |P_{\text{my}}|$, its composite branch weight $W_{\text{branch}}$ combines ordered ally prefix adherence $S_{\text{backbone}}$, unordered opponent precision/recall $S_{\text{opp}}$, Ban-root affinity $\mu_{\text{ban}}$, and an endgame viability decay factor $\gamma_{\text{viability}}$:

$$
W_{\text{branch}} = \left( S_{\text{backbone}}(M, P_{\text{my}}) + S_{\text{opp}}(O, P_{\text{opp}}) \right) \cdot \mu_{\text{ban}} \cdot \gamma_{\text{viability}}
$$

Whenever unpicked core cards in $R_{\text{my}} = M \setminus P_{\text{my}}$ have been banned ($n_{\text{ban}}$) or stolen by the opponent ($n_{\text{stolen}}$), $\gamma_{\text{viability}}$ applies exponential decay so broken branches immediately yield to intact pivot branches:

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

## 3. 2D Partial Affine Stabilization & Symmetric Occlusion Recovery (`vision/spatial_tracker.py`)

During 5v5 combat, ultimate animations induce camera zoom/pan translation and multi-second particle occlusion across both formations:

1. **Dynamic Baseline Anchoring**: Once 5 units of a camp are localized, `StateParser` locks an initial horizontal baseline $\{p_i^{(0)}\}_{i=1}^5$ (automatically rescaling anchors proportionally if the user resizes the game window mid-match).
2. **2D Partial Affine Camera Solve**: Whenever at least 3 units are visible on frame $t$, `cv2.estimateAffinePartial2D` solves the 4-DoF transformation matrix $\mathbf{H}_t \in \mathbb{R}^{2 \times 3}$ (scale, rotation, translation) mapping baseline anchors to current frame coordinates, projecting occluded unit positions $\hat{p}_k^{(t)}$:

$$
\hat{p}_k^{(t)} = \mathbf{H}_t \begin{bmatrix} p_{k,x}^{(0)} \\ p_{k,y}^{(0)} \\ 1 \end{bmatrix}
$$

3. **Symmetric EMA Trajectories & `3.5s` Grace Recovery**: Both ally and enemy camps maintain exponential moving average positions (`smooth_pos_my`, `smooth_pos_opp`). If a unit vanishes due to ultimate VFX ($\Delta t < 3.5\text{ s}$), its alive state and physical coordinates recover seamlessly via the fallback chain $\hat{p}_k^{(t)} \to \text{EMA}_k \to p_k^{(0)}$, preventing HUD flicker and mistargeted clicks.

---

## 4. Mixed-DPI Physical Projection & Safety-Critical Execution (`automation/` & `ui/`)

```mermaid
stateDiagram-v2
    [*] --> Green: Default Startup (0 MB VRAM)
    Green --> Yellow: Click Status Dot (Load YOLOv8 + Gemma-4 / Rule Tier)
    Yellow --> Red: Click Status Dot (Arm SafetyGuard + Restore GA_ROOT Focus)
    Red --> Yellow: F12 / ESC / Manual Mouse Takeover (< 50 ms Instant Brake, Keep VRAM & HUD)
    Red --> Green: Click Status Dot (Stop Combat Threads & Release CUDA VRAM)
    Yellow --> Green: Close Tactical Panel or Cycle Dot (Release CUDA VRAM)
```

### 4.1 Unified DWM DPI & Letterbox Physical Projection (`CoordinateMapper`)
On displays with Windows DWM scaling (e.g., `125%` or `150%`), GDI `BitBlt` against a non-DPI-aware game surface captures logical buffer dimensions $(W_{\text{cap}}, H_{\text{cap}})$ before `CaptureEngine` strips black letterbox bars $(x_0, y_0)$, whereas `GetClientRect` under our Per-Monitor DPI v2 process returns physical client pixels $(W_{\text{phys}}, H_{\text{phys}})$.

Both `BattleOverlayUI` and `CoordinateMapper` invoke `project_cropped_to_physical_client`, computing exact DWM scale factors from the uncropped capture frame $W_{\text{cap}} = W_{\text{img}} + 2x_0$:

$$
X_{\text{client}} = (x_0 + x_{\text{img}}) \cdot \frac{W_{\text{phys}}}{W_{\text{img}} + 2x_0}, \qquad
Y_{\text{client}} = (y_0 + y_{\text{img}}) \cdot \frac{H_{\text{phys}}}{H_{\text{img}} + 2y_0}
$$

Screen coordinates are translated via `ClientToScreen` and clamped strictly within the intersection of the client rect and `GetAncestor(hwnd, GA_ROOT)` root window bounds—guaranteeing 1:1 alignment across `16:9`, `16:10` (`1920×1200`), and negative-coordinate multi-monitor setups.

### 4.2 Unified 2-Step Casting & Orb-Shortage Downgrade
- **Unified 2-Step Targeting**: In *Onmyoji*, single-target skills, enemy AoE (`敵方全體`), and ally AoE (`我方全體`) share a consistent interaction contract: `AutoBattleExecutor` always resolves a valid alive unit on the target side for **Step 1 (Unit Lock Click)** followed by **Step 2 (Skill Button Click)**, emitting a zero-delta `MOUSEEVENTF_MOVE` hover pulse (`35–50 ms`) prior to `SendInput` mouse-down/up so DirectX/Messiah engines register the hover hitbox reliably.
- **Automatic `S1` Fallback**: If a recommended skill (`S2/S3/S4`) is unavailable due to insufficient orbs or cooldown (`skill_states[sk] == False`), the executor automatically downgrades to `[S1]` normal attack and re-targets the highest-priority alive enemy unit so the turn never stalls.
- **Dual Anti-Double-Click Guards**: Every cast engages both a `>= 2.2s` cooldown lock and a `True -> False -> True` turn-edge latch (with a `5.5s` consecutive extra-turn escape hatch for pull/extra-turn mechanics).

---

## 5. Threading Model, Lock Hierarchy & Resource Symbiosis

| Thread Name | Cadence / Trigger | Synchronization Boundary & Responsibility |
| :--- | :--- | :--- |
| **`MainThread` (Tkinter)** | `50 ms` UI queue drain + `.after(0)` | Sole owner of Tkinter widgets and Canvas primitives; floating panels apply 64-bit safe `WS_EX_NOACTIVATE` post-`deiconify()`. |
| **`ocr_worker`** | `0.9s / 1.3s / 2.5s / 4.5s` | Executes GDI draft capture, motion-diff gating, `RapidOCR`, and `DraftStateMachine`; publishes immutable tuples over `queue.Queue`. |
| **`BattleCoordinatorLoop`** | `0.25s` (ally) / `0.60s` (enemy) | Acquires `yolo_lock` for YOLOv8 + affine tracking; guards shared slot dictionaries via re-entrant `_state_lock (RLock)`. |
| **`AsyncGemmaWorker`** | On turn edge (`>= 3.0s` cooldown) | Acquires independent `llm_lock` for Gemma-4 GGUF completion so LLM prefill never blocks `0.25s` target tracking or auto-clicks. |
| **`RedModeKillSwitch`** | `50 ms` `GetAsyncKeyState` poll | Monitors `F12` / `ESC`; increments `ModeController._transition_epoch` and atomically disarms `auto_battle_enabled` in `< 50 ms`. |

---

## 6. Hardened Native Binary Release Pipeline (`build_release.py`)

1. **Sequential Low-RAM Cython 3.3 Compilation**: Discovers all **33 modules** under `src/onmyoji_helper/` and compiles each inside an isolated child process (`--workers 1` default) with `docstrings=False`, `binding=False`, `c_line_in_traceback=False`, `emit_code_comments=False`, MSVC `/O2 /DNDEBUG`, and `/DEBUG:NONE` symbol stripping.
2. **PyInstaller Minimalist Freeze**: Packages `OnmyojiHelper.exe` with `--onedir --windowed --uac-admin --contents-directory=data`, embedding the Windows `requireAdministrator` PE `RT_MANIFEST` and copying `best.pt` (plus local `.env` into `data/.env` for private builds) directly without archive re-compression.
3. **Automated Post-Build Leak Audit**: Verifies root directory cleanliness (`['OnmyojiHelper.exe', 'data']`), `requireAdministrator` manifest presence, `33/33` `.pyd` binary completeness, and **zero `.py` source files** across the output tree.
