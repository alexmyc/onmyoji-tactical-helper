# Security Policy & Credential Isolation Governance

This document defines the security architecture, credential-isolation controls, privilege boundaries, and vulnerability-reporting guidelines for **Onmyoji Tactical Helper**, aligned with **ISO/IEC 27001** (Information Security Management Controls) and **OWASP Desktop Application Security** best practices.

---

## 1. Supported Versions

| Release Track | Version | Security & Dependency Updates |
| :--- | :---: | :---: |
| **Gold Master Release** | `v1.0.x` | Supported (`main` branch & `v1.0.0` tag) |
| **Legacy Monolithic Pre-1.0** | `< 1.0.0` | Unsupported (superseded by the 7-layer modular architecture) |

---

## 2. Dual-Track Credential & Data Isolation (`ISO/IEC 27001` A.8.11 / A.8.12)

To allow frictionless technical evaluation without risking accidental exposure of private data sources or endpoint URLs, the system enforces strict **Dual-Track Isolation** across source control, runtime initialization, and binary distribution:

1. **Zero Tracked Secrets in Source Control**:
   - Local `.env` files, neural model weight binaries (`*.pt`, `*.gguf`), runtime logs (`logs/`), and build artifacts (`dist/`, `build/`) are strictly excluded via `.gitignore`.
   - Only the sanitized configuration template ([.env.example](.env.example)), containing empty keys, is tracked in Git.
2. **Credential-Free Default Packaging (`build_release.py`)**:
   - Running `python build_release.py` (default public packaging mode) bundles `.env.example` only and **never** copies a local `.env` into `dist/OnmyojiHelper/data/`.
   - Private cloud configuration is included **only** when the operator explicitly passes `--include-local-env` for a personal local build, enforced by an automated post-build hygiene audit before completion.
3. **Public Sandbox Fallback (`--demo` & Zero-Config Startup)**:
   - When `.env` is absent or `--demo` is supplied, `EnvLoader`, `AuthService`, and `SheetsSyncStore` bypass all external network requests and Windows UAC elevation prompts, operating 100% offline against the built-in sample dataset (`src/onmyoji_helper/domain/public_demo_data.py`).
4. **Local Cache Obfuscation**:
   - When Private Cloud Sync is explicitly configured via an untracked `.env`, synchronized playbook tables are stored under `%LOCALAPPDATA%` using vectorized NumPy XOR obfuscation rather than plaintext temp files in the repository workspace.

---

## 3. Process Privilege & Human-in-the-Loop Safety Controls

1. **Conditional Principle of Least Privilege (Windows UAC / UIPI)**:
   - Offline evaluation (`python run_app.py --demo`), Public Sandbox startups (without `.env`), and automated test suites (`pytest`) run as **standard unprivileged user processes** across Windows and Linux.
   - Windows Administrator elevation (`ShellExecuteW("runas")` / `requireAdministrator` manifest) is requested **exclusively** in live Windows private-track sessions where Windows User Interface Privilege Isolation (UIPI) otherwise blocks `BitBlt` or `SendInput` inspection of an elevated DirectX client.
2. **Non-Intrusive Window Compositing (`WS_EX_NOACTIVATE`)**:
   - Yellow Mode (Primary Showcase) applies `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE` to floating HUD overlays, guaranteeing zero input injection and zero foreground focus theft.
3. **Deterministic Fail-Safe & Circuit-Breaker Guardrails (Red Mode Research)**:
   - **Hardware-Level Emergency Kill-Switch (`< 50 ms`)**: A dedicated daemon thread (`RedModeKillSwitch`) polls `F12` and `ESC` every `50 ms`, immediately halting any pending cursor interpolation and downgrading to passive Yellow Mode.
   - **Bounded Per-Turn Retry Circuit Breaker (`MAX_TURN_RETRIES = 5`)**: If a turn remains unresolved after `5` consecutive retries (~`7–8s` window, e.g., due to a frozen or disconnected client), `AITacticalManager` automatically trips the circuit breaker, halts input dispatch, and downgrades to Yellow Mode.
   - **Foreground Ownership Verification**: Input dispatch executes only when the target `GA_ROOT` window owns foreground focus, pausing automatically whenever the user switches (`Alt+Tab`) to another application.

---

## 4. Responsible Use & Scope Boundary

This project is engineered and published **strictly for educational, portfolio review, and desktop computer-vision research purposes** (exploring real-time edge-AI inference, 2D affine occlusion tracking, and mixed-DPI Win32 window compositing). It does not read or modify process memory, inject DLLs, intercept network packets, or bypass anti-cheat/access controls; all perception operates purely on standard OS desktop frame buffers. Users are solely responsible for adhering to the terms of service of any third-party software.

---

## 5. Reporting a Vulnerability

If you discover a security issue (such as credential leakage in packaging scripts, unsafe file handling, or privilege-boundary defects):

1. **Do not open a public GitHub Issue** containing exploit details or sensitive paths.
2. Report the finding privately via **GitHub Private Vulnerability Reporting** on the [repository Security tab](https://github.com/alexmyc/onmyoji-tactical-helper/security) with reproduction steps and affected commit/tag (`v1.0.0`).
3. Maintainers aim to acknowledge reports within **72 hours** and publish a verified patch + regression test on `main`.
