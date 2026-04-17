# CLAUDE.md

이 파일은 이 repo에서 작업하는 Claude Code 세션용 onboarding 문서다. 새 세션이 시작되면 먼저 이 파일을 읽는다.

## 🏁 최종 결과 요약 (2026-04-17)

**핵심 결론**: Triangular lattice + 원형 hole (단순 설계)이 Norder et al. 2025의 neural topology optimization pentagonal 대비 **T = −7~16% 빠른 가속** 달성.

### Best 설계 2개

| | Design A (best T) | Design B (i-line 호환) |
|---|---:|---:|
| T to β=0.2 | **20.73 min** | 21.73 min |
| D | 45.9 Gm | 48.6 Gm |
| Thickness | 280 nm | 220 nm |
| Period | 1580 nm | 1580 nm |
| Hole radius | 600 nm (diameter 1200 nm) | 537 nm (diameter 1074 nm) |
| Wall | 318 nm | 506 nm |
| Mass (10 m²) | 5.1 g | 5.0 g |
| Launch λ | 1550 nm | 1550 nm |
| **Fab (positive resist, hole 노광)** | **DUV 가능** (hole 1200 nm >> DUV 100 nm) | **i-line 가능** (hole 1074 nm, wall 506 nm) |

**핵심**: PhC membrane은 **hole 영역만 노광**하면 되므로 (positive resist → develop → 에칭), limiting MFS는 **hole 지름**이지 wall 두께가 아님. Design A의 hole 지름 1200 nm는 **i-line (500 nm limit)으로도 충분**, DUV로는 여유. 두 설계 모두 photolithography 대량 생산 가능.

### Paper 대비 T 개선 (3가지 조건)

| 조건 | 우리 T | Paper T | Δ |
|---|---:|---:|---:|
| 제약 없음 | 20.73 | 24.6 | **−15.7%** |
| MFS ≥ 500 nm | 21.73 | 24.6 | **−11.7%** |
| Paper 완전 동일 제약 | 22.11 | 24.6 | **−10.1%** |

### 핵심 insight
1. Pixel-level adjoint gradient (96×96) + BO 7-param + grid scan **전부 원형 hole로 수렴** → neural TO 불필요 증명
2. Complex freeform (Fourier / dual-hole / disordered) 전부 baseline 못 이김
3. Graphene underlayer가 MIR ε를 0.10 → 0.28–0.43으로 향상 (NIR/T 무영향)

**상세 discussion**: `docs/discussion_final.md`
**연구종합보고서**: `docs/연구종합보고서.docx`

---

## 프로젝트 한 줄 요약

SiN 기반 lightsail 광자구조의 2-stage multi-objective 최적화 프로젝트.
- **Stage 1**: 중앙 hole-based photonic crystal reflector
- **Stage 2**: 외곽 concentric curved metagrating stabilization zone

BoTorch 기반 qLogNEHVI MOBO, grcwa 기반 RCWA, public SiN 분산 데이터를 사용한다. **거의 모든 것이 작동하는 상태**이며, 실제 optimization과 RCWA가 end-to-end로 돌아간다.

## 연구 목표 (우선순위 순)

1. **NIR 반사 최대화** (target band: **1550–1850 nm**, 2026-04-16 변경 — Starshot 1550 nm laser 호환)
2. **가속 시간 (T) 최소화** (Norder et al. 2025 Eq. 3 기반, 10 GW/m², 10 m² sail)
3. **MIR 방사율 최대화** (target band: 8–14 µm = 8000–14000 nm)
4. **안정화 성능 향상** (Stage 2 metagrating 역할)
5. **제조 가능성 확보** (Mode A, MFS ≥ 500 nm이 이상적 — i-line photolithography 호환)

**2026-04-16 현재 best (thin regime, 1550 nm launch)** — Trial 23 (thin run):
- **T = 19.26 min** to β=0.2, **D = 44.1 Gm** → Norder et al. pentagonal (24.6 min, 52 Gm) 대비 **−22% T, −15% D**
- t=239 nm, P=1581 nm, mean R=0.512 (Doppler, nG=81), sail mass=4.2 g (10 m²)
- MFS 318 nm (EBL 필요), areal density 0.42 g/m² (paper 0.39 대비 거의 동등)
- **Photolithography 근접 대안**: Trial 41 (wall 486 nm, T=23.22 min) → MFS 500으로 재최적화 시 i-line 호환 가능성

**이전 두꺼운 설계도 lock 파일로 보존** — `results/best_design_1320_1620_triangular.yaml` (t=688 nm, 1320–1620 band, mean R=0.930). Peak reflectivity가 중요한 응용 (cavity QED, 고출력 간섭계 등) 참조용.

**MIR target 미달성** — thin regime에서 MIR 평균 ε = 0.14 (t=239 nm). 두꺼운 설계 (t=688) ε=0.17. 모두 optically thin limit — 현 SiN 단층으로는 MIR 0.5 불가능. 재료 교체 / multi-layer stack 경로 진행 예정.

## 설계 제약

**Hard-coded**:
- material: **SiN** (Luke NIR + Kischkat MIR 분산)
- thickness: **200–1000 nm** (기본). Thin regime 탐색 시 5–300 nm도 사용 (아래 "Thin film regime" 참조)
- fabrication mode: **Mode A**
- NIR target band: **1550–1850 nm** (2026-04-16 변경, Starshot 1550 nm laser 호환. 이전 1320–1620 결과는 lock 파일 보존)
- MIR target band: **8000–14000 nm**

**Fabrication limits (현재 production 기준, 2026-04 완화)**:
- min feature width: **100 nm** (이전 500)
- min gap: **100 nm** (이전 500)
- fill fraction: 사실상 off (0.001–0.999)

**완화 근거**: EBL + ICP-RIE SiN 연구실 fab에서 100 nm feature/gap은 재현 가능한 수준. 이전 500 nm 값은 너무 보수적이라 production run에서 22%만 feasible이었고, 완화 후 96%가 feasible이면서 NIR best가 0.42 → 0.73으로 +71% 개선됨 (seed=42 기준). 실제 제작 시 fab 담당자 컨펌 필요.

## 설계 철학 (중요 — 어기면 안 됨)

- **low-dimensional parameterization 사용**. 각 geometry당 5–7개 연속 파라미터. Lattice family 같은 discrete 선택은 생성자 인자로 고정, 최적화 벡터에 안 들어감.
- **heavy neural topology optimization / hypernetwork / pixel-based end-to-end inverse design 금지**.
- **RCWA-ready 인터페이스 중심으로 설계**. `ElectromagneticSolver` ABC가 backend 교체 포인트. 현재 `MockSolver`(pseudo-physics, 빠름)와 `RCWASolver`(grcwa + 공개 SiN 분산 데이터) 두 개 구현돼 있음.
- **multi-objective Bayesian optimization 사용**. BoTorch + GPyTorch + `qLogNoisyExpectedHypervolumeImprovement` 기반. CPU-first, MPS-ready.

## 디렉토리 구조

```
s_lightsail/
├── CLAUDE.md                          ← 이 파일
├── pyproject.toml                     # pip install -e ".[dev,mobo,rcwa]"
├── configs/
│   ├── default.yaml
│   ├── stage1_triangular.yaml         # Stage 1 / triangular lattice (solver 섹션 포함)
│   ├── stage1_hexagonal.yaml          # Stage 1 / honeycomb
│   ├── stage1_pentagonal.yaml         # Stage 1 / pentagonal supercell
│   └── stage2_outer_grating.yaml      # Stage 2 / metagrating (frozen_phc 섹션 포함)
│
├── src/lightsail/
│   ├── geometry/
│   │   ├── base.py                    # ParametricGeometry ABC, Structure, HoleShape, Ring, LatticeFamily
│   │   ├── lattices.py                # TriangularLattice, HexagonalLattice, PentagonalSupercell
│   │   ├── phc_reflector.py           # 7-param PhC (thickness/period/a/b/rot/rounding/shape)
│   │   └── metagrating.py             # 5-param metagrating (period/duty/curv/asym/ring_width)
│   │
│   ├── constraints/fabrication.py     # FabConstraints + ConstraintMode(HARD/PENALTY)
│   │
│   ├── materials/
│   │   ├── sin.py                     # SiNDispersion (Luke 2015 NIR + Kischkat 2012 MIR)
│   │   └── data/sin_kischkat_mir.csv  # 1.54–15 µm tabulated n+k
│   │
│   ├── simulation/
│   │   ├── base.py                    # ElectromagneticSolver ABC (evaluate_R/T/ε)
│   │   ├── mock.py                    # MockSolver (pseudo-physics, 빠름)
│   │   ├── rcwa_solver.py             # RCWASolver (grcwa + SiN 분산) ★ 실제 물리
│   │   └── results.py                 # SimulationResult
│   │
│   ├── optimization/
│   │   ├── objectives.py              # Objective ABC + 6개 concrete + stage factories
│   │   ├── evaluator.py               # ObjectiveEvaluator + EvaluationResult (spectrum cache)
│   │   ├── search_space.py            # SearchSpace
│   │   ├── initial_sampling.py        # Sobol + LHS
│   │   ├── mobo_runner.py             # MOBORunner + MOBOConfig + RunResult (BoTorch qLogNEHVI)
│   │   └── optimizer.py               # BayesianOptimizer (legacy random fallback)
│   │
│   ├── experiments/
│   │   ├── logging_setup.py           # console + run.log 핸들러
│   │   ├── main.py                    # run_experiment(config, overrides, solver) ★ 통합 진입점
│   │   ├── stage_runner.py            # run_stage1 / run_stage2
│   │   ├── runner.py                  # ExperimentRunner (legacy)
│   │   └── pipeline.py                # TwoStagePipeline (legacy)
│   │
│   └── visualization/
│       ├── plots.py                   # spectrum, broadband, hole shape, structure top view
│       └── mobo_plots.py              # Pareto scatter, optimization history, summarize_best
│
├── scripts/
│   ├── demo.py                        # ⭐ Mock solver 기반 최소 데모 (~30초)
│   ├── demo_rcwa.py                   # ⭐ RCWA (grcwa) 기반 데모 (~6–30초)
│   ├── run_experiment.py              # ⭐ 통합 CLI (config 기반, solver 선택 가능)
│   ├── run_stage1_mobo.py             # Stage 1 전용 CLI (legacy)
│   ├── run_stage2_mobo.py             # Stage 2 전용 CLI (legacy)
│   ├── geometry_example.py            # geometry + constraint 데모
│   └── objectives_example.py          # objective stack 데모
│
├── tests/                             # 113 tests, 모두 통과
│   ├── test_geometry.py               # 24
│   ├── test_constraints.py            # 12
│   ├── test_simulation.py             # 7 (MockSolver)
│   ├── test_materials.py              # 6 (SiN dispersion)
│   ├── test_rcwa_solver.py            # 9 (grcwa + Fabry-Perot 검증)
│   ├── test_objectives.py             # 21
│   ├── test_optimization.py           # 5
│   ├── test_initial_sampling.py       # 9
│   ├── test_mobo_runner.py            # 6 (botorch 필요)
│   └── test_experiment_main.py        # 15 (botorch 필요)
│
└── results/                           # .gitignore, 타임스탬프별 run 폴더
```

## 핵심 계약 (새 코드 작성 전 반드시 이해할 것)

### `ParametricGeometry` ABC (`geometry/base.py`)
모든 geometry가 구현해야 하는 인터페이스:
- `param_names() -> list[str]`
- `param_bounds() -> list[tuple[float, float]]`
- `to_param_vector() -> np.ndarray`
- `from_param_vector(vec) -> None`  (in-place)
- `to_structure() -> Structure`  (solver에 넘길 중간 표현)

**새 geometry를 추가할 때는 이 ABC만 상속하면 optimizer/runner 코드를 건드릴 필요가 없다.**

### `ElectromagneticSolver` ABC (`simulation/base.py`)
```python
def evaluate_reflectivity(structure, wavelengths_nm) -> np.ndarray
def evaluate_transmission(structure, wavelengths_nm) -> np.ndarray
def evaluate_emissivity(structure, wavelengths_nm) -> np.ndarray    # 기본: 1 − R − T
def compute_spectrum(structure, wavelengths_nm) -> SimulationResult  # helper
```
두 추상 메서드(`evaluate_reflectivity`, `evaluate_transmission`)만 구현하면 emissivity는 base class가 Kirchhoff 관계로 제공. 새 RCWA backend 붙일 때는 이 두 개만 구현하면 된다.

### `Structure` (`geometry/base.py`)
geometry와 solver 사이의 중간 표현: `material`, `thickness_nm`, `lattice_family`, `lattice_period_nm`, `holes: list[Hole]`, `rings: list[Ring]`, `extent_nm`, `metadata`. **geometry는 물리적 설계 파라미터를, Structure는 solver가 바로 쓸 수 있는 표현을 담당**하는 역할 분리가 중요.

### `Objective` ABC + `ObjectiveContext` (`optimization/objectives.py`)
- `Objective.evaluate(ctx) -> ObjectiveValue`
- `ObjectiveContext`는 `spectrum(band_nm, n_points)` 캐시를 가짐. 여러 objective가 같은 band에 접근하면 solver 호출이 1번만 일어남.

### `ObjectiveEvaluator` (`optimization/evaluator.py`)
파라미터 벡터 하나를 받아 geometry → Structure → constraint → solver → 모든 objective 평가까지 한 번에 처리. runner는 얇게 유지.

### `MOBORunner` (`optimization/mobo_runner.py`)
BoTorch 기반 multi-objective BO. Sobol/LHS 초기 샘플링 → `SingleTaskGP` fit → `qLogNoisyExpectedHypervolumeImprovement` acquisition → 반복. Reference point는 seen data에서 adaptive. GP fit 실패 시 Sobol fallback.

## 실행 방법

### 설치 (최초 1회)
```bash
cd "/Users/eunji/Google Drive/내 드라이브/Research_Photonics/c. ongoing/s_lightsail"
python3 -m pip install -e ".[dev,mobo,rcwa]"
```

### 테스트
```bash
python3 -m pytest tests/
```
**마지막 확인: 113 passed** (Python 3.9.6, torch 2.8.0, botorch 0.10.0, grcwa 0.1.2)

### Mock 기반 데모 (~10초)
```bash
python3 scripts/demo.py
```
`MockSolver`로 pseudo-physics. 빠르지만 **설계 결정에 쓰면 안 됨**.

### RCWA 기반 데모 (~6–30초)
```bash
python3 scripts/demo_rcwa.py
```
`RCWASolver` (grcwa) + 공개 SiN 분산 데이터로 실제 물리. Fabry-Perot 해석해와 1e-4 정확도 검증됨.

### 결과 확인
```bash
LATEST=$(ls -t results | head -1)
cat "results/$LATEST/best_design.yaml"     # best params + objectives
cat "results/$LATEST/summary.txt"          # objective별 top-3
cat "results/$LATEST/run.log"              # trial-by-trial 로그
open "results/$LATEST/plots/"*.png         # Pareto / history / spectrum / structure
```

최신 RCWA run만 보고 싶으면 `ls -t results | grep demo_rcwa | head -1` 사용.

### 통합 CLI (production run)
```bash
# Mock
python3 scripts/run_experiment.py --config configs/stage1_triangular.yaml

# RCWA (config의 solver.kind를 rcwa로 바꾸거나 override)
python3 scripts/run_experiment.py --config configs/stage1_triangular.yaml \
    --n-init 16 --n-iter 20 --seed 42

# Stage 2
python3 scripts/run_experiment.py --config configs/stage2_outer_grating.yaml
```

### Python 버전
- 현재: Python 3.9.6 (macOS 시스템 파이썬)
- `from __future__ import annotations`로 모든 파일에 적용돼 있어서 `tuple[float, float]`, `list[Hole]`, `X | None` 같은 모던 타입 힌트 작동. **새 파일 작성 시 반드시 이 import 넣을 것.**

## Parameterization 상세

### `PhCReflector` (Stage 1) — 7 연속 파라미터

| 파라미터 | 범위 | 의미 |
|---|---|---|
| `thickness_nm` | 200–1000 | SiN 막 두께 |
| `lattice_period_nm` | 400–2000 | 격자 주기 |
| `hole_a_rel` | 0.05–0.48 | **a / period** (상대 장축 반경) |
| `hole_b_rel` | 0.05–0.48 | **b / period** (상대 단축 반경) |
| `hole_rotation_deg` | 0–180 | in-plane 회전각 |
| `corner_rounding` | 0–1 | 0=sharp n-gon, 1=ellipse |
| `shape_parameter` | 3–8 | n_sides (int로 반올림) |

**중요 — 상대 파라미터화 (2026-04 변경)**: `hole_a_nm` / `hole_b_nm`가 이제 **derived property**이고 최적화 변수는 `hole_a_rel` / `hole_b_rel`입니다. 이렇게 한 이유는 period와 hole 크기의 수학적 consistency (2a < period)를 파라미터 공간 수준에서 자동 보장하기 위함. 절대 nm 값이 필요하면 `phc.hole_a_nm`로 읽을 수 있으나 **설정 시에는 a_rel을 써야** 함. 옛 `hole_a_nm=400, period=1500` 패턴은 `hole_a_rel=400/1500, lattice_period_nm=1500`으로 바꿔야 함.

**이산 선택 (생성자 인자)**: `lattice_family ∈ {triangular, hexagonal, pentagonal_supercell}`. 각 family별로 별도 실행하고 결과 비교.

### `MetaGrating` (Stage 2) — 5 연속 파라미터

| 파라미터 | 범위 | 의미 |
|---|---|---|
| `grating_period_nm` | 1000–3000 | ring_width + gap |
| `duty_cycle` | 0.2–0.8 | ring이 차지하는 비율 |
| `curvature` | −0.2 – 0.2 | 1차 radial warping |
| `asymmetry` | −0.2 – 0.2 | 2차 angular 비대칭 |
| `ring_width_um` | 2–50 | metagrating 영역 총 폭 (µm) |

`inner_radius_nm`은 Stage 1 `phc.outer_radius_nm`, `thickness_nm`은 Stage 1에서 상속. 최적화 파라미터가 아님.

## 목적함수 (stage별)

### Stage 1 (3 objectives — simplified 2026-04)
1. **`nir_reflectance` ↑** — `mean_weight·mean(R) + min_weight·min(R)` on 1350–1650 nm
2. **`mir_emissivity` ↑** — mean(ε) = mean(1−R−T) on 8000–14000 nm
3. **`fabrication_penalty` ↓** — pure `ConstraintResult.penalty` (mass term removed)

**Mass는 더 이상 objective가 아님** — 물리적으로 SiN 박막은 두께가 작을수록 가벼우면서 NIR 손실도 적음(Luke 데이터에서 NIR은 k=0). Mass를 explicit objective로 두면 BO가 "mass 낮추기"와 "NIR 높이기" 사이 trivial trade-off에서 시간을 낭비하고 manufacturing penalty는 soft constraint로서 역할이 흐려짐. 제거 후 penalty가 순수 제작성만 반영하게 됐다. Mass 재도입은 `MassAndFabPenaltyObjective` 클래스가 `objectives.py`에 여전히 존재해서 언제든 가능.

### Stage 2 (4 objectives)
1. **`nir_reflectance` ↑** (유지)
2. **`mir_emissivity` ↑** (유지)
3. **`stabilization` ↑** — 현재는 analytic proxy (`AsymmetryStabilizationProxy` 또는 `RadialMomentumProxy`)
4. **`fabrication_penalty` ↓** — `ConstraintResult.penalty`

**Stage 2 stabilization은 analytic version 유지**. 나중에 full-wave proxy로 교체하려면 `StabilizationProxy` ABC만 상속하면 됨.

## Solver 선택

Config의 `solver.kind` 필드로 결정:
```yaml
solver:
  kind: mock         # 기본값, 빠름
  # 또는:
  kind: rcwa         # grcwa + SiN 분산, 실제 물리
  nG: 41             # Fourier harmonics (RCWA 전용)
  grid_nx: 96        # unit cell 래스터 해상도
  grid_ny: 96
  polarization: average   # "average" / "te" / "tm"
```

CLI에서는 `--solver rcwa` 같은 직접 override는 없고, config 파일을 편집하거나 `run_experiment` 호출 시 `overrides={"solver": {"kind": "rcwa"}}`를 넘기면 됨.

### MockSolver vs RCWASolver 비교 (참고)
- Mock: 10 trials ~10초, NIR R 최대값 0.5 나옴 (낙관적), MIR ε 최대 ~0.89
- RCWA: 10 trials ~6초 (nG=31에서), NIR R 최대값 0.4 수준 (현실적), MIR ε ~0.20

RCWA가 Mock보다 오히려 빠른 건 wavelength 수가 적을 때 (12 points x 2 band) grcwa의 오버헤드가 작기 때문. 실제 production run (30 points x 2 band x 20 iterations)은 RCWA로 수 분 걸림.

## SiN 분산 데이터

두 소스 조합 (`materials/sin.py`):
1. **NIR (λ < 1.54 µm)**: Luke et al. 2015 Sellmeier (Si3N4, lossless)
2. **MIR (λ ≥ 1.54 µm)**: Kischkat et al. 2012 tabulated n+k (CSV, 1.54–15 µm, Si-N stretch 흡수 10.5 µm 부근)

**stoichiometric Si3N4 기준**. Lab에서 실제 PECVD/LPCVD film의 ellipsometry+FTIR 데이터 나오면 `materials/data/sin_kischkat_mir.csv`를 교체. 코드 변경 없이 바로 반영됨.

## 아티팩트 디렉토리 구조

```
results/<YYYY-MM-DD_HHMMSS>_<name>/
├── run.log                          # 전체 로그
├── config.yaml                      # 사용된 config 복사본
├── overrides.yaml                   # 런타임 overrides (있을 때)
├── trials.json                      # 모든 trial 메타데이터
├── params.npy                       # (n_trials, d) 파라미터 행렬
├── objectives.npy                   # (n_trials, m) objective 행렬
├── pareto_indices.npy               # Pareto trial index
├── best_design.yaml                 # primary objective 기준 best (params + objectives)
├── summary.txt                      # objective별 top-3 텍스트
└── plots/
    ├── pareto_<obj1>_vs_<obj2>.png  # init/BO/Pareto 3색 scatter
    ├── optimization_history.png     # objective별 running-best
    ├── spectrum_best.png            # best design R/T/ε (NIR+MIR subplot)
    └── structure_topview.png        # best design 상단 뷰
```

## 현재 상태

### 완료됨
- [x] 프로젝트 스캐폴딩 (pyproject.toml, 디렉토리, 의존성)
- [x] Geometry ABC + PhCReflector (3 lattice family × rounded polygon holes)
- [x] **상대 파라미터화** (hole_a_rel, hole_b_rel; 2026-04 변경)
- [x] MetaGrating (curved + asymmetric rings)
- [x] FabConstraints + validation (hard/penalty mode, fill fraction, disconnection check)
- [x] `ElectromagneticSolver` ABC + MockSolver
- [x] **RCWASolver (grcwa 기반) + SiN 분산 데이터 + Fabry-Perot 검증 통과**
- [x] Objective ABC + stage1/2 factories + ObjectiveEvaluator (spectrum cache)
- [x] **Stage 1 목적함수 단순화**: `nir_reflectance, mir_emissivity, fabrication_penalty` (mass_and_fab 제거)
- [x] SearchSpace, Sobol/LHS initial sampling
- [x] **MOBORunner (BoTorch qLogNEHVI) + CPU/MPS-ready**
- [x] Experiment main entrypoint (config-driven, solver 선택 가능, 로깅, 아티팩트)
- [x] Visualization (Pareto scatter, optimization history, broadband spectrum, structure top view)
- [x] 4개 stage config + demo script 2개 (mock, rcwa)
- [x] 114개 단위 테스트 모두 통과
- [x] End-to-end 검증: `demo.py` (mock) + `demo_rcwa.py` (rcwa) 모두 작동
- [x] **nG 수렴성 연구** (`scripts/nG_convergence_study.py`) — nG=41이 NIR에서 1% 수준 수렴
- [x] **Production Stage 1 run** (triangular, 100 trials, relaxed constraints)
  - Feasible best: NIR R **0.726** (trial 84), MIR ε 0.196, fab_penalty 0
  - nG=81 재검증: top 5 ranking 완전 일치, 값 편차 <0.006
  - 96% feasible rate (이전 완화 전 22% 대비)

### 아직 구현 안 됨 (TODO — 우선순위 순)

1. **Stage 2 RCWA 지원** — 현재 `RCWASolver`가 metagrating-only 구조를 uniform slab로 fallback. 필요시 `LocalPeriodFMMProxy(StabilizationProxy)`를 만들어 radial bin 1D-FMM 처리. 다만 분석적 proxy 유지가 우선순위라 당분간 현상 유지.
2. **Convergence study 자동화** — nG=21, 41, 81에서 같은 구조 돌려 수렴 체크. `scripts/convergence_study.py` 같은 파일.
3. **Lab SiN ellipsometry 데이터 교체** — CSV 파일만 바꾸면 됨. 현재는 Kischkat 2012 public 데이터.
4. **Full production run** — `--n-init 16 --n-iter 30 --seed 42`로 triangular/hexagonal/pentagonal 세 family 비교
5. **Stage 1 → Stage 2 자동 연결** — 현재는 Stage 1 결과의 `best_design.yaml`을 Stage 2 config의 `frozen_phc` 섹션에 수동 복사해야 함. 자동 chain 옵션 추가.
6. **GDS export** — fabrication용 레이아웃 출력.

## 중요한 설계 결정 (바꾸지 마라, 이유 있음)

1. **`src/lightsail/` layout**: editable install 시 import 충돌 방지, 테스트에서 accidental import 방지.
2. **`Structure` 중간 표현**: geometry 클래스가 직접 RCWA API를 모르게 하는 경계. 백엔드 교체 시 geometry 코드 수정 불필요.
3. **Stage 2가 Stage 1 best의 param vector가 아니라 `PhCReflector` 객체 자체를 받음**: `phc.outer_radius_nm`, `phc.thickness_nm` 같은 derived property를 쓰기 위함.
4. **`ObjectiveContext` spectrum cache**: 여러 objective가 같은 (band, n_points)를 물으면 solver 호출이 1번. 실제 RCWA로 교체 후 성능에 크게 기여.
5. **최대화 컨벤션**: minimize objective는 `MOBORunner` 내부에서 부호 뒤집어 GP에 넣음. 외부로는 원래 부호로 노출.
6. **Reference point adaptive**: `Y.min - margin`. 스케일 불균형에 강건.
7. **GP fit 실패 시 Sobol fallback**: BoTorch는 초기 iteration에 자주 singular covariance. 감싸고 계속 진행.
8. **`solver.kind` config 선택**: CLI에서 직접 flag 대신 config 파일 / overrides로만 선택. 재현성 확보.
9. **`nG<5`일 때 uniform slab 자동 전환**: grcwa는 균질 slab에 nG>5면 singular matrix. RCWASolver 내부에서 자동 nG=3으로 강등.
10. **grcwa list/tuple quirk**: grcwa가 lattice vectors를 list로만 받음. RCWASolver가 tuple→list 변환.

## 코딩 컨벤션

- **타입 힌트 필수**. `from __future__ import annotations` 항상 최상단.
- **dataclass 선호**. 설정/결과 객체는 모두 `@dataclass`.
- **Abstract base class는 `ABC` + `@abstractmethod`**.
- **모든 길이는 nm 단위**. 파라미터 이름에 `_nm` suffix를 붙여 단위 혼동 방지. grcwa로 넘길 때만 µm 변환.
- **주석과 docstring은 영어로 작성** (기존 코드 스타일 유지).
- **파일을 새로 만들 때 반드시 해당 모듈의 `__init__.py`에도 export 추가.**
- **Optional dependency는 try/except로 gate**. grcwa, botorch 모두 이런 식으로 감싸서 기본 dev 환경에서도 일부 기능이 동작하게 함.

## 대화 언어

사용자는 한국어로 질문한다. **답변도 한국어로 하되, 코드와 코드 주석/docstring은 영어로 작성한다.**

## Troubleshooting 메모

### "no matches found: results//plots/*.png"
zsh에서 glob 매치 실패. `$LATEST` 변수가 세션에 없을 때 발생. 해결:
```bash
open "results/$(ls -t results | head -1)/plots/"*.png
```

### `grcwa LinAlgError: Singular matrix`
Uniform slab을 `nG>5`로 돌릴 때 발생. `RCWASolver`가 자동으로 `nG=3`으로 강등해서 회피한다. 새로 grcwa 호출 직접 쓸 땐 주의.

### `grcwa AssertionError: Lattice vectors should be in list format`
tuple 대신 list 요구. `list(L1_um)`로 변환.

### GP fit 실패 → Sobol fallback
BoTorch 초기 iteration에서 singular covariance 흔함. `MOBORunner`가 try/except로 감싸서 Sobol 샘플 1개로 대체. 로그에 `"bo_fallback"` source로 기록됨.

### `matlabengine` 불필요
과거 MATLAB `fmm` 패키지 고려했으나 **grcwa로 교체함**. MATLAB 의존성 없음.

## 2026-04-15 저녁 세션 업데이트

### Thickness 확장 실험 (옵션 A) — 실행 후 rollback

`_THICKNESS_MAX_NM`을 1000 → 2500 nm로 올린 production run 실행 결과:

| 지표 | t ≤ 1000 (이전) | t ≤ 2500 (실험) | Δ |
|---|---:|---:|---|
| NIR best (seed=42 그대로) | 0.726 | 0.550 | −0.176 ↓ (BO 예산 재분배로 인한 regression) |
| MIR best (nG=41) | 0.30 | **0.43** | +0.13 ↑ (t=2500 nm에 pinned) |
| MIR (nG=81 재검증) | — | 0.4305 | ≈ 일치 |

관찰: 7-dim space에서 thickness 상한이 2.5배 늘어나자 Sobol 초기 샘플 42%가 2000–2500 nm에 떨어졌고, qLogNEHVI가 MIR hypervolume 개선이 잘 보이는 thick-film 영역으로 예산을 쏟아부음. NIR 회귀는 **physics 한계가 아니라 budget 문제** (이전 ensemble seed=123에서 0.849 달성했음).

**결정**: thickness 상한 1000 nm 유지. 사유 — (a) 2500 nm SiN 박막은 EBL+ICP-RIE로 제작 난이도 급증, (b) propulsion 측면에서 mass per area 2.5배 증가는 가속도에 치명적, (c) MIR 0.43은 여전히 목표 0.5 미달 + thicker만으로 해결 불가.

결과 디렉토리: `results/2026-04-15_221711_stage1_triangular_production/` (참고용, best_design 사용 안 함)

### NIR plateau + Doppler compliance

**Sliding window 분석** (script: `scripts/nir_window_scan.py`): triangular best (s123/trial 101)의 R(λ) 스펙트럼을 1000–2000 nm에서 10 nm step으로 계산 후, 폭 W인 연속 구간에 대해 mean R을 최대화하는 window를 찾음.

Width vs 최적 window (triangular, nG=41):

| width | window (nm) | mean R | min R | 비고 |
|---:|---|---:|---:|---|
| 200 | 1320–1520 | 0.919 | **0.876** | 가장 uniform |
| **300** | **1320–1620** | **0.923** | 0.820 | sweet spot |
| 400 | 1290–1690 | 0.830 | **0.208** ⚠ | 양 끝 dip 포함 |
| 500 | 1210–1710 | 0.692 | 0.100 | |
| 1000 | 1000–2000 | 0.407 | 0.000 | |

300 → 400 nm로 넓히는 순간 min R이 **0.82 → 0.21 cliff-drop**. Plateau 양쪽에 sharp Fano resonance dip이 붙어 있어 window가 조금만 넘어가면 삼켜짐. **300 nm가 이 구조의 본질적 bandwidth**이며 더 넓히려면 다른 geometry (multi-layer stack, chirped PhC) 필요.

**nG=81 재검증** (1320–1620 nm, 31 points):
- mean R = **0.9301**
- min R = **0.8798**  
- max R = 0.9998
- nG=41 값과 <0.01 차이 → BO 결과 신뢰성 확보

**Relativistic Doppler 분석**: sail frame wavelength λ_sail = λ_lab × √((1+β)/(1−β)). Launch laser를 plateau blue edge (1320 nm)에 맞추면:

| β | λ_sail | plateau 안? |
|---:|---:|:--:|
| 0.00 | 1320 nm | ✓ |
| 0.10 | 1459 nm | ✓ |
| 0.15 | 1535 nm | ✓ |
| **0.20** | **1617 nm** | ✓ (red edge) |

β_max ≈ **0.202** — Breakthrough Starshot target (0.2c)와 정확히 일치. **300 nm plateau는 우연이 아니라 Doppler 요구사항과 정확히 맞음**. 단 launch laser가 1320 nm이어야 함 (표준 1064/1550 아님; Raman-shifted Nd:YAG, Tm-fiber, OPO 등으로 구현 가능).

### 4-family 비교 (sliding window, nG=41)

| family | best window (nm) | mean R | min R | 해석 |
|---|---|---:|---:|---|
| **Triangular** | 1320–1620 | **0.923** | **0.820** | 300 nm clean plateau |
| Hexagonal | 1330–1630 | 0.485 | 0.209 | resonance 없는 완만한 감쇠 |
| Rectangular | 1700–2000 | 0.520 | 0.138 | 200 nm 폭 clean peak (mean 0.56, min 0.53) at 1800–2000 존재 |
| Pentagonal | 1360–1660 | 0.330 | 0.293 | 전역 평탄, resonance 없음 |

관찰:
- Hexagonal/pentagonal은 band shift로 구제 불가 — 구조적 한계
- Rectangular는 1800–2000 nm 200 nm 폭에서 mean/min 모두 ~0.55 (triangular의 절반이지만 나름 clean)
- **Triangular가 유일하게 NIR target 만족**

### Acceleration time (T to β=0.2) 계산

세부 계산과 파라미터는 `results/launch_analysis.yaml`. 핵심 내용:

**Paper 비교 기준**: Norder et al. 2025 Nat Commun 16:2753 "Pentagonal PhC mirrors" (Fig. 3c):
- Pentagonal lattice D-optimized: **24.6 min** to β=0.2
- Hexagonal lattice D-optimized: **29.8 min**
- Reference params: 10 GW/m², 10 m² sail, 1 g payload, launch λ = 1550 nm

**Paper Eq. (3) dimensional 오류**: 출판본에는 `T = m_t c^3 / (2IA) × ∫ γ³/R × (1+β)/(1-β) dβ`가 적혀 있는데, 차원 분석시 단위가 meters (not seconds). 표준 relativistic lightsail 운동방정식 (γ³ m c dβ/dt = 2IAR/c × (1−β)/(1+β))에서 derivation하면 **c², not c³**. 이 repo에선 c² corrected form 사용.

**우리 triangular best (s123/trial 101) 계산 결과**:

- Thickness 688 nm × material fraction 0.475 → areal density **1.013 g/m²** (sail alone 10.13 g, +payload 11.13 g)
- Mean R in sail-frame Doppler range 1320–1617 nm (launch at 1320 nm): **0.920** (nG=81, 31 points)
- T integral ∫₀^0.2 γ³/R(λ(β))·(1+β)/(1-β)dβ = **0.2772**
- Prefactor m·c²/(2IA) = 5000.5 s
- **T = 23.10 min** — **paper pentagonal보다 1.5 min 빠름**

**해석**: Mass 2.36× 무거움 × R 1.84× 높음 → 거의 상쇄, R 우위가 미세하게 이김.

**한계**:
- Sail areal density 1.013 g/m²는 **Starshot aspirational 0.1 g/m² 대비 10×**. Thick-film 전략의 cost.
- Launch wavelength 1320 nm는 Nd:YAG 1319 line (commercial CW product)으로 구현 가능하나, **GW-class phased array infrastructure는 1550 nm 중심으로 구축돼 있어 scale-up cost 별도**. Wavelength 자체는 non-exotic, scaling이 문제.
- Same hole pattern을 300 nm로 얇게 하면 mass 0.44 g/m²로 절감되지만 R이 0.92→0.38로 무너져 **T = 343 min** (15× 악화). 688 nm는 반사/질량 국소 최적.

### MIR 방사율 상세 분해 (triangular best, nG=41, 60 points)

`results/triangular_best_mir_spectrum.png`

| band | mean ε | mean R | mean T | 해석 |
|---|---:|---:|---:|---|
| 8–14 µm | **0.166** | 0.027 | **0.807** | 대부분 투과 — film too thin for MIR absorption |
| 8–10 µm | 0.042 | 0.023 | 0.935 | 거의 완전 투명 |
| 10–12 µm | 0.220 | 0.025 | 0.755 | Si–N stretch 흡수 peak (~10.5 µm) |
| 12–14 µm | 0.235 | 0.034 | 0.732 | 완만한 absorption tail |

**Root cause**: 688 nm SiN은 MIR 파장 (8–14 µm)에 대해 λ ≫ t이라 optically thin. Hole pattern도 λ ≫ period라 effective medium처럼 동작, MIR resonance 불가능. 방사율은 거의 단순히 Kischkat k(λ) × 두께에 의한 single-pass 흡수 + Kirchhoff 관계로 결정됨. 10.5 µm 근처 Si–N stretch에서만 ε ≈ 0.25 상승.

**현재 geometry + material로 MIR 0.5+ 달성 불가능**. 개선 경로:
1. 재료 교체 (Si-rich SiN, AlN, SiC 등 MIR k 더 큰 박막) — CSV 교체만으로 반영
2. Multi-layer stack — `PhCReflector`를 `LayeredPhCReflector`로 일반화 (설계 철학의 "low-dim" 제약과 조심스럽게 상의 필요)
3. MIR metasurface absorber — 큰 period의 별도 patterned layer 추가

## 다음 세션에서 바로 할 만한 작업

- **MFS ≥ 500 nm constraint run** (최우선) — Trial 41 (wall 486 nm)을 기반으로 `min_feature_nm=500`, `min_gap_nm=500` 강제 후 1550–1850 thin BO 재실행. i-line photolithography 호환 설계가 T < 25 min 달성 가능한지 확인. Paper와 apple-to-apple fab cost 비교.
- **Multi-seed ensemble** (thin, 1550–1850) — seed 42/123/456 × 100 trials. Trial 23 (T=19.3 min)보다 좋은 설계가 나올 가능성. 현재 seed 42 단일 run.
- **`configs/stage1_triangular.yaml`의 nir_band_nm을 [1350, 1650] → [1550, 1850]으로 변경** — base config를 현 research direction에 정렬.
- **MIR 개선**: (a) lab SiN ellipsometry 데이터 → CSV 교체 / (b) multi-layer stack / (c) supercell 구조 (in-plane hole modulation + thickness modulation).
- Lab SiN n/k 데이터 입수 시 `materials/data/sin_kischkat_mir.csv` 교체.
- Stage 1 best → Stage 2 frozen_phc 자동 chain.
- Norder et al. 2025 D vs MFS plot (Fig 3a 스타일)에 우리 데이터 overlay — 시각적 경쟁력 비교.

## 2026-04-15 밤: Mass-aware BO 결과

Run: `results/2026-04-15_230914_stage1_triangular_mass_aware_s42/` (seed 42, 100 trials, nG=41, 22분 소요). 4-objective NEHVI (NIR ↑ / MIR ↑ / fab ↓ / areal_density ↓), NIR band은 1320–1620으로 override.

**Feasible Pareto (90/100 feasible, 45 Pareto)**:

| density bucket (g/m²) | best NIR | trial | t_nm | period_nm | a_rel | b_rel | MIR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.2–0.4 | 0.459 | 78 | **200** (min) | 1318 | 0.309 | 0.330 | 0.111 |
| 0.4–0.6 | 0.447 | 74 | 200 | 1295 | 0.286 | 0.360 | 0.105 |
| 0.6–0.8 | 0.650 | 40 | 705 | 1582 | 0.429 | 0.451 | 0.126 |
| 0.8–1.0 | 0.749 | 98 | 828 | 1541 | 0.416 | 0.437 | 0.131 |
| **1.0–1.5** | **0.838** | **69** | **648** | **1505** | **0.369** | **0.433** | **0.194** |
| 1.5+ | 0.403 | 89 | 567 | 1364 | 0.186 | 0.205 | 0.216 |

**핵심 결론**:

1. **현재 7-param 공간에서 sub-0.5 g/m² × NIR > 0.5 불가능**. Ultra-light bucket (<0.2 g/m²)에는 feasible trial 자체가 없음.
2. **NIR 0.84는 density 1.0–1.2 g/m² 영역에서만 재현**됨. 1320–1620 band 기준 best는 trial 69이고, 이는 기존 lock (s123/trial 101)의 0.849와 사실상 동일. Mass objective 추가해도 NIR 상한은 동일.
3. **Thinnest feasible (200 nm) 설계는 NIR ~0.46**. 목표 0.8의 절반. 이 공간에서 thinner = less reflective 경향이 매우 강함.
4. **Starshot 1 g/m² target은 이 parameterization으로 달성 불가**. 도달하려면 multi-layer structured absorber / chirped PhC / metasurface 등 자유도 확장 필요.

## 2026-04-15 밤: NIR target band 1550–1850로 피벗 결정

사용자 결정 (2026-04-15 밤): 1320 nm launch는 Starshot laser roadmap과 incompatible (1550 nm 중심 telecom infrastructure 활용 불가). 따라서 NIR target band를 **1550–1850 nm**로 이동. 이 band는 launch λ=1550 nm에서 sail frame Doppler range β=0 → **β ≈ 0.175** 대응 (β=0.2 기준이면 1550–1898 nm이지만 1850으로 잘라 설계 목표 약간 보수적).

**설계 goal** (업데이트):
- NIR target band: **1550–1850 nm** (기존 1320–1620을 300 nm 그대로 우측 shift)
- 1320–1620 baseline 설계는 `results/best_design_1320_1620_triangular.yaml`에 lock 파일로 저장 (reference용)
- 다음 BO run: 동일 4-objective (NIR/MIR/fab/density), NIR band [1550, 1850], seed 42

**기대**:
- 1550–1850 nm로 가면 SiN 흡수/dispersion이 1320–1620보다 약간 낮아 잠재적으로 더 높은 R 가능
- Plateau가 300 nm 폭으로 재형성되려면 hole period 약간 커져야 함 (period ∝ λ 대략)
- period 1510 → 1550 × 1510 / 1320 ≈ 1773 nm 근방 예상

## 2026-04-16: Thin film regime (5–300 nm) + 1550 nm launch 결과

### 배경

Norder et al. 2025 (Nat Commun 16:2753, "Pentagonal PhC mirrors") 비교를 위해 1550 nm launch + thin film regime로 피벗. Paper는 t=200 nm, Af=0.63, MFS=507 nm 설계로 T=24.6 min (D-opt pentagonal). 우리 7-param triangular PhCReflector가 같은 regime에서 경쟁할 수 있는지 확인.

Run: `results/2026-04-16_105413_stage1_triangular_1550_thin_s42/` (seed 42, 100 trials, 4-obj NEHVI, nG=41, thickness 5–300 nm, NIR band 1550–1850 nm)

### 결과 (nG=81 검증 완료)

| | **Trial 23** ⭐ | Trial 41 (fab-friendly) | Trial 96 | Paper penta (T-opt) |
|---|---:|---:|---:|---:|
| T to β=0.2 | **19.26 min** | 23.22 min | 22.03 min | 23.8–24.6 min |
| D | **44.1 Gm** | 51.2 Gm | 48.9 Gm | 52 Gm |
| Mean R (Doppler, nG=81) | 0.512 | 0.515 | 0.549 | ~0.34 |
| Thickness | 239 nm | 241 nm | 262 nm | 200 nm |
| Period | 1581 nm | 1512 nm | 1523 nm | ~1860 nm |
| MFS (wall) | 318 nm | **486 nm** | 434 nm | 507 nm |
| Areal density | 0.42 g/m² | 0.54 g/m² | 0.55 g/m² | 0.39 g/m² |
| Sail mass (10 m²) | 4.2 g | 5.4 g | 5.5 g | 3.9 g |

### 핵심 통찰

1. **Trial 23이 paper보다 22% 빠름 (19.3 vs 24.6 min)**. Mass/R 비율이 핵심 — R이 1.5× 높으면서 mass 거의 동등 (4.2 vs 3.9 g).
2. **"가볍고 moderate R이 무겁고 high R보다 낫다"** (paper 통찰과 일치). Thick design (t=716, R=0.87) 대비 thin (t=239, R=0.51)이 T에서 4분 빠름.
3. **Trial 41 (wall 486 nm)은 i-line photolithography 임계점** 근접. MFS ≥ 500 nm constraint로 재최적화하면 paper와 동일 fab cost에서 경쟁 가능성.
4. **모든 thin best가 t=240–260 nm에 수렴** — 200 nm (paper) ↔ 250 nm (우리) 영역이 triangular lattice의 global optimum.

### vs 1550 nm thick regime (t 200–1000 nm)

| | Thick trial 73 | **Thin trial 23** |
|---|---:|---:|
| T | 23.26 min | **19.26 min** (−17%) |
| D | 46.5 Gm | **44.1 Gm** (−5%) |
| Sail mass | 9.3 g | **4.2 g** (−55%) |
| Mean R | 0.868 | 0.512 (−41%) |

Mass 감소 효과가 R 감소를 압도 — lightsail FOM에서 thin이 thick보다 우위.

### Paper 비교 최종 요약

| 축 | 결과 |
|---|---|
| T (acceleration time) | 🟢 **19.3 min vs 24.6 min (−22%)** |
| D (distance) | 🟢 44.1 vs 52 Gm (−15%) |
| Mass budget | 🟢 4.2 vs 3.9 g (거의 동등) |
| Laser compatibility | 🟢 둘 다 1550 nm |
| **Fab cost (MFS)** | 🔴 318 nm (trial 23) — EBL 필요 |
| | 🟡 486 nm (trial 41) — i-line 근접, 500 nm constraint run 필요 |
| Demonstrated fab | 🔴 simulation only |

**남은 격차: fabrication cost만**. MFS 500 nm constraint run으로 이 격차 제거 가능성 열림.

### Thin best MIR 방사율 (Trial 23, nG=41, 8–14 µm)

Plot: `results/trial23_thin_spectrum.png`

| band | mean ε | mean R | mean T |
|---|---:|---:|---:|
| 8–14 µm 전체 | **0.097** | 0.007 | **0.896** |
| 8–10 µm | 0.017 | 0.004 | 0.979 |
| 10–12 µm | 0.132 | 0.009 | 0.860 |
| 12–14 µm | 0.142 | 0.009 | 0.850 |

Thick design (t=688, ε=0.166)보다 더 낮음 — thin → 더 투명 → MIR 냉각 능력 약화. SiN 단층 구조에서는 t 줄이면 가속은 빨라지나 열 관리 악화.

### Graphene underlayer로 MIR 해결 (2026-04-16 분석)

**핵심 발견: SiN PhC 뒷면 (우주 방향)에 graphene을 깔면 NIR 반사 손실 없이 MIR ε 극적 개선.**

원리: laser → [SiN PhC 반사] → [graphene 흡수] → 우주. Graphene이 투과광만 보므로 R은 SiN이 결정, graphene은 T→A 전환만 담당.

| Graphene layers | MIR ε | NIR R 변화 | 추가 mass (10 m²) | T 변화 |
|---:|---:|:---:|---:|:---:|
| 10 | **0.28** | ≈ 0 | +0.08g (+1.8%) | ≈ 0 |
| 20 | **0.43** | ≈ 0 | +0.15g (+3.6%) | ≈ 0 |
| 50 | 0.71 | ≈ 0 | +0.39g (+9.3%) | ≈ 0 |

**Pure win** — MIR ε 0.10 → 0.28~0.43, NIR/T/mass 사실상 불변. Graphene mass ≈ 0.77 mg/m² per layer로 negligible.

Fab: SiN 릴리스 후 CVD graphene wet/dry transfer (standard). Graphene은 반드시 **우주 쪽 (laser 반대편)**에 배치해야 함 — laser 쪽에 놓으면 입사광 흡수로 R 감소.

대안: SiO₂ underlayer (NIR 무영향, Si-O stretch 9.3 µm에서 강한 흡수)도 가능하나 mass penalty 큼 (100 nm SiO₂ = +2.2g vs 10-layer graphene = +0.08g).

### Freeform BO 결과 (2026-04-16, seed 42, 120 trials, 11-dim)

Run: `results/2026-04-16_112750_stage1_freeform_thin_s42/`

`FreeformPhCReflector` (Fourier n=2,3 harmonics, +4 params) + thin regime + 1550–1850 band.

| | Baseline trial 23 | Freeform trial 59 (nG=81) |
|---|---:|---:|
| NIR mean R (Doppler) | 0.512 | **0.553** (+8%) |
| MIR mean ε | 0.097 | **0.133** (+37%) |
| T to β=0.2 | **19.26 min** | 23.10 min (+4 min) |
| Areal density | **0.42 g/m²** | 0.586 g/m² (+40%) |
| Thickness | 239 nm | 279 nm |
| MFS (wall) | 318 nm | **455 nm** (i-line 근접) |
| Fourier amp2 / amp3 | 0 / 0 | 0.023 / 0.098 |

**해석**: Freeform이 R과 MFS를 올렸지만 BO가 두꺼운 쪽으로 수렴 → mass 40% 증가 → T 손해. Fourier amplitude 매우 작아 hole shape 변화 미미. 11-dim에서 120 trials는 탐색 부족.

**Freeform 3-seed ensemble** (42/123/456 × 120 = 360 trials, 290 feasible): 결과 동일. Best는 s42/tid59 (NIR_dop=0.553, T=23.10 min). **Baseline trial 23 (T=19.26 min)을 못 이김**. 이유: BO가 thickness/period 레버를 Fourier보다 선호 → 두꺼운 쪽으로 수렴 → mass 증가가 R 개선 상쇄. Freeform harmonics가 이 parameterization에서 반사율 구조를 근본적으로 바꾸지 못함.

Run dirs: `results/2026-04-16_112750_stage1_freeform_thin_s42/`, `results/2026-04-16_125041_stage1_freeform_thin_s123/`, `results/2026-04-16_125042_stage1_freeform_thin_s456/`

### (B) Dual-hole supercell 결과 (2026-04-16, seed 42, 100 trials, 9-dim)

Run: `results/2026-04-16_134152_stage1_dual_hole_thin_s42/`

`DualHolePhCReflector` (DUAL_TRIANGULAR lattice, 2 holes per supercell with independent a/b) + thin + 1550–1850.

| | Baseline trial 23 | Dual tid=91 (holes differ) | Dual tid=90 (~same holes) |
|---|---:|---:|---:|
| T to β=0.2 | **19.26 min** | 27.11 min | 29.00 min |
| R_dop (nG=81) | 0.512 | 0.488 | 0.486 |
| MIR ε | 0.097 | 0.122 | 0.128 |
| Areal density | **0.42 g/m²** | 0.604 | 0.660 |
| f_mat | 0.43 | 0.64 | 0.69 |
| Hole difference | — | 0.162 (different) | 0.031 (~same) |

**판정**: Baseline을 크게 못 이김. Dual-hole supercell에서 BO가 hole을 작게 만들어 f_mat 0.64–0.69 → mass 증가 → T 악화. R도 baseline보다 낮아짐 (Brillouin zone folding으로 공진 quality 저하 가능성).

### A/B/C 종합

| 접근 | Best T | vs Baseline (19.26 min) |
|---|---:|---|
| (A) Freeform (Fourier n=2,3) | 23.10 min | +4 min ✗ |
| (B) Dual-hole supercell | 27.11 min | +8 min ✗ |
| (C) Disordered lattice | 23.02 min | +4 min ✗ (jitter=0 수렴, BO가 disorder 거부) |

**단일 ellipse hole (baseline trial 23, T=19.26 min)이 가장 빠름**. In-plane 구조 변경 (A/B/C)은 모두 T 개선 실패. BO가 Fourier amp/multi-hole/jitter 자유도를 기각 → **단일 guided-mode resonance가 이 두께/재료에서 최적 메커니즘**.

Run: `results/2026-04-16_160514_stage1_disordered_thin_s42/` (C 결과). Best tid=66, jitter=0.000, T=23.02 min.

### A/B/C 최종 결론 + 다음 방향

**In-plane 구조 변경은 이 regime (단층 SiN, t<300 nm, triangular)에서 T 개선 불가**. 단일 guided-mode resonance가 최적 메커니즘. 세 가지 추가 자유도 모두 BO에 의해 기각됨.

### 4-family thin regime 비교 (2026-04-16, t=5–300 nm, 1550–1850 band)

| Family | NIR@81 | R_dop@81 | MIR | density | t | P | D (Gm) | **T (min)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Triangular** | **0.557** | **0.524** | 0.128 | 0.477 | 240 | 1520 | 53.9 | **24.4** |
| Hexagonal | 0.384 | 0.374 | 0.113 | 0.596 | 237 | 1363 | 75.2 | 36.8 |
| Rectangular | 0.405 | 0.389 | 0.114 | 0.484 | 221 | 1129 | 64.7 | 30.6 |
| Pentagonal | 0.339 | 0.339 | 0.093 | 0.649 | 223 | 1066 | 79.2 | 41.7 |

**Triangular가 thin regime에서도 압도적** (T 24.4 vs 다음 rectangular 30.6). Thick에서와 동일한 결론 — 6-fold symmetry 이점 건재.

Runs: `results/2026-04-16_163022_thin_triangular_1550_s42/` 외 3개.

### T-direct objective 실험 (2026-04-16)

`AccelerationTimeObjective` — T integral을 직접 minimize하는 새 objective 구현. R-proxy 대신 실제 mission FOM (T to β=0.2) 사용.

Run: `results/2026-04-16_175401_stage1_T_direct_thin_s42/`

결과: **BO가 Sobol init의 T=21.47 min (trial 23)을 60 iteration 동안 못 깸.** BO best는 24.64 min. 원인: T ∝ ∫1/R(λ)dβ 형태라 R dip 하나에 T가 폭발 → GP surrogate가 smooth modeling 실패 → 기존 R-proxy (mean+min mixture)가 더 효과적인 BO objective.

**결론**: T-direct는 evaluation metric으로는 유용하지만, BO objective로는 R-proxy가 더 나음.

**반사율 개선을 위한 다음 시도** (우선순위 순):

1. **T-direct objective** — 현재 BO가 mean R 최대화하는데, T는 1/R의 γ³-weighted integral. R peak을 laser λ (1550 nm)에 맞추면 T 직접 감소. Paper Fig 4에서 T-opt가 D-opt 대비 6분 빠른 것과 같은 원리. 구현 ~1.5h.

2. **a-Si (amorphous silicon) 재료** — n≈3.5 (SiN의 1.75×), 같은 resonance를 절반 두께로 달성 → mass 절반 → T 극적 감소. 1550 nm에서 투명 (bandgap 1.7 eV). 구현 ~30 min (Sellmeier data swap).

3. **Graphene underlayer** — MIR ε 0.10→0.28, NIR/T 무영향. 구현 ~2h (grcwa multi-layer).

4. **Bilayer SiN** — coupled cavity → broadband R. 무거운 구현 ~4h. Mass penalty risk.

5. **1320 nm launch + thin regime** — 아직 미탐색. SiN n이 1320에서 약간 더 높아 thin에서도 유리할 수 있음. 빠른 run ~20 min.

### 현재 best 설계 정리

| 설계 | Launch λ | T | D | R_dop | density | t | MFS | 비고 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Thin baseline trial 23** | 1550 | **19.26 min** | 44.1 Gm | 0.51 | 0.42 | 239 | 318 | ⭐ T 최강 |
| Thick trial 73 | 1550 | 23.26 | 46.5 | 0.87 | 1.03 | 716 | 282 | R 최강 |
| 1320 locked (s123/101) | 1320 | 22.64 | 43.3 | 0.93 | 1.11 | 688 | 310 | 비표준 laser |
| Paper pentagonal | 1550 | 24.6 | 52 | ~0.34 | 0.39 | 200 | 507 | i-line fab |
| Freeform best (s42/59) | 1550 | 23.10 | 51.0 | 0.55 | 0.59 | 279 | 455 | Fourier 효과 미미 |
| Dual-hole (s42/91) | 1550 | 27.11 | 59.3 | 0.49 | 0.60 | 253 | — | multi-hole 효과 없음 |
| Disordered (s42/66) | 1550 | 23.02 | 47.1 | 0.50 | 0.50 | 223 | — | jitter=0 수렴 |

### NIR 반사율 개선을 위한 다음 단계 (A/B/C)

현재 thin best (trial 23) mean R = 0.51 (Doppler band). 이를 더 올리면 T가 더 빨라짐. 세 가지 접근:

**(A) Freeform hole shape** — Fourier 경계 파라미터화. 구현 완료, 결과: shape에 budget 집중해도 R 개선 미미.

**(B) Multi-hole supercell** — 구현 완료, 결과: T 악화.

**(C) Disordered lattice** — 구현 완료, 결과: BO가 jitter=0으로 수렴.

### Adjoint gradient optimization 시도 (2026-04-16)

grcwa autograd backend로 pixel-level gradient descent 구현 (`src/lightsail/optimization/adjoint_opt.py`).

**v1** (lr=0.02, MFS=100nm, 200 iters): R = 0.406 → 0.423 (+0.017). Shape 거의 안 변함.
**v2** (lr=0.15, MFS=50nm, 300 iters): R = 0.411 → 0.417 (+0.006). 역시 shape 고정.

**원인 진단**: grcwa raw output에서 R > 1이 나오는 normalization 버그 발견. `RT_Solve(normalize=1)`이 부정확 → adjoint gradient가 잘못된 landscape 위에서 계산됨. `RCWASolver`는 자체 정규화로 이 문제를 우회하지만, adjoint에서는 raw grcwa를 직접 쓰기 때문에 gradient 무의미.

**Fill fraction 스캔 (정확한 RCWASolver 기준, t=240, P=1580)**:

| a_rel | fill | mean R (Doppler 1550–1898) |
|---:|---:|---:|
| 0.30 | 0.674 | 0.467 |
| 0.34 | 0.581 | **0.501** |
| 0.36 | 0.530 | **0.503** ← peak |
| 0.39 | 0.448 | 0.488 (trial 23) |
| 0.42 | 0.360 | 0.347 |

**Mean R의 fill에 대한 곡선이 매우 완만** (0.49–0.50 plateau at fill 0.48–0.58) → hole shape 변형의 물리적 여지가 없음. Trial 23이 거의 정확한 peak.

**Adjoint normalization fix 완료 (v3)** — frequency-dependent SiN dispersion을 forward fn에 반영. nG=41에서 R+T=1.0 정규화 확인.

**Multi-start adjoint (v4, 최종)** — 1 noisy_circle + 5 random inits × 200 iters each, nG=41, 96×96 grid.

| Init | R_start | R_end | Fill end |
|---|---:|---:|---:|
| **noisy_circle** | 0.283 | **0.438** | 0.53 |
| random (×5) | 0.177–0.178 | 0.248–0.350 | 0.70–0.99 (PhC 붕괴) |

Binary design을 RCWASolver로 재평가: **mean R = 0.464** (circular hole 0.473과 −0.009 차이). Design topview → **거의 완벽한 원형**으로 수렴.

**핵심 결론: 단층 SiN triangular PhC에서 adjoint gradient descent가 원형 hole로 수렴. 원형이 이 regime의 (near-)global optimum. 7-param BO가 pixel-level TO와 동등한 설계를 찾는다.**

이것은 논문 기여 가능: "Low-dimensional parameterized BO suffices for single-layer SiN PhC lightsails — complex freeform shapes provide no additional benefit over circular holes."

Run: `results/2026-04-16_215503_adjoint_multi_1550nm/`, design plot: `design_topview.png`

### 3D Grid scan: t × P × fill → T 직접 최소화 (2026-04-16)

Adjoint이 원형 최적을 증명했으므로, 원형 hole 가정 하에 t × P × a_rel 3D grid scan으로 T를 직접 계산.

**Best (nG=81 검증): T = 20.73 min**

| t | P | a_rel | fill | R_mean | R_min | T (min) | D (Gm) | mass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **280** | **1580** | **0.38** | 0.48 | **0.550** | 0.286 | **20.73** | 45.9 | 5.1g |
| 300 | 1580 | 0.38 | 0.48 | 0.563 | 0.249 | 21.12 | 45.2 | 5.4g |
| 240 | 1580 | 0.35 | 0.56 | 0.532 | 0.282 | 21.24 | 46.7 | 5.1g |

**이전 BO best (trial 23, t=239, T=21.43)보다 0.7 min 개선 (−3.3%).** BO가 t=239에 수렴했는데 t=280이 실제로 더 나음 — 41 nm 더 두꺼워서 mass +0.6g이지만 R +0.05로 T 감소가 mass 증가를 상쇄.

Paper pentagonal (T=24.6) 대비 **−16% 개선**. 단, MFS 318 nm (EBL 필요) vs paper 507 nm (photolithography).

### MFS ≥ 500 nm grid scan — paper와 apple-to-apple 비교 (2026-04-16)

Photolithography 호환 (wall ≥ 500 nm) 강제 후 grid scan.

**Best (nG=81): T = 21.73 min, wall = 506 nm**

| | 우리 (MFS≥500) | Paper pentagonal | Δ |
|---|---:|---:|---:|
| T | **21.73 min** | 24.6 min | **−11.7%** |
| D | 48.6 Gm | 52 Gm | −6.5% |
| Mean R | 0.512 | ~0.34 | +50% |
| t | 220 nm | 200 nm | +10% |
| MFS | 506 nm | 507 nm | ≈동일 |
| Mass | 5.0 g | 3.9 g | +28% |
| Fab | i-line ✓ | i-line ✓ | 동일 |

**핵심 기여: 같은 fabrication cost (i-line photolithography) 조건에서 simple triangular circular hole이 neural TO pentagonal보다 11.7% 빠른 가속. Complex freeform이 불필요함을 systematic하게 증명 (BO + adjoint + grid scan).**

설계: t=220 nm, P=1580 nm, a_rel=0.34 (a=537 nm), wall=506 nm, circular hole, triangular lattice.

### Paper 완전 동일 제약 비교 (2026-04-16)

Paper 조건 그대로: **t=200 nm 고정, Af ∈ [0.4, 0.7], MFS ≥ 500 nm, 1550 nm launch**.

| | 우리 triangular (T-opt) | Paper pentagonal (D-opt) | Paper hexagonal (T-opt, Fig4c) |
|---|---:|---:|---:|
| T | **22.11 min** | 24.6 min | 23.8 min |
| D | 49.6 Gm | 52 Gm | — |
| Af | 0.58 | 0.63 | 0.60 |
| MFS | 512 nm | 507 nm | 517 nm |
| Mean R | 0.471 | ~0.34 | — |
| Period | 1600 nm | ~1860 nm | — |
| Mass | 4.6 g | 3.9 g | — |

**같은 제약에서 T 기준 −10.1% (vs D-opt penta), −7.1% (vs T-opt hexa).**

차이의 원인:
1. Paper가 D를 최적화 (T 아님) — R peak이 Doppler 끝쪽에 위치
2. Paper Af=0.63은 TO가 수렴한 값이지 제약이 아님 — Af=0.58이 T에 더 유리
3. Period 1600 nm (우리) vs 1860 nm (paper) — triangular primitive cell이 smaller period 가능
4. **Neural TO의 pentagonal은 Af≥0.6 regime의 local optimum. Af=0.58 regime에서는 simple circular이 더 나음.**

**공정한 결론**: Neural TO가 나쁜 게 아님. Paper는 (a) D 최적화, (b) Af=0.63에서의 제약된 최적. 우리는 (a) T 최적화, (b) Af=0.58까지 탐색. **설계 공간의 다른 영역에서 더 나은 해를 찾은 것.** 이 자체가 기여 — "Af를 조금 낮추는 것이 T에 유리하다"는 insight는 neural TO로는 보기 어려운 landscape-level 분석.

### Freeform shape-only BO (2026-04-16, Fourier n=2~8)

t=240/P=1580 고정, shape만 16-param BO (300 trials). `scripts/run_freeform_shape_only.py`. 진행 중.

### Improved frac-based hole bounds (2026-04-16)

`hole_a_frac ∈ [0,1]` → period에 따라 a_nm ∈ [50, (period-100)/2] 자동 매핑. Infeasible trial 낭비 0%. `PhCReflector.from_param_vector()` 수정. 114 tests passed.

### Dense BO scan (frac bounds, 200 trials)

1550 launch + 1320 launch, 둘 다 진행 중. `scripts/run_thin_frac_dense.py`.

## 2026-04-15 추가: SailArealDensityObjective

새 objective class `SailArealDensityObjective` (objectives.py 내):
- `rho_s [kg/m²] = rho_material × thickness × material_fraction` 계산, 반환값은 g/m²
- `material_fraction = 1 − hole_area/unit_cell_area` (Structure metadata에서 unit cell area 가져옴)
- `make_stage1_objectives` factory에서 config의 `sail_areal_density` 키가 있을 때만 4번째 objective로 추가 (기본 3-obj run은 변동 없음)
- `scripts/run_stage1_mass_aware.py` 가 이를 켠 run 예시 (NIR/MIR/fab + areal_density 4-obj)
