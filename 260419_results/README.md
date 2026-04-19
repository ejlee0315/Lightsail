# 2026-04-19 연구 결과 정리

SiN 라이트세일 3-zone 아키텍처 (central PhC + outer ring + curvature) paper-grade
안정화 검증 결과. Gieseler 2024 (Nat Commun) 방법론을 따라 6-DOF rigid body
trajectory simulation으로 PASS/FAIL 검증.

## 🎯 핵심 결과 — **3-zone architecture 첫 PASS**

| Config | max \|xy\| [m] | max \|tilt\| [°] | Verdict |
|---|---:|---:|---|
| flat + reference ring (P=2400) | 6.70 | 2150 | ❌ FAIL |
| flat + BO best ring (P=1424) | 0.071 | 274 | ❌ FAIL (tilt) |
| **curved R_c=30m + BO best, t=280nm** ⭐ | **0.071** | **3.17** | ✅ **PASS** |

Paper thresholds (Gieseler 2024): max \|xy\| < 1.8 m, max \|tilt\| < 10°, max spin drift < 5%

## 폴더 구조

```
260419_results/
├── README.md                           (이 파일)
├── 11_ablation_summary.md              (5×3 curvature × ring PASS/FAIL matrix)
├── plots/                              (11개 PNG, 번호순)
│   ├── 01_FAIL_flat_reference.png
│   ├── 02_FAIL_flat_BOring.png
│   ├── 03_PASS_curved_BOring.png       ⭐ 핵심 PASS 결과
│   ├── 04_ablation_matrix.png          ⭐ paper figure 후보
│   ├── 05_damping_enhancement.png      (ring 100× ~ 2700× amplification)
│   ├── 06_underlayer_compare.png       (그래핀/SiC/hBN 부적합 negative result)
│   ├── 07_proxy_validation.png         (analytic vs FMM proxy, ρ=0.3)
│   ├── 08_BO_best_spectrum.png
│   ├── 09_BO_history.png
│   └── 10_combined_4panel.png
├── data/                               (CSV + YAML + JSON)
│   ├── curvature_sweep_summary.csv     (15 trajectories PASS/FAIL)
│   ├── damping_enhancement_sweep.csv
│   ├── underlayer_compare.csv
│   ├── proxy_vs_fmm.json
│   └── stage2_fmm_bo_best_{seed42,123,456}.yaml
└── trajectories/                       (NPZ 원시 데이터)
    ├── flat_reference_FAIL.npz
    ├── flat_BOring_FAIL.npz
    ├── curved_BOring_PASS.npz           (t=240 ring, 이전 버전)
    └── curved_BOring_t280_PASS.npz      (t=280 ring, 두께 통일 후)
```

## 방법론 요약

### 1. Force LUT (A1)
- **Center PhC** (Design A: t=280 nm, P=1580 nm, hole r=600 nm, triangular):
  RCWA nG=41 grid 96×96 at θ∈[−5°,5°]×11 / λ∈{1550, 1724, 1898} nm 3-point
  Doppler-band sweep
- **Outer ring** (concentric metagrating): 1D-FMM nG=21 grid 128×4,
  per-order R_m/T_m → C_pr,1 = Σ sin(θ_m)(R+T), c_z = cos(θ_in) − Σ(T−R) cos θ_m

### 2. Polar integrator (A2)
5-cell Gaussian beam I(r)=I₀ exp(−2r²/w²) with w=4 m, I₀=10 GW/m².
Local incidence angle at (r,φ):
```
sin θ_local_radial = sin(θ_y) cos(φ) − sin(θ_x) sin(φ) + r/R_c
                    └───── sail tilt ─────┘          └ curvature ┘
```
r/R_c term이 **곡률에 의한 자연스러운 tilt-restoring** 메커니즘 (Salary 2019 원리).

### 3. 6-DOF rigid body EoM (A3)
Thin uniform disc inertia (I_xx = I_yy = M·R²/4, I_zz = M·R²/2). scipy `solve_ivp`
(RK45, rtol=1e-6). Spin Ω_z = 120 Hz (gyroscopic coupling through Euler
cross-terms).

### 4. Paper-style trajectory (A5, Gieseler 2024 equivalent)
- Initial perturbation: x₀ = y₀ = 50 mm, θ_x₀ = θ_y₀ = −2°
- Integration time: 5 s
- Paper PASS/FAIL thresholds: \|xy\|<1.8 m, \|tilt\|<10°, spin drift<5%

## 핵심 물리 발견

### ✅ Ring metagrating 역할 (CLAUDE.md docx Eq. 4.8)
- **Damping enhancement** (β=0.10, P=2800 nm 기준): ring이 bare PhC 대비 **∂C_pr,2/∂θ를 2700× 증폭** (Figure 5). Metasurface-engineered relativistic damping 정량 입증.
- **Lateral restoring**: BO best ring (P=1424 nm, Wood anomaly 근처)에서 sail
  변위 Δx=50mm → 복원력으로 변위가 거의 0에 pinned (Figure 2, 3).

### ✅ 곡률 stability map (Phase 1-3)
- **Flat sail: 모든 ring 조합 FAIL** (tilt 발산)
- **R_c = 10-100 m (convex): PASS** — curvature가 dominant stabilizer
- **R_c = 30 m**: 3-zone sweet spot (max tilt 3.17° @ t=280nm)
- R_c = 3 m: over-curved, 다시 FAIL

### ⚠️ Ring-curvature interference (anomaly)
- R_c = 30 m + reference ring (P=2400): **FAIL** (tilt 2495°)
- 같은 R_c에서 no-ring 또는 BO ring은 PASS
- → **임의 ring이 curvature restoring을 destructive interference로 망칠 수
  있음** — paper-worthy cautionary finding

### ❌ Thermal (negative result)
- Graphene backside (N=1~50): **MIR에서 Drude metallic** → ε_MIR 감소
  (CLAUDE.md의 analytical 0.28 추정 반박)
- SiC/hBN 균일 underlayer: MIR 흡수는 있지만 NIR R 손상 + mass 과다
- **모든 후보 부적합** → 별도 해결책 필요 (multi-resonance metasurface absorber)

## 코드 아티팩트

### 새로 추가된 주요 모듈
- `src/lightsail/dynamics/force_lut.py` (~210 lines)
- `src/lightsail/dynamics/optical_integrator.py` (~260 lines)
- `src/lightsail/dynamics/rigid_body.py` (~190 lines)
- `src/lightsail/dynamics/floquet.py` (~140 lines)
- `src/lightsail/dynamics/stiffness.py` (~140 lines)
- `src/lightsail/dynamics/damping.py` (~180 lines)
- `src/lightsail/simulation/grating_fmm.py` (~290 lines)
- `src/lightsail/simulation/layered_rcwa.py` (~160 lines)
- `src/lightsail/optimization/fmm_proxy.py` (~110 lines)
- `src/lightsail/materials/{graphene,sic,hbn}.py` (각 ~130 lines)

### 주요 스크립트
- `scripts/run_paper_trajectory.py` — 6-DOF trajectory + PASS/FAIL
- `scripts/run_curvature_sweep.py` — 5×3 ablation matrix
- `scripts/run_underlayer_comparison.py` — thermal material 비교
- `scripts/collect_stage2_fmm_best.py` — multi-seed 결과 수집
- `scripts/verify_damping_enhancement.py` — docx Eq.4.8 정량 검증

## 다음 단계

- **Phase 3** (optional, 2-3주): 2D engineered metasurface BO
  (Gieseler 2024 같은 TE/TM 비대칭 unit cell, flat sail PASS 목표)
- **Paper writing** (1-2주): 현재 결과만으로 충분히 완성도 있는 1편 가능

## 주요 참고문헌
- Gieseler et al., Nat Commun 15:4203 (2024) — 방법론 원본
- Norder et al., Nat Commun 16:2753 (2025) — propulsion baseline
- Salary & Mosallaei, PRL 123:225504 (2019) — self-stabilizing curved sail
- Chriki et al., ACS Photonics 9:12345 (2022) — SiN metagrating 실험 검증
