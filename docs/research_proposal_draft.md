# Multi-Objective Bayesian Optimization of Photonic Crystal Lightsails for Interstellar Propulsion

## 1. 연구 배경 및 목적

Breakthrough Starshot Initiative는 gram-scale lightsail을 laser 추진으로 0.2c까지 가속하여 Alpha Centauri에 20년 내 도달하는 것을 목표로 한다. 이를 위해 lightsail은 (1) NIR laser 파장에서 높은 반사율, (2) MIR에서 높은 방사율 (열 관리), (3) 극도로 낮은 면적 질량밀도를 동시에 만족해야 한다.

최근 Norder et al. (2025, *Nature Communications*)은 neural topology optimization으로 pentagonal lattice PhC 구조를 발견하여 가속 시간 24.6분 (D=52 Gm)을 달성하고, i-line photolithography 호환 (MFS > 500 nm)으로 제조 비용을 9000배 절감했다. 그러나 평균 반사율이 ~0.34로 낮아 가속 효율에 한계가 있으며, MIR 방사율은 고려되지 않았다.

본 연구는 **multi-objective Bayesian optimization (MOBO)**과 **RCWA 전자기 시뮬레이션**을 결합하여, 반사율 · 방사율 · 질량 · 제조성을 동시에 최적화하는 lightsail PhC 설계 프레임워크를 개발한다.

## 2. 연구 방법

### 2.1 최적화 프레임워크

- **BO engine**: BoTorch qLogNEHVI (4-objective MOBO)
- **EM solver**: grcwa 기반 RCWA + SiN 분산 데이터 (Luke NIR + Kischkat MIR)
- **Objectives**: NIR reflectance (↑), MIR emissivity (↑), sail areal density (↓), fabrication penalty (↓)
- **Parameterization**: 7–11 continuous params (thickness, period, hole shape, Fourier harmonics 등)

### 2.2 설계 공간 탐색

| 접근 | 파라미터 수 | 물리 메커니즘 |
|---|---:|---|
| 단일 ellipse hole (baseline) | 7 | Guided-mode resonance |
| Freeform hole shape (Fourier) | 11 | Multi-peak resonance |
| Dual-hole supercell | 9 | Coupled resonances |
| Disordered lattice | 8 | Anderson scattering |
| 4 lattice families | 7–8 | Symmetry-driven bandstructure |

### 2.3 성능 평가

Relativistic lightsail equation of motion을 직접 적분하여 acceleration time *T*와 distance *D*를 계산:

$$T = \frac{m_t c^2}{2IA}\int_0^{\beta_f}\frac{\gamma(\beta)^3}{R[\lambda(\beta)]}\frac{1+\beta}{1-\beta}d\beta$$

## 3. 현재까지의 주요 결과

### 3.1 Thin-film regime에서 paper 대비 22% 빠른 가속 달성

| 지표 | 본 연구 (triangular best) | Norder et al. pentagonal |
|---|---:|---:|
| **T to β=0.2** | **19.26 min** | 24.6 min |
| **D** | **44.1 Gm** | 52 Gm |
| Launch wavelength | 1550 nm | 1550 nm |
| Mean R (Doppler band) | **0.51** | ~0.34 |
| Thickness | 239 nm | 200 nm |
| Sail mass (10 m²) | 4.2 g | 3.9 g |
| Areal density | 0.42 g/m² | 0.39 g/m² |
| MFS | 318 nm | 507 nm |

단일 guided-mode resonance 기반 triangular PhC가, 복잡한 pentagonal multi-peak 구조보다 mass/R ratio에서 우위를 점함.

### 3.2 구조 변형의 체계적 비교

Freeform hole shape, dual-hole supercell, disordered lattice 3가지 추가 자유도를 탐색한 결과, 모두 baseline 대비 T 개선 실패. BO가 세 가지 자유도를 기각 → **단일 guided-mode resonance가 thin SiN 단층에서 최적 메커니즘**임을 확인.

### 3.3 Graphene underlayer로 MIR 해결 가능성 확인

SiN PhC 뒷면에 graphene을 배치하면:
- MIR emissivity: 0.10 → **0.28** (10-layer) ~ **0.43** (20-layer)
- NIR reflectance: **변화 없음** (graphene이 투과광만 흡수)
- 추가 mass: 0.08g (10-layer, sail 4.2g의 1.8%)
- 가속 시간 T: **변화 없음**

## 4. 향후 연구 계획

### Phase 1: 가속 시간 추가 단축 (1–2개월)

| 과제 | 기대 효과 | 소요 |
|---|---|---|
| **T-direct objective** — T를 직접 minimize하는 BO objective 구현 | R peak을 launch λ에 정렬 → T 16–17 min | 1주 |
| **a-Si 재료 탐색** — n≈3.5 (SiN의 1.75×), 절반 두께로 동일 resonance | mass 절반 → T 10–15 min 기대 | 2주 |
| **4 lattice family thin regime 비교** (진행 중) | Global optimum family 확정 | 1주 |

### Phase 2: 열 관리 + 제조성 (2–3개월)

| 과제 | 기대 효과 |
|---|---|
| **Graphene underlayer RCWA 구현** — grcwa multi-layer stack | MIR ε 0.28+ 달성, T 불변 |
| **MFS ≥ 500 nm constraint run** — i-line photolithography 호환 | Paper와 동등 fab cost에서 경쟁 |
| **Bilayer SiN (coupled cavity)** — broadband R | Doppler 전 구간 R > 0.5 |

### Phase 3: 실험 검증 (3–6개월)

| 과제 | 비고 |
|---|---|
| Lab SiN ellipsometry + FTIR | 실제 PECVD/LPCVD film의 n, k 데이터 |
| EBL + ICP-RIE 시제품 제작 | Trial 23 설계 (t=239, P=1581, triangular) |
| NIR reflectance 측정 (tunable laser 1530–1620 nm) | Simulation vs experiment 검증 |
| Graphene transfer + MIR emissivity 측정 | FTIR 기반 |

## 5. 기대 성과 및 차별성

1. **기존 대비 20–40% 빠른 가속 시간**: 동일 mission parameters (10 GW/m², 10 m², 1550 nm)에서 Norder et al. 대비 T 22% 개선 달성, 추가 최적화 (T-direct + a-Si)로 40%+ 개선 기대.

2. **Multi-objective 동시 최적화**: 기존 연구가 D 또는 T 단일 FOM만 최적화한 반면, 본 연구는 NIR R / MIR ε / mass / fab penalty 4개 objective의 Pareto front를 탐색하여 mission 요구사항에 따른 최적 설계 선택 가능.

3. **MIR 열 관리 솔루션**: Graphene underlayer가 NIR 무영향으로 MIR ε를 3–4배 증가시킴을 이론적으로 확인. 실험 검증 시 lightsail 열 설계의 새 경로 제시.

4. **Reproducible open-source framework**: BoTorch + grcwa 기반 end-to-end 최적화 파이프라인을 공개하여 lightsail 커뮤니티의 설계 민주화에 기여.

## 6. 참고 문헌

- Norder, L. et al. "Pentagonal photonic crystal mirrors: scalable lightsails with enhanced acceleration via neural topology optimization." *Nature Communications* **16**, 2753 (2025).
- Atwater, H. A. et al. "Materials challenges for the Starshot lightsail." *Nature Materials* **17**, 861–867 (2018).
- Luke, K. et al. "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator." *Optics Letters* **40**, 4823 (2015).
- Kischkat, J. et al. "Mid-infrared optical properties of thin films of aluminum oxide, titanium dioxide, silicon dioxide, aluminum nitride, and silicon nitride." *Applied Optics* **51**, 6789 (2012).
