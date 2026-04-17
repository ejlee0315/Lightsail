# 다음 10시간 실험 계획 (2026-04-17 09:00 이후)

**현재 상황 기준**: 밤새 7개 실험 완료 예정 (1320 re-run, MFS≥500 multi-seed, fab tolerance, 1550 validation).

## 선택지 A — 재료 탐색 (a-Si) ⭐ 추천

**왜**: n=3.5 (SiN의 1.75×)로 **절반 두께에서 동일 resonance** → mass 절반 → T 극적 감소 기대. 1550 nm에서 a-Si는 투명 (bandgap 1.7 eV = 730 nm cutoff).

**할 일**:
1. a-Si Sellmeier coefficients를 materials 모듈에 추가 (Pierce & Spicer 1972 데이터)
2. AlSiDispersion class 구현 (SiNDispersion과 동일 인터페이스)
3. 1550 thin BO 재실행 (seed 42/123/456 × 3, 각 60 min)
4. Grid scan t × P × fill for a-Si (2h, nG=81)
5. Paper 조건 대비 비교 + paper-style D vs MFS plot 재현

**예상 결과**: T = 10–15 min (paper 24.6 대비 **−40~60%**). Game-changing.

**총 시간**: **~8시간** (3 multi-seed + grid scan + 분석)

**리스크**: a-Si 제작 난이도 증가 (compressive stress → membrane buckling). 논문에서 "옵션 탐색" 수준으로 제시 가능.

---

## 선택지 B — 열 관리 완성 (graphene multi-layer RCWA)

**왜**: 현재 graphene 계산은 analytical만 (ε ≈ 0.28 for 10-layer). RCWA multi-layer로 정확히 검증 필요. Paper가 다루지 않은 **차별화 포인트**.

**할 일**:
1. grcwa의 Add_LayerGrid + Add_LayerUniform으로 multi-layer stack 구현
2. Graphene conductivity model (Drude + interband) 추가
3. 1550 best 설계 위에 graphene 1/5/10/20/50 layer 적층 → R(λ), T(λ), ε(λ) 계산
4. Thermal steady-state 계산: T_sail = (P_laser × (1-R-T) / (A × σ × ε × 4))^(1/4)
5. Mass + R + ε + T_sail 4-objective Pareto 도출

**예상 결과**: MIR ε > 0.4 with R 무변, thermal survival margin 정량화.

**총 시간**: **~6–8시간** (구현 3h + 실행 2h + 분석 2h)

**리스크**: Graphene Drude model 계수 (scattering rate, E_F) 가정 필요. 논문에서 "assuming typical CVD graphene with E_F = 0.3 eV" 같은 caveat 필요.

---

## 선택지 C — Bilayer SiN + Si (Brewer et al. 2024 확장)

**왜**: Brewer et al. 2024 Nano Letters에서 R>91% broadband 실험 검증 완료된 설계. 우리 4-objective framework로 재최적화하면 T를 더 줄일 수 있을지 시험.

**할 일**:
1. 2-layer stack 지원 추가 (grcwa native)
2. SiN PhC + Si uniform membrane (또는 둘 다 patterned) geometry 클래스
3. 6-param BO (t_SiN, t_Si, P, a_rel, ...)
4. Paper bilayer 대비 T 비교

**예상 결과**: Broadband R → T 감소 but mass 증가. Net 이득 불확실.

**총 시간**: **~10시간** (코딩 4h + 실행 4h + 분석 2h)

**리스크**: 구현 복잡도 높음. Brewer 데이터 (R>91%)와 정량 비교 어려울 수 있음.

---

## 선택지 D — Deep characterization (기존 결과 강화)

**왜**: 논문 argument를 **statistically bulletproof**하게 만듦. 새 insight는 적지만 reviewer-proof.

**할 일**:
1. Seed 6개로 ensemble (seed 42/123/456/789/1000/1337)
2. nG=81로 전체 production 검증 (속도 4×)
3. Pareto front 정량화 (T vs D vs mass vs MFS)
4. Uncertainty quantification: mean ± std for all best T claims

**예상 결과**: "T = 20.7 ± 0.3 min across 6 seeds" 같은 robust claim.

**총 시간**: **~8시간**

**리스크**: 낮음. 새 결과 없음.

---

## 추천 조합

**A + B 병렬** (각 6–8h): a-Si와 graphene은 독립적이라 동시 진행 가능.
- a-Si → **반사율 근본 개선** (T 40% 추가 감소 potential)
- Graphene → **열 관리 완성** (paper가 못 다룬 영역 장악)

둘 다 끝나면 논문이 매우 강해짐:
- "우리 framework는 재료 무관 (SiN, a-Si 모두)"
- "우리 설계는 thermal survival까지 보장"

## 돌릴 준비

A/B 중 원하시는 걸 말씀해주시면 9:00 이후 바로 시작하겠습니다. 아니면 둘 다 동시 시작 (~10시간).

단, a-Si dispersion data를 먼저 구해야 함 (Pierce & Spicer 1972 Sellmeier, 또는 Palik 1985 tabulated). 제가 literature 기반으로 approximate 데이터 준비할 수 있습니다.
