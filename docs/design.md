# 결정론적 NeuralGCM의 변수별 learned physics 분해

## 유지하는 부분과 변경하는 부분

공식 NeuralGCM의 입력·출력 변환, 구면 격자와 수직 좌표, primitive-equation dynamics core, 시간 적분기를 사용한다. 변경 대상은 learned physics의 tendency를 만드는 신경망이다. 온도·바람·수분별 network는 전체 현재 상태와 외부 forcing을 보고, 지정된 변수의 tendency를 출력한다. 별도 memory, SDE, diffusion 또는 무작위 noise sampler는 추가하지 않는다.

공식 deterministic checkpoint에도 이전 상태를 쓰는 `MemoryVelocityAndValues`와 `DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder`가 포함되어 있다. 따라서 **원본 checkpoint를 그대로 실행하는 기준선**과 **native memory까지 제거한 이번 실험 모델**을 구분해야 한다. memory feature나 network 구조를 변경한 뒤에는 원본 checkpoint와 수치적으로 동일하다고 주장할 수 없으며, 변경된 부분은 새로 학습해야 한다.

## 변수별 벡터장과 상호작용

격자화된 상태를 \(x=(\zeta,\delta,T',\log p_s,q_v,q_l,q_i)\), 외부 forcing을 \(c\)라고 하자. 실험의 연속 RHS는 다음과 같이 해석한다.

\[
\dot x=F(x,c)=F_{\mathrm{dycore}}(x)+P_T f_T(x,c)+P_{uv}f_{uv}(x,c)+P_q f_q(x,c)+r(x,c).
\]

\(P_i\)는 network 출력을 정해진 상태 변수에 배치하는 연산자이다. 바람 head는 nodal \(u,v\) tendency를 출력하고 공식 div/curl 변환을 거쳐 \(\delta,\zeta\) tendency로 바뀐다. 각 head가 전체 상태를 입력으로 받으므로 온도→바람, 바람→수분 등의 의존성을 유지한다. 분해한 head를 서로 독립적인 ODE로 따로 적분하지 않고, 합쳐진 tendency를 공식 dynamics core와 함께 적분한다.

이것은 **변수 좌표에 따른 벡터장 분해**이다. 자료만으로 온도 head가 복사, 바람 head가 난류를 각각 유일하게 식별한다고 볼 수 없다. 복사·대류·난류 같은 물리 과정은 여러 상태 변수를 동시에 바꾸며, 과정별 정답이나 추가 제약 없이 그 기여를 유일하게 분리할 수 없다. 개별 head가 물리적 제약 다양체의 접공간에 속하거나 서로 직교한다는 주장도 하지 않는다. 제약 다양체의 구조를 쓰려면 별도의 물리적 metric과 보존·수지 제약을 정의해야 한다.

공식 설정에서 learned `log_surface_pressure` tendency는 기본적으로 꺼져 있다. 기압 변화는 dynamics core의 질량 연속식과 다른 변수와의 결합으로 계산된다. 이번 기본 설계에서도 직접 기압 tendency를 만들지 않는다. SST와 sea-ice cover는 입력 forcing이므로 이 모델은 해양 상태를 예측하는 양방향 결합 해양 모델이 아니다.

잔차 \(r\)는 결정론적 미설명 tendency이다. 통계적 noise, 독립적인 확률 과정, 또는 실제 난류를 식별했다고 해석하지 않는다. 잔차의 자유도를 제한하지 않으면 같은 전체 \(F\)를 여러 head와 잔차 조합으로 표현할 수 있다. 현재 trainer는 잔차 network **parameter의 평균 제곱**을 정규화한다. 이것은 출력 tendency의 크기나 물리적 에너지를 직접 제한하는 항이 아니며, 물리적 식별 가능성을 보장하지 않는다.

## 다시 미분한다는 것

각 변수가 다른 변수에 주는 영향을 보려면 합쳐진 1차 벡터장을 궤적 방향으로 미분한다.

\[
\frac{d F_i}{dt}=D_xF_i[F]+D_cF_i[\dot c]
=\sum_j D_xF_i[F_j]+D_cF_i[\dot c].
\]

\(D_xF_i[F_j]\)에 head 간 교차 미분이 들어간다. `total_derivative`는 합쳐진 \(F\)에 JAX JVP를 적용하므로 이 항을 보존한다. 예를 들어 \(\dot T=2u,\dot u=-3T\)이면 자기 변수에 대한 두 대각 미분은 0이지만 \(\ddot T=-6T,\ddot u=-6u\)이다. 각 식을 자기 변수로만 미분하면 이 결합이 사라진다.

`forcing_rate`를 주지 않으면 forcing을 고정한 방향 미분이다. 이를 시간에 따라 변하는 실제 forcing까지 포함한 완전한 가속도라고 부르지 않는다. 일사량처럼 명시적 시간 의존성이 있으면 state의 `sim_time`과 그 tangent 1을 JVP에 포함해야 한다. `sim_time`은 오차 계산에서는 제외하지만, 미분 전에 제거해서는 안 된다.

이 미분값은 보조 학습 신호 또는 진단량이다. 새로운 2차 ODE를 적분하는 방법으로 교체하지 않는다. 공식 `DycoreWithPhysicsCorrector`는 outer physics interval 안에서 신경망 tendency를 고정하고 IMEX dycore substep을 수행한다. 연속 RHS의 JVP와 이 이산 알고리즘의 정확한 2차 미분은 서로 다르므로, 주 학습 loss는 반드시 실제 공식 rollout 결과에 계산한다. RHS를 구성할 때는 dycore의 explicit와 implicit 항을 모두 포함해야 한다.

## Loss와 단위

\[
\mathcal L=\mathcal L_{\mathrm{rollout}}+\lambda_v\mathcal L_{\mathrm{tendency}}
+\lambda_a\mathcal L_{\mathrm{acceleration}}+\lambda_p\frac{\|\theta_{\mathrm{residual}}\|_2^2}{N_{\mathrm{residual}}}.
\]

주 학습 항은 실제 decoded rollout과 정답의 scaled weighted MSE이다. 기본 trainer는 여기에 작은 잔차 parameter L2 항(`residual_parameter_weight=1e-6`)을 더하며, tendency와 acceleration의 기본 가중치는 0이다. 잔차 head가 없으면 parameter 항도 0이다. 보조 tendency와 acceleration 항을 켜면 정답과의 차이를 줄이며, 가속도 자체를 0으로 만드는 penalty는 사용하지 않는다. 모든 상태·정답·미분값은 같은 좌표와 단위를 사용해야 한다. encoded modal state에서는 NeuralGCM의 무차원 시간으로 \(\Delta t\)를 표현하고 변수별 scale도 그 표현에 맞춰 지정한다. decoded Kelvin, m/s 값과 encoded 무차원 값을 직접 비교하면 안 된다.

`losses.residual_size_penalty`는 잔차 **출력 tendency**의 scaled 평균 제곱을 계산하는 별도 utility이다. 현재 기본 trainer에는 이 항이 연결되어 있지 않다. 출력 정규화를 실험하려면 rollout에서 평가한 잔차 tendency를 전달하고 그 항을 학습 objective에 명시적으로 추가해야 한다. parameter L2를 출력 energy penalty와 같은 것으로 해석하지 않는다.

균일 간격의 정답 궤적으로부터 내부 시점의 근사 정답을 얻는다.

\[
\widehat{\dot x}_k=\frac{x_{k+1}-x_{k-1}}{2\Delta t},\qquad
\widehat{\ddot x}_k=\frac{x_{k+1}-2x_k+x_{k-1}}{\Delta t^2}.
\]

차분 정답에는 자료 동화 increment, 샘플링 간격, encoder 오차가 포함될 수 있다. 따라서 장시간 간격 차분을 순간적인 순수 물리 acceleration 정답으로 해석하면 안 된다. 학습 target의 시간 간격을 검사하고, 주 예측 loss에 작은 가중치부터 추가하여 비교한다. 시간 간격·변수 scale은 학습 자료에서 정하며 검증·시험 자료로 조정하지 않는다.

현재 trainer의 acceleration은 forcing을 고정하지만, ERA5 세 시점의 차분에는 실제 SST·sea-ice forcing 변화 효과가 들어갈 수 있다. 따라서 두 값의 matching은 근사적인 보조 목적이다. 변화하는 forcing까지 일치시키려면 `native_acceleration`에 processed forcing의 시간 미분을 추가하거나, 고정 forcing 조건에서 생성한 비교 궤적을 사용해야 한다. native 정답은 학습 시작 시 reference encoder로 계산하여 고정한다. `--scope all`로 encoder까지 갱신하더라도 이 정답을 따라 갱신하지 않으므로, 보조 항은 reference 좌표의 고정된 상태 지점에서 동역학을 제약한다.

`weighted_mse`는 각 변수의 \((\hat x-x)/s_i\) 제곱오차를 유효 위치 weight로 평균한 뒤, 변수 weight로 다시 평균한다. `s_i`는 표준편차 같은 양의 scale이며 분산을 그대로 넣는 인자가 아니다. decoded 격자에서 면적 평균을 쓰려면 실제 격자의 quadrature weight를 전달한다. 위도·경도 배열 순서도 확인한다. 구면 harmonic modal coefficient에는 위도별 면적 weight를 직접 곱하지 않는다. 변환 후 nodal norm을 사용하거나 정의한 spectral norm을 사용해야 한다.

## 코드 API

```python
from neuralgcm_expert.losses import (
    rollout_mse, finite_difference_targets, total_derivative,
    coupled_derivative_loss, residual_size_penalty,
)

# prediction/target: decoded trajectory dict 또는 같은 구조의 pytree.
loss = rollout_mse(prediction, target,
    variable_scales={"temperature": temperature_scale},
    spatial_weights=latitude_quadrature_weights)

# encoded state와 같은 단위의 dt 사용. 결과는 interior 1:-1에 대응.
targets = finite_difference_targets(encoded_truth, dt=nondimensional_dt)

# rhs(state, forcing)는 완전한 dycore + learned physics RHS를 반환.
# 단일 시점 API이며 여러 시점은 jax.vmap으로 처리할 수 있다.
acceleration = total_derivative(rhs, state, forcing, forcing_rate=None)
loss_a = coupled_derivative_loss(rhs, state, target_acceleration,
    forcing=forcing, variable_scales=acceleration_scales)
loss_r = residual_size_penalty(residual_tendency,
    variable_scales=tendency_scales)
# loss_r은 utility 예시이다. 기본 trainer는 이 값 대신 residual parameter L2를 쓴다.
```

`floating_dynamic_fields`는 중첩 dict 및 등록된 JAX pytree의 floating 변수만 선택하고 `sim_time`, 정수 counter, boolean, None을 제외한다. flattened key는 `tracers/specific_humidity` 형태이다. scale/weight에는 같은 문자열, `("tracers", "specific_humidity")` tuple, 또는 leaf 이름을 사용할 수 있다. 전체 경로 사용을 권장한다. `finite_difference_targets`는 이 flattened dict 두 개(tendency, acceleration)를 반환하며, `weighted_mse`는 원래 중첩 구조와 flattened target도 같은 경로로 비교한다.

## 평가와 근거

분해 head의 보조 loss와 최종 forecast 평가를 구분한다. 같은 입력 자료·초기 시점·forcing·해상도·lead time을 사용하여 원본 기준선과 실험 모델의 decoded 예측을 비교한다. 공식 NeuralGCM/WeatherBench2 방식의 변수별·기압면별 RMSE 등 실제 평가 결과를 보고하며, 학습·장기 안정성 실험을 실행하기 전에는 정확도 향상이나 물리 분해 성공을 주장하지 않는다.

- 공식 저장소: https://github.com/neuralgcm/neuralgcm
- 공식 API와 자료 준비: https://neuralgcm.readthedocs.io/en/latest/
- 기본 논문: https://www.nature.com/articles/s41586-024-07744-y
- 공식 learned physics: https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/parameterizations.py
- 공식 dynamics/physics 적분: https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/correctors.py
- 공식 결정론 설정: https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/reference_code/paper_configs/deterministic_2_8_deg.gin
- 공식 평가 reference code: https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/reference_code/metrics.py
