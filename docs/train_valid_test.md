# 원본 NeuralGCM과 전문가 모델: train → validation → test

이 실험은 공식 deterministic checkpoint를 출발점으로 한 **추가 학습 비교**다.
원 논문의 대규모 사전 학습을 처음부터 재현하는 실험은 아니다. 첫 비교에서는
ODE tendency/acceleration loss를 끄고, 실제 forecast loss를 공통으로 사용한다.
실제 ERA5 본 학습이나 GPU 성능 결과는 아직 없다.

## 비교군과 해석

| 결과 이름 | 구조와 학습 | 목적 |
|---|---|---|
| `baseline_pretrained` | 공식 checkpoint 그대로, 추가 학습 없음 | 배포 원본의 기준 성능 |
| `baseline_finetuned` | 공식 구조/native memory 유지, 동일 train으로 physics 추가 학습 | 추가 자료·학습의 효과 통제 |
| `experts` | memory 제거, 변수별 network와 residual 학습 | 이번 제안 모델 |
| `memoryless` (선택) | memory 제거, 단일 network 학습 | memory 제거와 분해 효과 구분 |

전문가망은 새로 초기화되며 원본은 pretrained physics를 유지한다. 동일 epoch가
동일한 사전 학습량·parameter 수·연산량을 의미하지 않는다. 최종 비교에는
parameter 수, update 수, 수렴 곡선과 학습 시간을 함께 보고한다. 구조 자체의
효과를 주장하려면 `--include-memoryless` 비교와 여러 seed도 필요하다.

## 1. 서버 설치

서버에서 허용된 작업 디렉터리를 사용한다. 기존 공유 ERA5 원본은 이동하거나
수정하지 않고 별도의 전처리 결과를 만든다. 아래 `/path/to/workdir`는 실제
작업 디렉터리로 바꾼다.

```bash
cd /path/to/workdir
git clone https://github.com/nayehyeon61-glitch/neural_gcm_for_expert.git
cd neural_gcm_for_expert
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install 'jax[cuda13]==0.7.2' 'dask[array]'
export CUDA_VISIBLE_DEVICES=0
python -c 'import jax; print(jax.__version__); print(jax.devices())'
```

이미 clone했다면 그 디렉터리에서 `git pull --ff-only` 후 설치한다.
JAX 출력에 GPU가 보여야 한다. CUDA 13 wheel의 Linux driver 요구사항은 580
이상이며 저장소의 JAX 0.7.2 고정 버전을 유지한다. 현재 trainer는 단일 장치
학습이며 GPU 두 장을 자동으로 합쳐 쓰지 않는다. pip CUDA 라이브러리와
서버의 `LD_LIBRARY_PATH` 충돌은 공식 JAX 설치 문서에 따라 확인한다.

처음에는 작은 deterministic 2.8° checkpoint를 사용한다.

```bash
mkdir -p checkpoints
curl -fL https://storage.googleapis.com/neuralgcm/models/v1/deterministic_2_8_deg.pkl \
  -o checkpoints/deterministic_2_8_deg.pkl
```

`run_comparison.py`는 원본 checkpoint의 SHA-256, 코드 commit, 주요 package
버전을 `protocol.json`에 기록한다.

## 2. ERA5 자료 준비와 기간 분할

**현재 loader는 raw 0.25° 일별 파일 폴더를 자동 전처리하지 않는다.**
공식 data-preparation 절차로 checkpoint의 Gaussian 격자·기압면·단위에 맞춘
하나의 NetCDF 또는 Zarr를 먼저 준비한다. 수평 격자 변환은 공식 conservative
regridding 예제를 따른다. SST/해빙의 육지 결측도 공식 예제처럼 명시적으로
처리한다. 누락된 물리 변수를 0이나 대체 변수로 임의 생성하지 않는다.

정확한 요구 변수는 checkpoint의 `input_variables`와 `forcing_variables`이다.
표준 v1 pressure-level checkpoint에서는 다음을 확인한다.

| 종류 | 변수 |
|---|---|
| 대기 | temperature, u_component_of_wind, v_component_of_wind, geopotential, specific_humidity, specific_cloud_ice_water_content, specific_cloud_liquid_water_content |
| 경계 forcing | sea_surface_temperature, sea_ice_cover |
| 대기 차원 | time, level, longitude, latitude |

기존 파일의 `skt`는 SST와 다른 변수다. `u10/v10`, `t2m`도 다층 대기 변수의
대체 입력이 아니다. 일별 데이터는 **00 UTC 순간장인지 일평균인지** 확인한다.
아래 설정은 하루 간격의 순간장이다. 일평균을 공식 순간장 입력과 같은 것으로
해석하지 않는다. 6시간 정답을 보간해서 생성하지 않는다.

첫 추가 학습 비교의 기간 제안:

| 용도 | 기간 | 사용 |
|---|---|---|
| train | 2019–2021 | gradient 갱신, normalization scale 계산 |
| validation | 2022 | epoch·hyperparameter 선택 |
| test | 2023 | 선택 완료 후 최종 비교 |
| 이후 확인용 | 2024–2025 | 첫 test 결과로 재설계했다면 별도 확인 자료로 보존 |

이 기간은 이번 실험 제안이며, 모든 공식 checkpoint가 같은 사전 학습 기간을
쓴다는 뜻은 아니다. 다른 checkpoint로 바꾸면 사전 학습·개발 기간과 test
중복 여부를 다시 확인한다. 자료를 시간순으로 나누며 random split을 쓰지 않는다.
각 split 내부에 초기값과 모든 미래 target이 있어야 하므로 끝부분의 불완전
예측 window는 사용하지 않는다.

**이미 전처리된, 하루 간격 데이터**를 분할하는 예시다. 경로는 실제 준비한
파일로 바꾼다. 이 코드는 regridding이나 변수 추가를 수행하지 않는다.

```bash
python - <<'PY'
from pathlib import Path
import xarray as xr

source = Path('/path/to/prepared/daily_2p8_prepared.zarr')
destination = Path('data/daily')
destination.mkdir(parents=True, exist_ok=True)
ds = xr.open_zarr(source).chunk({'time': 4})
ranges = {'train': ('2019', '2021'), 'valid': ('2022', '2022'),
          'test': ('2023', '2023')}
for name, (start, end) in ranges.items():
    path = destination / f'{name}.nc'
    if path.exists():
        raise FileExistsError(path)
    subset = ds.sel(time=slice(start, end))
    if subset.sizes.get('time', 0) == 0:
        raise ValueError(f'No data in {start}–{end}')
    subset.to_netcdf(path, engine='h5netcdf')
    print(path, subset.sizes)
ds.close()
PY
```

NetCDF 원본은 위 `xr.open_zarr` 대신
`xr.open_dataset(source, chunks={'time': 4})`로 연다. 자료가 hourly라면 먼저
매일 같은 시각의 snapshot을 선택한 후 분할한다. 사전에 변수별 normalization을
valid/test 전체에서 계산하지 않는다. 아래 runner가 train에서만 계산한다.

## 3. 먼저 작은 실행 확인

본 학습 전에 train과 valid에서 각각 7일을 잘라 배선·GPU 메모리를 확인한다.
valid의 기간은 train과 겹치지 않아야 하며 최소 6개 일별 시점이 있어야
5일 forecast를 검증할 수 있다.

```bash
python - <<'PY'
import xarray as xr
from pathlib import Path
for source, output in [('train', 'train_smoke'), ('valid', 'valid_smoke')]:
    destination = Path(f'data/daily/{output}.nc')
    if destination.exists():
        raise FileExistsError(destination)
    with xr.open_dataset(f'data/daily/{source}.nc') as ds:
        ds.isel(time=slice(0, 7)).to_netcdf(destination, engine='h5netcdf')
PY

python scripts/run_comparison.py train \
  --checkpoint checkpoints/deterministic_2_8_deg.pkl \
  --train-data data/daily/train_smoke.nc \
  --validation-data data/daily/valid_smoke.nc \
  --epochs 1 --validation-stride 1 \
  --output runs/smoke-daily-seed0
```

첫 JAX compile은 추가 시간이 걸린다. 이 실행은 실제 다년 학습 성능 검증이
아니다. 7일 scale도 본 실험에 재사용하지 않는다. 정상 종료 후 아래처럼 별도
출력 디렉터리에서 본 학습을 시작한다.

## 4. Train + validation

```bash
python scripts/run_comparison.py train \
  --checkpoint checkpoints/deterministic_2_8_deg.pkl \
  --train-data data/daily/train.nc \
  --validation-data data/daily/valid.nc \
  --interval 24h --train-steps 1 --evaluation-steps 5 \
  --epochs 10 --learning-rate 1e-4 \
  --validation-stride 7 --test-stride 1 --seed 0 \
  --output runs/comparison-daily-seed0
```

이 명령은 세 초기 checkpoint를 무조건 학습시키는 것이 아니라, 원본 복사본을
보존하고 **공식 구조 추가 학습 → 우리 모델 학습**을 순차 실행한다. 기본은
physics만 갱신하고 encoder/decoder는 고정한다. memoryless 비교도 하려면 같은
명령에 `--include-memoryless`를 추가한다.

- 학습: 하루 앞 예측 loss. `--train-steps 2`이면 이틀 rollout으로 늘어난다.
- Validation: 매 epoch, 2022년의 7개 sample마다 초기화하여 1–5일을 예측한다.
  daily 자료의 stride 7은 주 1회다. 모든 초기일을 쓰려면 stride 1로 바꾼다.
- Checkpoint 선택: 초기 시점의 MSE를 합산한 후, **train에서 계산한 scale**로
  정규화하고 변수·기압면·lead에 걸쳐 평균한다. RMSE 숫자를 단위 구분 없이
  평균하거나 첫 시점만으로 선택하지 않는다.
- 두 학습 모델은 같은 train/valid·window 순서 seed·학습률·epoch·scale·forcing을
  사용한다. 추가 residual 정규화는 우리 residual network에만 적용된다.
- `--evaluation-steps 5`는 validation과 test의 공통 horizon이다. test를 본 후
  유리한 lead만 선택하지 않는다.

`baseline_finetuned/validation-selection.json`과 `experts/validation-selection.json`
에서 best epoch와 여러 초기 시점의 score를 확인한다. `training.jsonl`은 update별
loss를 기록한다. 10 epoch는 출발 설정이며 새 전문가망의 수렴을 보장하지 않는다.
epoch나 학습률 조정은 validation으로 결정한다.

runner는 기존 결과 덮어쓰기를 거부한다. 새 설정은 새 `--output`을 쓴다.
현재 trainer checkpoint에는 optimizer state가 저장되지 않아 중단 후 완전한
학습 재개는 지원하지 않는다. 한 번의 train 호출 안에서는 optimizer state가
모든 epoch에 걸쳐 유지된다. 터미널 종료로 중단되지 않도록 서버의 기존 job
scheduler 또는 tmux 세션을 이용한다.

## 5. 선택 완료 후 test

```bash
python scripts/run_comparison.py test \
  --run runs/comparison-daily-seed0 \
  --test-data data/daily/test.nc
```

원본 checkpoint, 추가 학습한 원본 best, 우리 best를 **같은 test 초기일과
1–5일 lead**로 평가한다. train 단계에서는 test 파일을 읽지 않는다. test가
validation 이후인지 검사하고, interval/horizon/stride/forcing 설정은 train
단계의 protocol을 재사용한다. SST·해빙은 모든 모델에서 초기값을 유지한다.
이는 초기 관측 boundary persistence 실험이며 논문 전체 hindcast 재현은 아니다.

결과:

| 경로 (`runs/comparison-daily-seed0/` 아래) | 내용 |
|---|---|
| `protocol.json` | 자료 기간, 설정, source checksum, package/code 버전 |
| `baseline_finetuned/best.pkl`, `experts/best.pkl` | validation에서 선택한 checkpoint |
| `test/baseline_pretrained/metrics.nc` | 원본 그대로의 변수·기압면·lead RMSE |
| `test/baseline_finetuned/metrics.nc` | 추가 학습한 원본 RMSE |
| `test/experts/metrics.nc` | 우리 모델 RMSE |
| `test/experts_vs_baseline_pretrained.csv` | 배포 원본 대비 개선율 |
| `test/experts_vs_baseline_finetuned.csv` | 동일 추가 학습 기준선 대비 개선율 |

CSV는 `100 * (baseline_rmse - candidate_rmse) / baseline_rmse`를 기록한다.
양수이면 우리 RMSE가 더 낮다. 기준 RMSE가 0이면 개선율을 비워 둔다.
초기 시점·lead·단위·평가 조건·유효 표본 수가 맞지 않으면 비교를 중단한다.
전체 forecast 저장은 생략하고 `predictions.nc`에는 비교에 사용한 초기 시점
좌표를 남겨 저장 공간을 줄인다. 표준 요약은 T850(K), Z500(geopotential,
m²/s²), UV700(m/s), Q700(g/kg)이며 단순 평균으로 한 숫자로 합치지 않는다.
ACC가 필요하면 별도로 준비한 climatology와 `ngcm-expert evaluate`를 사용한다.

한 seed의 개선율은 유의성 검정이나 물리 과정 식별의 증거가 아니다. 설계가
정해진 뒤 별도 output으로 seed 0/1/2를 반복하고, test는 모델 선택에 사용하지
않는다. test를 보고 설계를 바꾸었다면 이후 독립 기간에서 다시 확인한다.

## 참고

- [NeuralGCM data preparation](https://neuralgcm.readthedocs.io/en/latest/data_preparation.html)
- [NeuralGCM inference demo](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [JAX 0.7.2 package metadata](https://pypi.org/project/jax/0.7.2/)

이 문서와 runner는 준비된 자료의 경로를 입력받아 실행 절차를 제공한다.
사용자 서버의 원본 ERA5 파일·필수 변수·가용 GPU 메모리는 아직 직접 확인하지
않았다. raw 자료 준비가 끝나기 전에는 본 학습을 시작할 수 없다.
