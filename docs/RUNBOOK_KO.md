# 로컬 GPU·랩실·HPC 실행 안내

## 1. 설치와 코드 검증

Python 3.11 또는 3.12로 시작하는 것을 권장한다. 기존 서비스와 별도 가상환경을 만들고 저장소 최상위에서 실행한다.

```bash
git clone --branch codex/engagement-research https://github.com/sijoon-sung/AI-focus-mangement-agent.git
cd AI-focus-mangement-agent
python -m venv .venv
```

Linux: `source .venv/bin/activate`

PowerShell: `.\.venv\Scripts\Activate.ps1`

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS/드라이버에 맞는 torch/torchvision을 먼저 설치한다. 특정 CUDA wheel을 모든 HPC에 공통 지정하지 않는다.

```bash
python -m pip install -r requirements-research.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -m unittest discover -s tests -v
python -m research.smoke
```

smoke는 무작위 영상 12개로 CPU 1 epoch, 체크포인트, 보정·시험 출력을 검증한다. 수치는 성능 근거가 아니다. 재실행은 `--out runs/smoke2`처럼 새로운 폴더를 사용한다. 본 학습에서는 `--device cuda`를 지정해 GPU 설정 문제를 CPU로 조용히 우회하지 않도록 한다.

## 2. 데이터 준비

[DAiSEE 공식 페이지](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)에서 신청·이용 조건을 확인하고 내려받는다. 예상 구조:

```text
/datasets/DAiSEE/
  Labels/TrainLabels.csv
  Labels/ValidationLabels.csv
  Labels/TestLabels.csv
  DataSet/Train/<subject_id>/<clip_id>/<clip_id>.avi
  DataSet/Validation/<subject_id>/<clip_id>/<clip_id>.avi
  DataSet/Test/<subject_id>/<clip_id>/<clip_id>.avi
```

```bash
python -m research.prepare --root /datasets/DAiSEE --out data/daisee.csv
```

Windows 예: `python -m research.prepare --root "D:/datasets/DAiSEE" --out data/daisee.csv`

`data/daisee.audit.json`의 사람·클립·클래스 수를 확인한다. 배포본 구조나 `ClipID`, `Engagement` 필드가 다르면 adapter를 맞춰야 한다. 작성 환경에서 실제 DAiSEE는 확보하지 않았으므로 실물 데이터 검증은 아직 하지 않았다. 누락·손상 파일은 조용히 무시하지 않는다.

자체 CSV도 `clip_id,subject_id,split,path,label`로 입력할 수 있다. label은 0~3, split은 train/val/test다. 상대 경로는 CSV 위치 기준이다. 같은 사람의 split 중복은 거부한다. 다른 ID가 같은 사람인지 코드가 알 수 없으므로 정확한 ID 관리가 필요하다.

## 3. GPU 확인과 본 학습

```bash
python -m research.train --manifest data/daisee.csv --model tiny_tcn --weighted-loss --epochs 1 --batch-size 4 --device cuda --out runs/gpu_check
```

확인 후 핵심 설정을 실행한다. 각 명령은 기본 30 epochs/16 frames/112 pixels/batch 8이다.

```bash
python -m research.train --manifest data/daisee.csv --model tiny_mean --seed 42 --device cuda --out runs/e0_s42
python -m research.train --manifest data/daisee.csv --model tiny_tcn --seed 42 --device cuda --out runs/e1_s42
python -m research.train --manifest data/daisee.csv --model tiny_tcn --weighted-loss --seed 42 --device cuda --out runs/e2_s42
python -m research.train --manifest data/daisee.csv --model resnet18_tcn --pretrained --weighted-loss --seed 42 --device cuda --out runs/e3_s42
```

seed 43, 44 및 새로운 out 경로로 반복한다. Windows는 workers 0부터, Linux는 `--workers 4`를 고려한다. OOM이면 batch부터 줄인다. frames/size를 바꾸면 비교 설정에도 동일 적용하고 기록한다.

ResNet 사전학습 가중치는 최초 다운로드가 필요하다. 네트워크 차단 compute node에서는 로그인 노드에서 먼저 실행해 공용 캐시에 저장한다.

```bash
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

가중치 다운로드가 안 된다고 `--pretrained`를 빼면 다른 실험이다. 데이터 decoding이 병목이면 로컬 scratch에 복사하고 manifest를 재생성한다. 현재 frame cache·AMP·DDP·중단 재개는 미지원이다. `last.pt`는 마지막 모델 보관용이며 optimizer/RNG 재개 파일이 아니다.

## 4. Slurm

```bash
export MANIFEST=/absolute/path/to/data/daisee.csv
sbatch scripts/train_slurm.sh
```

가상환경을 활성화하고 저장소 최상위에서 제출한다. account/partition/GPU 요청·시간·메모리는 랩 정책에 맞춰 수정한다. 환경이 상속되지 않으면 스크립트 안에서 활성화한다. 4설정×3seeds=12개 작업이 각각 GPU 1개를 요청한다. 동시 실행 제한은 `#SBATCH --array=0-11%2` 등으로 설정한다. 첫 epoch 실측 후 전체 시간과 자원 예산을 결정한다.

## 5. 설정 확정 후 평가

```bash
python -m research.evaluate --manifest data/daisee.csv --checkpoint runs/e2_s42/best.pt --calibrate --device cuda --out runs/e2_s42/eval
```

각 실험·seed에 대해 평가한다. 학습 당시와 clip/subject/split/label 목록이 다르면 거부하며 데이터 경로만 바뀐 경우는 허용한다. 원본 영상 내용 해시는 기록하지 않으므로 배포 버전과 전처리 이력을 별도로 보관한다. test를 보며 설정을 다시 고르지 않는다.

| 파일 | 내용 |
|---|---|
| config.json, split.json | 설정·분할·manifest hash |
| environment.json, packages.txt | Python/PyTorch/GPU/commit·설치 버전 |
| history.jsonl | epoch별 train loss·validation 지표 |
| best.pt, last.pt | 검증 최적/마지막 모델 |
| eval/metrics.json | raw/calibrated 지표·majority/prior·비용·coverage |
| eval/predictions.csv | clip/subject ID·정답·예측·확률 |

train loss는 배치 손실의 샘플 수 가중 평균이며 weighted/unweighted 실행의 절대값을 직접 비교하지 않는다. 처리 시간은 decoding 포함 평균값으로 순수 batch=1 latency가 아니다. temperature는 클래스 순서를 유지해 macro-F1을 바꾸지 않는다. coverage 표로 test에서 배포 threshold를 선택하지 않는다.

## 6. 학습 후 전달

우선 `config.json`, `environment.json`, `packages.txt`, `history.jsonl`, `eval/metrics.json`을 보내면 된다. 사용자 단위 분석에는 가명 ID의 `predictions.csv`를 추가한다. 원본 얼굴 영상은 필요하지 않다. `best.pt`는 추후 추론 통합에 필요하며 Git 일반 커밋에 넣지 않는다.

| 실험 | seed | test macro-F1 | balanced accuracy | 클래스별 recall |
|---|---|---|---|---|
| E0 | 42/43/44 | 미실행 | 미실행 | 미실행 |
| E1 | 42/43/44 | 미실행 | 미실행 | 미실행 |
| E2 | 42/43/44 | 미실행 | 미실행 | 미실행 |
| E3 | 42/43/44 | 미실행 | 미실행 | 미실행 |

모든 seed를 모아 평균±표준편차를 보고한다. 가장 높은 결과만 선택하지 않는다.
