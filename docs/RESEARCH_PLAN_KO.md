# 대학원 지원용 연구 계획

작성일 2026-10-14. 기존 코드 조사 기준 commit: `c2a2b43fb05a14967e7b82b443bf4121438c3b02`. 아래는 연구 설계이며 검증된 실험 성과가 아니다. 문헌은 공식 데이터셋 페이지·논문 초록·출판사 페이지를 중심으로 조사했다. 논문 전체의 완전 재현이나 체계적 문헌고찰을 완료했다는 뜻은 아니다.

## 1. 기존 프로젝트 진단

웹캠·앱 맥락·LLM 제안을 연결한 시스템 구현이 출발점이다. 다만 자체 학습과 정량 검증 근거는 없었다.

- `core/vision.py`: MediaPipe FaceLandmarker에서 EAR 0.20, 고개 방향 거리비 0.4~2.5를 기준으로 점수를 누적한다. 직접 학습한 CNN이나 시선 좌표 추정은 아니다. 얼굴 미검출도 감점해 가림/역광과 이탈을 구분하지 못한다.
- `core/feedback.py`: 최근 ActivityWatch 이벤트 하나를 사용한다. 프롬프트에는 잦은 앱 전환과 장시간 오류 창 체류가 있지만 실제로 그 이력을 계산하는 기능은 확인되지 않았다. API 오류 시 Focused로 처리하므로 향후 unknown 처리가 필요하다.
- `main.py`: 60초마다 현재 점수를 읽는다. 60초 구간 평균이라는 보장은 없다.
- 원래 README는 제목뿐이었고 학습·데이터 분할·평가 파이프라인이 없었다.

연구 동기는 “관찰된 행동 신호를 어떤 조건에서 믿을 수 있는가”다.

## 2. 주제와 가설

**Subject-Independent Lightweight Video Engagement Estimation for Focus-Support Agents**

한국어: **집중 지원 에이전트를 위한 사용자 독립적 경량 영상 참여도 추정**

주 질문: 처음 보는 사용자에 대해 시간 정보를 쓰는 작은 CNN이 특징 평균 방식보다 좋은가? 큰 사전학습 모델 대비 정확도·비용 절충은 어떤가?

보조 질문: 클래스 불균형 처리와 확률 보정이 낮은 참여도 구간의 누락과 과신을 줄이는가?

이는 검증할 가설이다. TCN이나 가중 손실이 항상 우수하다고 가정하지 않는다. 실패한 가설도 반복 실험과 오류 분석이 있으면 연구 경험이 된다.

## 3. 선행연구

| 문헌 | 확인한 내용 | 프로젝트에서 사용할 부분 |
|---|---|---|
| [Gupta et al., DAiSEE, 2016 preprint](https://arxiv.org/abs/1609.01885) | 112명, 9,068개 영상 조각, 네 정서·참여 상태 각각 4단계 라벨 | engagement 4분류 공개 벤치마크. 실제 배포 파일 수는 audit로 확인 |
| [Zhang et al., Appearance-Based Gaze Estimation in the Wild, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.pdf) | 일상 노트북 환경 시선 추정과 사용자·데이터셋 간 일반화 | 시선과 참여도는 별개. MPIIGaze를 집중력 정답으로 사용하지 않음 |
| [Abedi & Khan, ResNet+TCN, 2021](https://arxiv.org/abs/2104.10122) | 프레임 ResNet 특징과 시간 합성곱을 이용한 DAiSEE 분류 | 직접적인 구조 참고. 본 구현은 논문의 완전 재현이 아님 |
| [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | 분류 확률 과신과 temperature scaling | validation으로 온도 선택, test ECE/NLL 비교. 도메인 이동 해결을 보장하지 않음 |
| [Santoni et al., Bagging Ensemble Deep Learning, IEEE Access 2024](https://ieeexplore.ieee.org/document/10591967/) | DAiSEE의 1D CNN·ResNet 앙상블 | 특징·분할 조건을 확인한 뒤 비교해야 함 |
| [Leveraging part-and-sensitive attention network and transformer, 2024](https://www.sciencedirect.com/science/article/pii/S1110016824006835) | DAiSEE/EmotiW-EP의 hybrid transformer | 대형 시간 모델도 이미 선행됨 |
| [Safa, Abedi & Khan, Supervised Contrastive Learning for Ordinal Engagement Measurement, 2025 preprint](https://arxiv.org/abs/2505.20676) | 불균형과 참여도 수준의 순서를 반영한 학습 | ordinal/contrastive objective는 후속 후보, 현재 미구현 |
| [Data augmented hybrid GCN transformer, 2026](https://www.sciencedirect.com/science/article/pii/S1110016826001055) | 랜드마크 기하와 시간 모델의 결합 | 기하·시간 결합 자체를 새로운 발견으로 주장하지 않음 |

문헌마다 평가 단위, 클래스 수, split, 필터링, 사전학습이 다를 수 있다. 논문 accuracy를 그대로 모아 SOTA 순위를 만들지 않는다. 본 실험 전에 핵심 ResNet+TCN 본문의 학습·평가 조건을 대조하고 재현 범위를 기록한다.

## 4. 데이터와 관측 대상

[DAiSEE 공식 배포](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)는 신청서와 연구용 이용 조건을 제공하며 다운로드 크기를 약 15GB로 안내한다. 원본을 GitHub에 재배포하지 않는다. 압축 해제 공간은 별도 확보한다.

참여도 라벨은 관찰 기반 평가다. 화면을 보아도 다른 생각을 할 수 있고, 고개를 돌려도 생각에 몰입할 수 있다. 인지적 집중력·개발 생산성·의학적 상태를 측정했다고 표현하지 않는다.

현재 입력은 전체 프레임 RGB다. 얼굴 crop 및 배경 의존성 비교는 후속 실험이다. 영상 전체에서 16프레임을 균등 추출하므로, 10초 영상 기준 빠른 눈 깜빡임 분석에는 적합하지 않다.

공식 split 구조의 subject 폴더명으로 ID를 읽는다. 배포 README와 대조해 ID의 의미를 확인한다. 같은 사람 또는 clip ID/path가 여러 split에 등장하면 중단한다. 랜덤 프레임 분할은 사용하지 않는다. 다른 이름으로 복제된 동일 영상의 내용 해시 검사는 현재 구현하지 않았다.

라벨이 있는데 영상이 없으면 조용히 제외하지 않고 중단한다. 배포본 누락을 확인하면 제외 목록과 사유를 남기고 adapter를 수정한다. 논문의 9,068개가 모든 배포 버전의 유효 영상 수와 같다고 가정하지 않는다.

## 5. 통제 실험

시작 설정: RGB 16프레임, 112×112, batch 8, AdamW lr 3e-4, weight decay 1e-4, 30 epochs, seeds 42/43/44. 이 값이 최적이라는 주장은 아니다. 설정 변경·탐색은 validation만 사용하고 탐색 횟수를 기록한다.

| ID | 구성 | 목적 |
|---|---|---|
| Majority/Prior | train 최빈 클래스·클래스 비율 | 불균형에 의한 겉보기 정확도 점검 |
| E0 | TinyCNN + temporal mean + CE | 자체 CNN 기준선 |
| E1 | TinyCNN + TCN + CE | E0 대비 시간 모델 기여 |
| E2 | TinyCNN + TCN + weighted CE | E1 대비 불균형 처리 |
| E3 | ImageNet ResNet18 + TCN + weighted CE | 더 큰 사전학습 모델과 실용 비교 |
| 선택 E4 | random-init ResNet18 + TCN + weighted CE | E3 대비 초기화 효과 분리 |

TinyCNN은 채널 16/32/64/128의 Conv2d·GroupNorm·ReLU와 global pooling이다. TCN은 커널 3, dilation 1/2인 두 Conv1d다. clip 전체 분류용으로 causal streaming 모델이 아니며 지연 없는 실시간 모델이라고 표현하지 않는다. E2→E3는 모델 크기와 초기화가 동시에 변하므로 사전학습 효과만의 증거가 아니다.

클래스 가중치는 train에서만 계산한다. validation macro-F1로 best checkpoint를 선택한다. test로 모델·온도·알림 threshold를 고르지 않는다. 가장 좋은 seed 하나가 아닌 세 seed 모두와 평균±표준편차를 보고한다.

주 지표는 macro-F1. balanced accuracy, 클래스별 recall/support, confusion matrix, accuracy, ordinal MAE도 보고한다. macro-F1에는 없는 클래스의 F1=0이 포함되고 balanced accuracy는 존재하는 클래스만 평균하므로 support를 함께 읽는다.

확률 보정은 validation logits에서 T=0.1~10을 선택한다. 모델 선택과 보정에 동일 validation을 사용한 점을 명시하며, 충분한 데이터에서는 별도 보정 subset을 고려한다. argmax가 유지되므로 보정으로 macro-F1 향상을 기대하지 않는다. test confidence별 coverage/error는 기술 통계이며 threshold 최적화 절차가 아니다.

비용은 파라미터, GPU, batch, 입력 크기, 처리 시간을 함께 기록한다. 현재 시간 측정은 decode를 포함한 평균 처리 시간이다. 실시간성 주장 전에 목표 노트북에서 warmup 후 batch=1 모델 지연을 별도 측정한다.

후속 통계로 동일 사용자 단위 paired bootstrap을 고려한다. clip을 독립 표본처럼 재표집하지 않는다. bootstrap 분석 코드는 현재 미구현이다.

## 6. 기존 휴리스틱과 개발 환경으로의 확장

EAR 점수와 DAiSEE 4단계 라벨은 척도가 다르다. 임의의 네 구간으로 잘라 동등한 기준선이라고 하지 않는다. 후속 비교는 동일 clip의 EAR/yaw 특징에서 train/val만 이용해 4분류 매핑을 학습하거나, 동의 기반 자체 데이터의 행동 라벨로 별도 평가한다. 이 기준선은 아직 미구현이다.

DAiSEE만으로 개발 작업에 대한 일반화를 입증할 수 없다. 동의한 소수 참여자의 조명·안경·고개 움직임·자리 비움 조건을 탐색하고 사용자 전체를 held-out으로 둔다. 작은 pilot을 일반적 효과로 해석하지 않는다. 자기보고·관찰 행동·과제 성과를 구분해 수집한다.

향후 에이전트 출력은 클래스 확률, 관측 시각, 신호 유효성, abstain을 포함한다. 미검출을 낮은 참여도와 구분하며 LLM은 설명을 담당한다. 알림 빈도와 사용자 수정 기능을 둔다. 생산성 개선 주장에는 알림 유무를 비교하는 별도 사용자 연구가 필요하다.

## 7. 8주 계획

| 주 | 작업 | 남길 근거 |
|---|---|---|
| 1 | 핵심 논문, 데이터 확보, audit, CPU/GPU 점검 | 분할·환경·실패 목록 |
| 2 | E0/E1 1 seed | 학습 곡선, 오류 분석 |
| 3 | E2/E3와 3 seeds | 통제된 실험 기록 |
| 4 | 설정 동결 후 test·보정 평가 | 실제 지표와 예측 CSV |
| 5 | 배경·조명·가림 분석, 필요시 crop 비교 | 실패 사례와 가설 수정 |
| 6 | 개발 환경 탐색적 pilot | 도메인 이동과 한계 |
| 7 | abstain 포함 서비스 통합 | 데모, 추론 비용, 알림 기록 |
| 8 | 4~6쪽 보고서·발표·포트폴리오 | 실측 결과와 재현 저장소 |

현재 구현은 1~4주 실행 기반이다. 데이터 확보, 실제 학습 성과, 사용자 연구, 서비스 통합은 후속 작업이다. 시간이 부족하면 E0/E2와 3 seeds를 우선한다.
