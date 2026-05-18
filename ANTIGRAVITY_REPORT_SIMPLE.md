# Antigravity — AI 생산성 데몬
### 기말 프로젝트 보고서

---

## 목차

1. [필요성 및 목적](#1-필요성-및-목적)
2. [설계 변천 과정](#2-설계-변천-과정)
3. [최종 시스템 개요](#3-최종-시스템-개요)
4. [LangGraph 워크플로우 설계](#4-langgraph-워크플로우-설계)
5. [컴포넌트 설명](#5-컴포넌트-설명)
6. [구현 방법](#6-구현-방법)
7. [기술 스택](#7-기술-스택)

---

## 1. 필요성 및 목적

### 문제 인식

컴퓨터 앞에 4시간 앉아 있었더라도 실제로 과제에 집중한 시간은 1시간도 안 될 수 있다. 문제는 **본인이 그걸 인식하지 못한다**는 점이다.

- 유튜브, 메신저 등으로 흘러가는 시간을 "잠깐"이라고 느끼지만 누적하면 수 시간
- 어떤 시간대에 집중이 잘 되는지, 어떤 작업에서 자주 이탈하는지 패턴을 모름
- 결과적으로 "열심히 했는데 왜 이것밖에 못 했지?"라는 반복

### 해결 방향

**"지금 집중하고 있는가"를 데이터로 측정하고, 하루가 끝나면 분석 리포트를 제공한다.**

구체적으로:
1. 사용자가 태스크를 선언하면 그 태스크 기준으로 집중도를 측정
2. 컴퓨터 사용 데이터(앱, 창 제목)로 지금 하는 활동이 태스크와 관련 있는지 AI가 판단
3. 웹캠으로 자리를 비웠는지 감지
4. 산만하다고 판단되면 알림으로 개입
5. 하루 종료 시 집중 패턴 분석 리포트를 이메일로 발송

---

## 2. 설계 변천 과정

> 보고서에서 가장 중요한 섹션. 처음 설계와 최종 설계 사이의 고민 과정을 기록했다.

### 1차 설계: ActivityWatch 단독 사용

처음엔 ActivityWatch(로컬 앱 사용 추적 도구)의 데이터만으로 집중도를 측정하려 했다.

```
AW 데이터 → LLM 분석 → 토스트 알림
```

**문제점**: ActivityWatch는 어떤 앱을 켜놨는지는 알려주지만, 사용자가 실제로 화면을 보고 있는지는 모른다. 컴퓨터를 켜놓고 자리를 비워도 "집중 중"으로 판단된다.

→ **웹캠 기반 얼굴 감지 추가 결정**

---

### 2차 설계: Vision + ActivityWatch 조합, 매 1분 주기

웹캠으로 얼굴의 EAR(눈 깜빡임 비율)과 고개 방향(Yaw)을 분석해 집중도 점수를 산출하고, 1분마다 LangGraph 에이전트를 실행하는 구조.

**문제점 1 — Vision이 너무 민감함**

EAR 임계값 0.20을 기준으로 했더니, 눈을 깜빡이거나 잠깐 고개를 돌리는 것만으로 점수가 0으로 폭락했다. 아래는 실제 출력 로그다.

```
[Vision] EAR: 0.24, YAW: 0.91 | Score: 10 | Focus: True
[Vision] No face detected! Score: 0
[Vision] No face detected! Score: 0
[Vision] EAR: 0.33, YAW: 1.05 | Score: 20 | Focus: True
[Vision] EAR: 0.29, YAW: 1.08 | Score: 100 | Focus: True  ← 갑자기 100
[Vision] EAR: 0.16, YAW: 1.11 | Score: 0  | Focus: False  ← 다시 0
```

점수가 0과 100 사이를 불규칙하게 오갔고, 실제 집중 상태와 무관한 노이즈가 너무 많았다.

**문제점 2 — 1분 주기 알림이 너무 잦음**

1분마다 에이전트가 실행되면 잦은 개입으로 오히려 집중을 방해한다. 알림 피로도(notification fatigue) 문제.

**문제점 3 — Vision의 역할 재정의 필요**

EAR로 졸음까지 감지하려는 게 과도한 목표였다. 카메라 환경(조명, 거리, 각도)에 따라 신뢰도가 너무 달라진다. Vision의 역할을 **"자리에 있는가(Presence)"** 감지로 단순화하는 것이 현실적이다.

---

### 3차 설계 (최종): 역할 분리 + 10분 주기

위 문제들을 해결하기 위해 다음과 같이 변경했다.

| 항목 | 2차 | 3차 (최종) |
|------|-----|-----------|
| Vision 역할 | EAR + 고개방향으로 집중도 직접 산출 | "자리에 있는가"만 판단 (보조 신호) |
| EAR 임계값 | 고정 0.20 | 시작 시 개인 캘리브레이션 (60프레임) |
| 점수 계산 | 프레임마다 즉시 반영 | 슬라이딩 윈도우 50프레임 평균 |
| 분석 주기 | 1분 | 10분 |
| 알림 조건 | 1회 Distracted면 즉시 | 연속 2회 이상 Distracted 시에만 |
| 집중도 판단 주체 | Vision 점수 | ActivityWatch + LLM 관련도 분류 |
| 태스크 맥락 | 없음 | CLI 태스크 선언으로 제공 |

**핵심 설계 변화**: Vision은 "자리 감지" 보조 신호로 역할을 좁히고, **집중도 판단의 주체는 ActivityWatch + LLM 기반 관련도 분류**로 이전했다.

---

## 3. 최종 시스템 개요

### 전체 흐름

```
[사용자]
   │  CLI: python cli.py start "태스크명"
   ▼
[세션 관리자]  현재 태스크 컨텍스트 유지
   │
   │  10분마다 자동 실행
   ▼
[LangGraph 에이전트]
   ├─ [vision_node]    ← 웹캠: 자리에 있는가?          ┐ 병렬
   ├─ [activity_node]  ← ActivityWatch: 지금 뭘 하나?  ┘ 실행
   │
   ├─ [fusion_node]    ← 두 신호 + 태스크 컨텍스트 합산
   ├─ [analyst_node]   ← Gemini: 상태 평가 (Focused / Distracted / Blocked)
   └─ [coach_node]     ← Distracted/Blocked 시 알림 발송

[daily_report.py]  하루 종료 시 수동 실행
   ├─ 오늘 세션 집계 + 시각화 (matplotlib)
   └─ Gmail API로 본인에게 발송
```

### 측정 로직 요약

```
집중도 = (자리 있음 여부 × 30%) + (태스크 관련 활동 비율 × 70%)

태스크 관련도 판단 (LLM):
  선언: "cycleGAN 구현"
  창 제목 "CycleGAN arXiv paper"  → HIGH  (리서치)
  창 제목 "Stack Overflow GAN"    → HIGH  (문제 해결)
  창 제목 "소드 아트 온라인 BD"   → NONE  (이탈)
  창 제목 "텝스 단어 암기법"      → NONE  (무관한 활동)
```

---

## 4. LangGraph 워크플로우 설계

### 그래프 구조

```
         START
           │
  ┌────────┴────────┐
  │                 │
[vision_node]  [activity_node]   ← 병렬 실행 (Fan-out)
  │                 │
  └────────┬────────┘
           │         ← Fan-in
     [fusion_node]
           │
     [analyst_node]
           │
    ┌──────┴──────┐
  Focused    Distracted
    │         or Blocked
   END             │
             [coach_node]
                   │
                  END
```

**병렬 실행을 선택한 이유**: vision_node(카메라 처리)와 activity_node(HTTP 요청)는 서로 독립적이고 각각 수백 ms의 지연이 생길 수 있다. 순차 실행하면 불필요하게 대기 시간이 늘어나므로 병렬로 처리한다.

**조건부 분기**: analyst_node의 평가 결과가 Focused면 그대로 종료, Distracted 또는 Blocked면 coach_node로 이동해 사용자에게 개입한다.

### 에이전트 상태 (DeveloperState)

```python
class DeveloperState(TypedDict):
    # 세션 컨텍스트
    timestamp: str
    declared_task: str       # CLI로 선언한 태스크 ("cycleGAN 구현")
    session_type: str        # "work" | "break"

    # vision_node 출력
    presence_score: int      # 슬라이딩 윈도우 평균 (0~100)
    is_away: bool            # 연속 미감지 여부

    # activity_node 출력
    app_context: str         # 현재 앱 이름
    window_title: str        # 현재 창 제목
    relevance: str           # LLM 판단: "HIGH" | "MEDIUM" | "LOW" | "NONE"

    # fusion_node 출력
    focus_index: int         # 통합 집중도 (0~100)

    # analyst_node 출력
    evaluation_status: str   # "Focused" | "Distracted" | "Blocked"
    suggested_action: str    # AI 제안 메시지

    # 연속 비집중 카운터 (main.py에서 관리)
    consecutive_distracted: int
```

### 노드별 역할

**vision_node**
- VisionSensor의 슬라이딩 윈도우 평균 점수를 읽어 상태에 기록
- 얼굴이 연속 10프레임 이상 미감지면 `is_away = True`

**activity_node**
- ActivityWatch REST API에서 최근 이벤트 1건 수집
- LLM(Gemini)을 도구로 호출해 창 제목과 선언된 태스크의 관련도 분류
- `relevance`: HIGH / MEDIUM / LOW / NONE

**fusion_node**
- 두 신호를 가중 합산
  ```python
  weight = {"HIGH": 100, "MEDIUM": 70, "LOW": 30, "NONE": 0}
  if session_type == "break":
      focus_index = presence_score         # 휴식 중엔 자리만 확인
  else:
      r = weight[relevance]
      focus_index = presence_score * 0.3 + r * 0.7
  ```
- Vision이 불안정한 환경에서도 Activity 신호가 주도적으로 동작하도록 가중치를 설계했다

**analyst_node**
- Gemini에게 현재 상태(태스크, 앱, 집중도 점수)를 전달
- Focused / Distracted / Blocked 중 하나와 suggested_action을 JSON으로 반환
- Blocked 기준: 에러 로그나 특정 창에 장시간 머무는 경우 ("막힌 상태")

**coach_node**
- `suggested_action`을 plyer로 시스템 토스트 알림 발송
- main.py에서 consecutive_distracted 카운터 관리 → **2회 연속 이상일 때만** 이 노드가 실제로 알림 발송

---

## 5. 컴포넌트 설명

### VisionSensor (vision.py)

웹캠 영상에서 MediaPipe FaceLandmarker로 468개 얼굴 랜드마크를 추출해 두 가지를 측정한다.

**EAR (Eye Aspect Ratio)**
```
EAR = (수직 거리 합) / (2 × 수평 거리)
눈을 뜨면 EAR이 높고, 눈을 감으면 낮아진다
```

**개인 캘리브레이션**: 시작 후 60프레임(약 6초) 동안 본인의 EAR 분포를 측정해 임계값을 자동 설정한다. 고정값 0.20을 쓰면 눈이 작은 사람에게 너무 엄격해지는 문제를 해결했다.

**슬라이딩 윈도우**: 최근 50프레임의 집중/비집중 판정 평균을 `current_score`로 노출한다. 눈 깜빡임이나 1~2프레임 미감지에 점수가 폭락하는 노이즈를 제거했다.

### SessionManager (session.py)

CLI 명령을 받아 현재 세션 상태를 관리하고 세션 종료 시 JSON으로 저장한다.

```
IDLE ──[start]──► WORKING ──[break]──► BREAK ──[resume]──► WORKING
                      └──────────[stop]──────────────────────┘
                                                          → session_log.json 저장
```

세션 로그 예시:
```json
{
  "task": "cycleGAN 구현",
  "start": "2025-01-15T14:00:00",
  "end": "2025-01-15T16:30:00",
  "focus_timeline": [82, 91, 45, 60, 88, 72, 30, 65, 90, 87, 76, 80],
  "distraction_events": [
    {"time": "15:20", "type": "irrelevant_app", "duration_min": 23, "app": "YouTube"},
    {"time": "16:10", "type": "absent", "duration_min": 8}
  ]
}
```

### daily_report.py

하루 종료 후 수동으로 실행하는 독립 스크립트.

1. `session_log.json`에서 오늘 세션 전체 로드
2. ActivityWatch에서 앱별 사용 통계 수집
3. Gemini로 마크다운 리포트 생성
4. matplotlib으로 시각화 2종 생성
   - 시간대별 집중도 꺾은선 그래프
   - 이탈 유형 파이차트
5. Gmail OAuth API로 본인에게 발송

---

## 6. 구현 방법

### 프로젝트 구조

```
Antigravity/
├── main.py              # 진입점: 비전 센서 + LangGraph 루프
├── cli.py               # CLI 명령 처리
│
├── core/
│   ├── __init__.py
│   ├── vision.py        # VisionSensor
│   ├── feedback.py      # LangGraphAgent
│   └── session.py       # SessionManager
│
├── daily_report.py      # 일일 리포트 (독립 실행)
├── face_landmarker.task # MediaPipe 모델
└── .env                 # GOOGLE_API_KEY
```

### 실행 방법

```bash
# 1. 태스크 시작
python cli.py start "cycleGAN 구현"

# 2. 데몬 실행 (별도 터미널)
python main.py

# 3. 휴식 / 재개
python cli.py break
python cli.py resume

# 4. 세션 종료
python cli.py stop

# 5. 하루 리포트
python daily_report.py
```

### LangGraph 그래프 구현 핵심

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(DeveloperState)

# 노드 등록
workflow.add_node("vision_node", self.vision_node)
workflow.add_node("activity_node", self.activity_node)
workflow.add_node("fusion_node", self.fusion_node)
workflow.add_node("analyst_node", self.analyst_node)
workflow.add_node("coach_node", self.coach_node)

# 병렬 실행: START에서 두 노드 동시 시작
workflow.add_edge(START, "vision_node")
workflow.add_edge(START, "activity_node")

# Fan-in: 두 노드가 끝나면 fusion으로
workflow.add_edge("vision_node", "fusion_node")
workflow.add_edge("activity_node", "fusion_node")

workflow.add_edge("fusion_node", "analyst_node")

# 조건부 분기
workflow.add_conditional_edges(
    "analyst_node",
    self.should_intervene,      # Focused → end, else → coach
    {"intervene": "coach_node", "end": END}
)

workflow.add_edge("coach_node", END)
self.app = workflow.compile()
```

---

## 7. 기술 스택

| 역할 | 기술 |
|------|------|
| 에이전트 그래프 | LangGraph (StateGraph, 병렬 노드, 조건부 분기) |
| LLM 체인 / 프롬프트 | LangChain |
| LLM 모델 | Gemini 2.5 Flash |
| 얼굴 감지 | MediaPipe FaceLandmarker |
| 컴퓨터 비전 | OpenCV |
| 활동 추적 | ActivityWatch (로컬 REST API) |
| 이메일 발송 | Gmail API (OAuth 2.0) |
| 시각화 | matplotlib |
| 시스템 알림 | plyer |

### requirements.txt

```
langchain-google-genai
langchain-core
langgraph
mediapipe
opencv-python
python-dotenv
pydantic
plyer
matplotlib
google-auth-oauthlib
google-api-python-client
```

---

*팀: 1인 | 제출: 2025-01*
