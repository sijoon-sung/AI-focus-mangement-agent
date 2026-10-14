# Planwise Local

**계획을 실행으로 연결하는 로컬 생산성 에이전트.** ActivityWatch·외부 API·LLM 없이 실행되는 Windows 데스크톱 MVP입니다. Python 표준 라이브러리와 Tkinter만 사용합니다.

상태: **작동하는 규칙 기반 MVP**. 학습된 집중력 모델이나 생산성 향상 효과가 검증된 제품은 아닙니다. 화면·카메라 관측, 소형 신경망, EBM, IDE 실행 완료 연동은 후속 연구 범위입니다.

![Planwise 계획과 실행 화면 — 예제 데이터](docs/assets/planwise.png)

## 바로 실행

Python 3.11 이상(Tkinter 포함)을 설치한 뒤:

```bash
git clone --branch codex/local-plan-agent https://github.com/sijoon-sung/AI-focus-mangement-agent.git
cd AI-focus-mangement-agent
python -m local_agent
```

Windows에서는 `Start-Planwise.cmd`를 더블클릭해도 됩니다. `pip install`은 필요 없습니다. 연구용 PyTorch 의존성을 설치할 필요도 없습니다.

예제 작업으로 먼저 보려면 실제 데이터와 분리해서 실행하세요.

```bash
python -m local_agent --demo --data-dir local_data/demo
```

## 구현 기능

- 완료 조건·다음 행동·예상 시간·우선순위·선행 작업이 있는 계획 관리.
- 작업 세션 시작, 중단 메모, 복귀, 사용자 완료 확인.
- 대기 확인 타이머와 가용 시간 기반 다음 작업 추천.
- 실제 점수 기여·제외 사유·추정 한계를 보여주는 근거 카드.
- 개인 완료 작업의 세션 시간 보정과 명시적으로 확인한 앱 범주 연결 후보.
- 선택적 Windows 앱 범주/유휴 관측, 기간 제한·범주 제외·삭제·집계 내보내기.

**기본 수집은 꺼짐입니다.** 창 제목·URL·키 입력 내용·스크린샷·카메라·클립보드는 수집하지 않습니다. 데이터는 로컬 SQLite에 저장되며 DB 암호화는 없습니다. 동기화 폴더 밖에 저장하려면 `--data-dir`로 지정하세요.

대기 타이머는 다시 확인할 시각을 알려줄 뿐 실행 완료를 감지하지 않습니다. 창을 닫으면 세션과 수집이 끝납니다. 트레이 상주·백그라운드 서비스·OS 토스트는 후속 기능입니다.

## 문서

- [프로젝트 계획서](docs/PROJECT_PLAN_KO.md)
- [사용 안내](docs/LOCAL_AGENT_GUIDE_KO.md)
- [검증 기록](docs/LOCAL_AGENT_VALIDATION.md)
- [이전 CNN 연구 트랙](docs/RUNBOOK_KO.md): 선택적 시각 모듈 실험 기반

## 테스트

```bash
python -m unittest discover -s tests -p test_local_agent.py -v
python -m local_agent --demo --ui-smoke --data-dir runs/ui-smoke
```

`local_agent/store.py`는 저장, `planner.py`는 설명 가능한 규칙, `sensors.py`는 Windows 수집, `app.py`는 UI를 담당합니다.

기존 `main.py`와 `core/`는 이전 MediaPipe·ActivityWatch·Gemini 프로토타입입니다. 새 앱은 `python -m local_agent`로 실행하며 이전 서비스를 호출하지 않습니다.
