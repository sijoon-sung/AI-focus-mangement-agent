import urllib.request
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# 시스템 토스트 알림 패키지
try:
    from plyer import notification
except ImportError:
    notification = None

class DeveloperState(TypedDict):
    timestamp: str
    active_file: str            # Activity Watcher 데이터
    app_context: str
    biometric_focus_score: int  # 생체 센서 평균 점수 (0-100)
    anomalies: List[str]        # 잦은 시선 이탈 등 이상 행동 기록
    evaluation_status: str      # LLM 평가 결과 (Focused, Distracted, Blocked)
    suggested_action: str       # 워크플로우 개입 내용

class LLMEvaluation(BaseModel):
    evaluation_status: str = Field(description="One of: Focused, Distracted, Blocked")
    suggested_action: str = Field(description="Action to take, or empty if Focused")

class LangGraphAgent:
    def __init__(self):
        self.aw_url = "http://localhost:5600/api/0"
        self.bucket_id = self._get_window_bucket()
        
        # Gemini API 연동 (환경변수 GOOGLE_API_KEY 필요)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # LangGraph StateGraph 구축
        workflow = StateGraph(DeveloperState)
        
        workflow.add_node("data_collector", self.data_collector_node)
        workflow.add_node("analyzer", self.analyzer_node)
        workflow.add_node("action", self.action_node)
        
        workflow.add_edge(START, "data_collector")
        workflow.add_edge("data_collector", "analyzer")
        
        # 조건부 엣지: Focused면 종료, Distracted/Blocked면 Action 노드로 이동
        workflow.add_conditional_edges(
            "analyzer",
            self.should_take_action,
            {
                "continue": "action",
                "end": END
            }
        )
        
        workflow.add_edge("action", END)
        self.app = workflow.compile()

    def _get_window_bucket(self):
        try:
            req = urllib.request.Request(f"{self.aw_url}/buckets/")
            with urllib.request.urlopen(req) as response:
                buckets = json.loads(response.read().decode())
                for b_id in buckets.keys():
                    if b_id.startswith("aw-watcher-window_"):
                        return b_id
        except Exception:
            pass
        return None

    def data_collector_node(self, state: DeveloperState) -> DeveloperState:
        # AW 데이터 획득 로직
        active_file = "Unknown"
        app_context = "Unknown"
        if not self.bucket_id:
            self.bucket_id = self._get_window_bucket()
            
        if self.bucket_id:
            try:
                # 최근 1개의 윈도우 포커스 이벤트만 추출
                req = urllib.request.Request(f"{self.aw_url}/buckets/{self.bucket_id}/events?limit=1")
                with urllib.request.urlopen(req) as response:
                    events = json.loads(response.read().decode())
                    if events:
                        latest = events[0]
                        app_context = latest.get("data", {}).get("app", "Unknown")
                        active_file = latest.get("data", {}).get("title", "Unknown")
            except Exception as e:
                print(f"AW fetch error: {e}")
                pass
                
        return {
            "active_file": active_file,
            "app_context": app_context,
            "timestamp": datetime.now().isoformat()
        }

    def analyzer_node(self, state: DeveloperState) -> DeveloperState:
        active_file = state.get("active_file", "")
        app_context = state.get("app_context", "")
        focus_score = state.get("biometric_focus_score", 100)
        
        parser = JsonOutputParser(pydantic_object=LLMEvaluation)
        
        template = """
당신은 개발자의 생산성 분석 데몬입니다.
현재 사용자의 상태:
- 활성화된 앱: {app_context}
- 활성화된 창/파일: {active_file}
- 생체 집중도 점수: {focus_score} (0-100)

이 정보를 바탕으로 현재 상태를 'Focused', 'Distracted', 'Blocked' 중 하나로 평가하세요.
만약 Distracted (점수가 50 미만이거나 앱을 너무 자주 바꿈) 거나 
Blocked (특정 에러 로그 창에 너무 오래 머뭄) 라면, 
개발자를 돕기 위한 짧은 팁이나 휴식 권고(suggested_action)를 제공하세요.
Focused 라면 suggested_action은 빈 문자열로 두세요. 한국어로 답변하세요.

{format_instructions}
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["app_context", "active_file", "focus_score"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        
        try:
            res = chain.invoke({
                "app_context": app_context,
                "active_file": active_file,
                "focus_score": focus_score
            })
            eval_status = res.get("evaluation_status", "Focused")
            suggested = res.get("suggested_action", "")
        except Exception as e:
            eval_status = "Focused"
            suggested = f"Error: {e}"
            
        return {
            "evaluation_status": eval_status,
            "suggested_action": suggested
        }

    def should_take_action(self, state: DeveloperState) -> str:
        status = state.get("evaluation_status", "Focused")
        if status in ["Blocked", "Distracted"]:
            return "continue"
        return "end"

    def action_node(self, state: DeveloperState) -> DeveloperState:
        suggested = state.get("suggested_action", "")
        if suggested and notification:
            try:
                # 윈도우 시스템 토스트 알림 띄우기
                notification.notify(
                    title='AI 생산성 데몬',
                    message=suggested,
                    app_name='Productivity Daemon',
                    timeout=5
                )
            except Exception as e:
                print(f"Toast notification failed: {e}")
        return {"suggested_action": suggested}

    async def run_cycle(self, focus_score: int):
        initial_state = {
            "timestamp": datetime.now().isoformat(),
            "active_file": "",
            "app_context": "",
            "biometric_focus_score": focus_score,
            "anomalies": [],
            "evaluation_status": "",
            "suggested_action": ""
        }
        final_state = await self.app.ainvoke(initial_state)
        return final_state
