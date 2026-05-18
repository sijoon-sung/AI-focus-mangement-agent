import asyncio
import threading
import time
from core.vision import VisionSensor
from core.feedback import LangGraphAgent

async def analyze_loop(analyzer, vision_sensor, interval_seconds=60):
    """주기적으로 워크플로우를 분석하여 상태를 모니터링합니다."""
    print(f"[{time.strftime('%X')}] AI 생산성 데몬(LangGraph)이 시작되었습니다. (주기: {interval_seconds}초)")
    while True:
        await asyncio.sleep(interval_seconds)
        
        current_score = vision_sensor.get_score()
        print(f"\n[{time.strftime('%X')}] === [LangGraph 주기적 평가 실행] ===")
        print(f"현재 생체 집중도 점수: {current_score}/100")
        
        try:
            state = await analyzer.run_cycle(current_score)
            eval_status = state.get('evaluation_status', 'N/A')
            print(f"평가 결과: {eval_status}")
            
            if state.get('suggested_action'):
                print(f"AI 조언: {state.get('suggested_action')}")
                
        except Exception as e:
            print(f"분석 중 에러 발생: {e}")
            
        print("==========================================\n")

def start_async_loop(loop, analyzer, vision_sensor):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(analyze_loop(analyzer, vision_sensor))

def main():
    # 1. 생체 비전 센서 (백그라운드 스레드)
    vision_sensor = VisionSensor()
    vision_thread = threading.Thread(target=vision_sensor.run, daemon=True)
    vision_thread.start()
    print("비전 센서가 시작되었습니다.")

    # 2. LangGraph 에이전트 실행 (이벤트 루프)
    analyzer = LangGraphAgent()
    analyzer_loop = asyncio.new_event_loop()
    
    # 3. 분석 루프 스레드 실행
    try:
        start_async_loop(analyzer_loop, analyzer, vision_sensor)
    except KeyboardInterrupt:
        print("\n데몬을 종료합니다...")
        vision_sensor.stop()

if __name__ == "__main__":
    main()
