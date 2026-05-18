import os
import json
import urllib.request
from datetime import datetime
import base64
from email.message import EmailMessage

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Google API client 라이브러리
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS") # 사용하지 않음
AW_URL = "http://localhost:5600/api/0"
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_aw_bucket_id():
    try:
        req = urllib.request.Request(f"{AW_URL}/buckets/")
        with urllib.request.urlopen(req) as response:
            buckets = json.loads(response.read().decode())
            for b_id in buckets.keys():
                if b_id.startswith("aw-watcher-window_"):
                    return b_id
    except Exception as e:
        print(f"AW 버킷 연결 에러: {e}")
    return None

def fetch_today_stats():
    bucket_id = get_aw_bucket_id()
    if not bucket_id:
        return {}
        
    print("ActivityWatch에서 오늘 데이터를 가져오는 중...")
    try:
        req = urllib.request.Request(f"{AW_URL}/buckets/{bucket_id}/events?limit=3000")
        with urllib.request.urlopen(req) as response:
            events = json.loads(response.read().decode())
            
        app_stats = {}
        for event in events:
            data = event.get('data', {})
            app_name = data.get('app', 'Unknown')
            title = data.get('title', 'Unknown')
            duration = event.get('duration', 0)
            
            if app_name not in app_stats:
                app_stats[app_name] = {'total': 0, 'titles': {}}
            
            app_stats[app_name]['total'] += duration
            if title not in app_stats[app_name]['titles']:
                app_stats[app_name]['titles'][title] = 0
            app_stats[app_name]['titles'][title] += duration
            
        sorted_stats = {}
        for app, info in sorted(app_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            if info['total'] >= 60: # 1분 이상 사용
                top_titles = sorted(info['titles'].items(), key=lambda x: x[1], reverse=True)[:3]
                sorted_stats[app] = {
                    'total_minutes': round(info['total'] / 60, 1),
                    'top_activities': [t[0] for t in top_titles]
                }
                
        return sorted_stats
        
    except Exception as e:
        print(f"AW 데이터 가져오기 실패: {e}")
        return {}

def generate_report(app_stats):
    print("Gemini API로 리포트를 생성하는 중...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        template = """
당신은 개발자의 일일 생산성을 냉정하고 전문적으로 분석하는 AI 코치입니다.
추측성 발언을 배제하고, 제공된 데이터를 바탕으로 분석하세요.

[앱/프로그램 역할 사전]
- Antigravity.exe: 개발자를 돕는 AI 기반 생산성 데몬 및 에이전트 시스템입니다. (게임이 아닙니다!)
- Code.exe: Visual Studio Code (개발 IDE)
- pycharm64.exe: PyCharm (개발 IDE)
- chrome.exe / msedge.exe: 웹 브라우저 (자료 조사, 개발 문서 참조 또는 딴짓)

다음은 사용자의 오늘 하루 애플리케이션 사용 통계입니다. (단위: 분)
각 앱에서 가장 오래 머문 상위 3개의 창 제목(top_activities)도 포함되어 있습니다.

{app_stats}

이 데이터를 바탕으로 마크다운 형식의 깔끔한 일일 생산성 보고서를 작성해 주세요:
1. 오늘의 핵심 작업 (창 제목을 기반으로 어떤 개발/업무를 했는지 분석)
2. 생산성 앱 vs 비생산적 요소 분석 (어떤 활동이 집중을 방해했는지)
3. 내일의 워크플로우 개선을 위한 3가지 구체적 조언
"""
        prompt = PromptTemplate(template=template, input_variables=["app_stats"])
        chain = prompt | llm | StrOutputParser()
        
        report = chain.invoke({"app_stats": json.dumps(app_stats, indent=2, ensure_ascii=False)})
        return report
    except Exception as e:
        print(f"보고서 생성 실패: {e}")
        return f"보고서 생성에 실패했습니다. 에러: {e}"

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # 현재 디렉토리에서 client_secret JSON 파일 찾기
            client_secret_file = next((f for f in os.listdir('.') if f.startswith('client_secret_') and f.endswith('.json')), None)
            if not client_secret_file:
                raise Exception("client_secret_*.json 파일을 찾을 수 없습니다. OAuth 2.0 클라이언트 정보를 다운로드하여 폴더에 넣어주세요.")
            
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            print("\n[알림] 브라우저 창이 열리면 Google 계정으로 로그인하고 권한을 허용해 주세요.")
            creds = flow.run_local_server(port=0)
            
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('gmail', 'v1', credentials=creds)

def send_email_oauth(report_content):
    try:
        service = get_gmail_service()
        
        # OAuth 인증된 계정의 이메일 주소를 자동으로 가져옵니다.
        profile = service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress')
        
        print(f"작성된 보고서를 Gmail API ({user_email}) 계정으로 전송하는 중...")
        message = EmailMessage()
        message.set_content(report_content)
        
        today_str = datetime.now().strftime("%Y년 %m월 %d일")
        message['Subject'] = f"[AI 데몬] {today_str} 일일 생산성 보고서"
        message['From'] = user_email
        message['To'] = user_email

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        service.users().messages().send(userId="me", body=create_message).execute()
        print("이메일이 성공적으로 전송되었습니다! Gmail 보관함을 확인해주세요.")
        
    except Exception as e:
        print(f"이메일 전송 실패: {e}")
        print("\n--- 보고서 내용 ---")
        print(report_content)

if __name__ == "__main__":
    stats = fetch_today_stats()
    if not stats:
        print("수집된 데이터가 없습니다.")
    else:
        report = generate_report(stats)
        send_email_oauth(report)
