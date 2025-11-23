# -*- coding: utf-8 -*-
"""
Colab에서 FastAPI 서버를 실행하고 ngrok으로 터널링하는 스크립트

사용 방법:
1. Colab에서 이 파일을 업로드하거나 내용을 복사
2. 필요한 패키지 설치
3. 이 스크립트 실행
4. 생성된 ngrok URL을 프론트엔드에 설정
"""

import os
import sys
import subprocess
import threading
import time
import requests
from pathlib import Path

# ===== 설정 =====
# ngrok 토큰 (https://ngrok.com에서 무료 가입 후 발급)
NGROK_TOKEN = ""  # 여기에 ngrok 토큰 입력 (예: "2abc123def456ghi789jkl012mno345pq")

# 서버 포트
SERVER_PORT = 8000

# ===== 패키지 설치 =====
print("📦 필요한 패키지 설치 중...")
packages = [
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pyngrok",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "torch",
    "scikit-learn",
    "numpy",
    "requests"
]

for package in packages:
    try:
        __import__(package.split("[")[0])
        print(f"  ✅ {package} 이미 설치됨")
    except ImportError:
        print(f"  📥 {package} 설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)

# ===== ngrok 설정 =====
if NGROK_TOKEN:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    print("✅ ngrok 토큰 설정 완료")
    use_ngrok = True
else:
    print("⚠️ ngrok 토큰이 설정되지 않았습니다.")
    print("💡 ngrok 토큰 발급 방법:")
    print("   1. https://ngrok.com 접속")
    print("   2. 무료 회원가입")
    print("   3. Dashboard > Your Authtoken 복사")
    print("   4. 이 스크립트의 NGROK_TOKEN 변수에 붙여넣기")
    use_ngrok = False

# ===== 작업 디렉토리 설정 =====
# Colab에서 실행 시 /content 디렉토리 사용
if os.path.exists("/content"):
    base_dir = Path("/content")
    # 프로젝트가 클론되어 있는지 확인
    if (base_dir / "siot-OSS").exists():
        os.chdir(base_dir / "siot-OSS" / "backend")
        sys.path.insert(0, str(base_dir / "siot-OSS" / "backend"))
        print(f"✅ 작업 디렉토리: {os.getcwd()}")
    else:
        print("⚠️ siot-OSS 프로젝트를 찾을 수 없습니다.")
        print("💡 GitHub에서 클론하세요:")
        print("   !git clone https://github.com/your-username/siot-OSS.git")
        print("   %cd siot-OSS/backend")
else:
    # 로컬 실행 시
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    sys.path.insert(0, str(backend_dir))
    print(f"✅ 작업 디렉토리: {os.getcwd()}")

# ===== 서버 실행 함수 =====
def run_server():
    """FastAPI 서버 실행"""
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'app.main:app',
            '--host', '0.0.0.0',
            '--port', str(SERVER_PORT),
            '--reload'  # 개발 모드 (Colab에서는 제거 가능)
        ])
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")

# ===== 서버 시작 =====
print("\n" + "="*80)
print("🚀 FastAPI 서버 시작 중...")
print("="*80)

# 백그라운드에서 서버 시작
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# 서버 시작 대기
print("⏳ 서버 시작 대기 중... (모델 로딩 포함 약 2-5분 소요)")
server_ready = False
max_wait = 60  # 최대 10분 대기

for i in range(max_wait):
    try:
        response = requests.get(f'http://localhost:{SERVER_PORT}/health', timeout=5)
        if response.status_code == 200:
            print(f"\n✅ 서버 준비 완료! ({i+1}번째 시도, 약 {i*10}초)")
            server_ready = True
            break
    except requests.exceptions.ConnectionError:
        pass  # 아직 서버가 시작 중
    except Exception as e:
        if i % 6 == 0:  # 1분마다 출력
            print(f"   ⏳ 대기 중... ({i+1}/{max_wait})")
    
    time.sleep(10)  # 10초마다 체크

if not server_ready:
    print("\n⚠️ 서버가 시작되지 않았습니다. 로그를 확인하세요.")
    print("💡 모델 로딩이 오래 걸릴 수 있습니다. 잠시 후 다시 시도하세요.")
else:
    # ===== ngrok 터널 생성 =====
    if use_ngrok:
        print("\n" + "="*80)
        print("🌐 ngrok 터널 생성 중...")
        print("="*80)
        
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(SERVER_PORT)
            print(f"\n✅ 터널 생성 완료!")
            print(f"\n📝 공개 URL: {public_url}")
            print(f"\n💡 프론트엔드 설정:")
            print(f"   이 URL을 프론트엔드의 API_URL로 설정하세요:")
            print(f"   {public_url}/api/poem/generate")
            print(f"\n⚠️ 주의:")
            print(f"   - Colab 세션이 종료되면 URL이 변경됩니다")
            print(f"   - 무료 ngrok은 세션당 2시간 제한이 있습니다")
            print(f"   - URL은 이 셀을 다시 실행하면 변경됩니다")
        except Exception as e:
            print(f"❌ ngrok 터널 생성 실패: {e}")
            print(f"\n💡 대안:")
            print(f"   1. Colab의 '코드 셀' > '변수' 메뉴에서 포트 포워딩 사용")
            print(f"   2. 또는 로컬에서 ngrok 사용: ngrok http 8000")
    else:
        print(f"\n📝 로컬 서버 주소:")
        print(f"   http://localhost:{SERVER_PORT}")
        print(f"\n💡 Colab에서 외부 접근을 위해서는 ngrok이 필요합니다.")

# ===== 서버 상태 확인 =====
print("\n" + "="*80)
print("📊 서버 상태")
print("="*80)

try:
    response = requests.get(f'http://localhost:{SERVER_PORT}/health', timeout=10)
    if response.status_code == 200:
        health_data = response.json()
        print(f"✅ 서버 정상 작동 중")
        print(f"   - 모델 타입: {health_data.get('model_type', 'N/A')}")
        print(f"   - 디바이스: {health_data.get('device', 'N/A')}")
        print(f"   - GPU 사용: {health_data.get('has_gpu', False)}")
    else:
        print(f"⚠️ 서버 응답 오류: {response.status_code}")
except Exception as e:
    print(f"⚠️ 서버 상태 확인 실패: {e}")

print("\n" + "="*80)
print("✅ 설정 완료!")
print("="*80)
print("\n💡 서버를 계속 실행하려면 이 셀을 실행 상태로 유지하세요.")
print("💡 서버를 중지하려면 런타임 > 세션 중단을 선택하세요.")

