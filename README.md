# 사용방법
## 초기 세팅
``` bash
sudo apt update
sudo apt upgrade -y

# Python 및 pip 설치 (이미 설치되어 있지 않은 경우)
sudo apt install -y python3 python3-pip python3-dev
sudo apt install -y python3-venv

# 가상환경 생성 (선택사항이지만 권장)
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip
# requirements.txt의 패키지 설치
pip install --no-cache-dir -r requirements.txt
```

## 서버 부팅후 사용법
redis server 실행
``` bash
redis-server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```







