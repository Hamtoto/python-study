import sys
import os
from yt_dlp import YoutubeDL


def progress_hook(d):
    status = d.get('status')
    if status == 'downloading':
        percent = d.get('_percent_str', '').strip()
        speed = d.get('_speed_str', '').strip()
        eta = d.get('_eta_str', '').strip()
        print(f"다운로드: {percent} @ {speed} ETA {eta} ", end='\r')
    elif status == 'finished':
        print("\n다운로드 완료")


def download_video(url: str, folder: str = 'videos') -> None:
    os.makedirs(folder, exist_ok=True)
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(folder, '%(title)s.%(ext)s'),
        'progress_hooks': [progress_hook],
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\n완료: {url}\n")
    except Exception as e:
        print(f"[에러] 다운로드 실패 {url}: {e}\n")


def main():
    print("YouTube Downloader Ver.25062401")
    print("링크 입력시 해당 영상 다운로드")
    print("'exit' 또는 'quit' 입력시 프로그램 종료\n")
    while True:
        try:
            url = input("URL> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료")
            sys.exit(0)
        if not url:
            continue
        if url.lower() in ('exit', 'quit'):
            print("종료")
            break
        download_video(url)

if __name__ == '__main__':
    main()

# -------------------
# Packaging & Distribution
# -------------------
# 1. requirements.txt 생성:
#    yt-dlp>=2025.6.24
#
# 2. pip 패키지로 배포할 경우:
#    - setup.py 추가:
#        from setuptools import setup
#        setup(
#            name='youtube-downloader',
#            version='0.1.0',
#            py_modules=['tmp'],  # 또는 파일명
#            install_requires=['yt-dlp>=2025.6.24'],
#            entry_points={'console_scripts': ['yt-downloader=tmp:main']}
#        )
#    - 배포:
#        pip install twine wheel
#        python setup.py sdist bdist_wheel
#        twine upload dist/*
#
# 3. 단일 실행 파일(EXE)로 만들기:
#    pip install pyinstaller
#    pyinstaller --onefile tmp.py --name yt_downloader
#    # 생성된 dist/yt_downloader.exe 를 배포
#
# 4. GitHub에 소스 코드를 저장하고 Releases 기능 활용 권장
#    - README.md에 사용법, 요구사항, 설치 방법 등 문서화
#
# 이제 위 방법 중 원하는 방식을 선택해 배포할 수 있습니다.
