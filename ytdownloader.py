import sys
import os
from yt_dlp import YoutubeDL
import pandas as pd




def download_video(url: str, folder: str = 'videos', filename: str = None) -> None:
    os.makedirs(folder, exist_ok=True)
    if filename:
        outtmpl = os.path.join(folder, f'{filename}.%(ext)s')
    else:
        outtmpl = os.path.join(folder, '%(title)s.%(ext)s')
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("✓ 완료")
        return True
    except Exception as e:
        print(f"✗ 다운로드 실패: {e}")
        return False


def download_from_excel(excel_file: str = 'list.xlsx', sheet_name: str = '다운로드용') -> None:
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"엑셀 파일에서 {len(df)}개의 항목을 발견했습니다.\n")
        
        success_count = 0
        fail_count = 0
        
        for index, row in df.iterrows():
            idx = row['IDX']
            url = row['링크']
            
            if pd.isna(url) or not url.strip():
                print(f"[건너뜀] {idx}: URL이 비어있습니다.")
                continue
                
            print(f"[{index+1}/{len(df)}] 다운로드 시작: {idx}")
            
            if download_video(url, filename=str(int(idx))):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n=== 다운로드 완료 ===")
        print(f"성공: {success_count}개")
        print(f"실패: {fail_count}개")
        print(f"전체: {len(df)}개")
        
    except FileNotFoundError:
        print(f"[에러] 파일을 찾을 수 없습니다: {excel_file}")
    except Exception as e:
        print(f"[에러] 엑셀 파일 처리 중 오류: {e}")


def main():
    print("YouTube Downloader Ver.25062401")
    print("1. 직접 링크 입력 모드")
    print("2. 엑셀 파일 일괄 다운로드 모드")
    print("'exit' 또는 'quit' 입력시 프로그램 종료\n")
    
    while True:
        try:
            mode = input("모드 선택 (1/2)> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료")
            sys.exit(0)
            
        if mode.lower() in ('exit', 'quit'):
            print("종료")
            break
        elif mode == '1':
            manual_mode()
        elif mode == '2':
            download_from_excel()
        else:
            print("올바른 모드를 선택해주세요 (1 또는 2)\n")


def manual_mode():
    print("\n=== 직접 링크 입력 모드 ===")
    print("링크 입력시 해당 영상 다운로드")
    print("'back' 입력시 메인 메뉴로 돌아가기\n")
    
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
            sys.exit(0)
        if url.lower() == 'back':
            print("\n메인 메뉴로 돌아갑니다.\n")
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
