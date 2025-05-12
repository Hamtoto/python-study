import asyncio
import time

"""
참조 Link
https://www.daleseo.com/python-asyncio/
https://wikidocs.net/125092
"""

#동기 프로그래밍

def find_user_sync(n):
    for i in range(1, n + 1):
        print(f'{n}명 중 {i}번 째 사용자 조회 중...')
        time.sleep(1)
    print(f'> 총 {n}명 사용자 동기 조회 완료!')

def process_sync():
    start = time.time()
    find_user_sync(3)
    find_user_sync(2)
    find_user_sync(1)
    end = time.time()
    print(f'>>> 동기 처리 총 소요 시간: {end - start}')


#비동기 프로그래밍
#async 선언으로 동기 함수를 비동기(coroutin)으로 변경

async def find_users_async(n):
    for i in range(1, n + 1):
        print(f'{n}명 중 {i}번 째 사용자 조회 중...')
        await asyncio.sleep(1)
    print(f"> 총 {n}명 사용자 비동기 조회 완료!")

async def process_async():
    start = time.time()
    await asyncio.gather(
        find_users_async(3),
        find_users_async(2),
        find_users_async(1)
    )
    end = time.time()
    print(f'>>> 비동기 처리 총 소요 시간: {end - start}')

if __name__ == '__main__':
    #동기 호출
    #process_sync()
    #비동기 호출
    asyncio.run(process_async())