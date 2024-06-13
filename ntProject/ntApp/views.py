from django.shortcuts import render, redirect
from .models import Video
from .forms import VideoForm
import os
import cv2
import numpy as np
import uuid

def upload(request):
    if request.method == 'POST':
        videoForm = VideoForm(request.POST, request.FILES)

        if videoForm.is_valid():
            videoForm.save()
            return redirect('result')
        else:
            return render(request, 'ntApp/upload.html', { 'videoForm': videoForm })
    else:
        videoForm = VideoForm()
        return render(request, 'ntApp/upload.html', { 'videoForm': videoForm })

def result(request):
    path = os.getcwd() + '\\media\\' + str(Video.objects.last().file).replace('/', '\\')

    # Optical Flow - Lucas-Kanade

    # video 읽기
    video = cv2.VideoCapture(path)
    ret, frame = video.read()

    # 특징점 추출 관련 인자
    features_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 3, useHarrisDetector=False, k=0.04)
    # Lucas-Kanade 계산 관련 인자
    params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # 랜덤 색상
    color = np.random.randint(0, 255, (100, 3))


    # 첫번째 프레임의 coner detection
    prev_frame = frame
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    f0 = cv2.goodFeaturesToTrack(prev_frame_gray, mask = None, **features_params)

    mask = np.zeros_like(prev_frame)

    # 오른쪽 방향의 프레임 인덱스 배열
    rFrames = []
    frame_idx = 0
    maxRResult = (0, 0, 0)

    # 영상 읽기
    while True:
        ret, frame = video.read()

        if not ret:
            print('Check your video.')
            break

        next_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow 계산
        f1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, next_frame_gray, f0, None, **params)

        # 유효한 특징점 찾기
        if f1 is not None:
            f0_good = f0[st == 1]
            f1_good = f1[st == 1]

        rFlag = False
        # 오른쪽 방향으로 이동한 것의 거리 합 => 최대 거리를 구하기 위함
        rDistance = 0
        for i, (new, old) in enumerate(zip(f1_good, f0_good)):
            x1, y1 = new.ravel()
            x0, y0 = old.ravel()

            # 오른쪽 이동 방향
            # 이동 방향의 정도를 직접 정해줘야 한다는 한계 (threshold)
            if (x1 - x0 > 5):
                rFlag = True
                rDistance += (x1 - x0)


            # 그리기
            mask = cv2.line(mask, (int(x1),int(y1)), (int(x0),int(y0)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(x1),int(y1)), 5, color[i].tolist(), -1)
        
        # 오른쪽 방향 프레임 배열에 저장
        if (rFlag):
            rFrames.append((frame_idx, rDistance))

        drawn_frame = cv2.add(frame, mask)
        
        # 출력
        cv2.imshow('optical flow', drawn_frame)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

        # 이전 프레임 갱신
        prev_frame_gray = next_frame_gray.copy()
        f0 = f1_good.reshape(-1, 1, 2)

        frame_idx += 1

        # 연속성을 판단하기 위한 변수
        rCount = 0
        newRFrames = []

        # 구간별, 거리의 합을 구하기 (시작점, 끝점, 거리)
        # ex. [(1, 10), (2, 20), (3, 30), (7, 70)] -> [(1, 3, 60), (7, 7, 70)]
        if rFrames:
            rFrame = rFrames[0][0]
            rTotalDistance = 0

            for i, item in enumerate(rFrames):
                rCount += 1
                rTotalDistance += item[1]

                # 어느정도까지 연속된다고 평가할 것인가를 직접 정해야 한다는 한계 (threshold)
                # ex. 3의 경우, 1, 2, 5 까지 연속으로 평가
                if item[0] > rFrame + rCount + 3:
                    newRFrames.append((rFrame, rFrame + rCount, rTotalDistance))
                    rFrame = item[0]
                    rCount = 0
                    rTotalDistance = 0
                elif i == len(rFrames) - 1 and rFrame == rFrames[0][0]:
                    # 오른쪽으로만 움직인 경우,
                    newRFrames.append((rFrame, rFrame + rCount, rTotalDistance))

            # 오른쪽으로 최대 이동한 프레임 구간 찾기
            if newRFrames:
                maxRResult = max(newRFrames, key = lambda x: x[2])

    # 오른쪽으로 최대 이동한 구간의 프레임 저장
    video = cv2.VideoCapture(path)
    i = 0
    my_id = uuid.uuid1()
    result_list = []

    while True:
        ret, frame = video.read()

        if not ret:
            print('Check your video')
            break

        if (i >= maxRResult[0] and i <= maxRResult[1]):
            title = 'ntApp/static/' + str(my_id) + 'right_frame' + str(i) + '.jpg'
            cv2.imwrite(title, frame)
            result_list.append(str(my_id) + 'right_frame' + str(i) + '.jpg')

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

        i += 1

    # 안전하게 파일이 저장되었는지 확인
    while True:
        all_files_exist = all(os.path.exists(os.getcwd() + '\\ntApp\\static\\' + result) for result in result_list)
        if not all_files_exist:
            print("Error: Some files were not saved correctly.")
        else:
            break
        
    return render(request, 'ntApp/result.html', { 'results': result_list, 'standard': result_list[0] })
