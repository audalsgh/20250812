# 0812~0814 3일간, 5인 1조로 로보플로우 + SegFormer 프로젝트 후, 교육 끝!
[사용한 AI HUB 이미지 자료 링크](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%8F%84%EB%A1%9C&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM003&aihubDataSe=data&dataSetSn=71625)

마지막날에 다시 런파드에서 횡단보도 가중치 넣고 시도했으나 오류가 너무 많아 시간내에 하는건 포기<br>

[엔비디아 딥러닝 교육 모음](https://www.nvidia.com/en-us/training/find-training/?query=10)<br>
**->지금까지 2~3개 정도 교육을 소개해주셨지만, 유료에서 무료로 바뀐 교육들도 존재하니 들어가볼것!**

## 로보플로우 내에서 한 일 5단계.
다운받은 모델파일, zip파일은 구글 드라이브에 백업.<br>
[구글 드라이브 링크](https://drive.google.com/drive/u/1/folders/116VKrgoh3fVBCaZIOt_Q4APXTUCF-8jQ)

1. 조원분들과 라벨링 작업. 800장 목표.
<img width="1457" height="910" alt="image" src="https://github.com/user-attachments/assets/4b61ed5d-5f94-4db3-80f1-c8f17146a470" />

2. 라벨링 완료 후, 데이터셋을 coco 옵션으로 다운받기.
<img width="1462" height="897" alt="image" src="https://github.com/user-attachments/assets/11b4dbe3-6e8a-4a99-9122-6c7586dc525a" />

3. 프로젝트 타입이 segmentation 인 프로젝트를 새로 만들기.
<img width="1306" height="404" alt="image" src="https://github.com/user-attachments/assets/e1ed2284-e3f2-4998-a524-1b677e71c8a5" />

4. 새로만든 segmentation 프로젝트에 800장짜리 coco 데이터셋 zip파일을 업로드하기.
<img width="2505" height="804" alt="image" src="https://github.com/user-attachments/assets/ee39bb48-0eaf-43a7-99e8-a5b1181e5532" />

5. create new version을 만들고, download dataset을 눌러서 Semantic Segmentation Masks 옵션으로 다운받기.
<img width="1445" height="753" alt="image" src="https://github.com/user-attachments/assets/e1550101-168c-4d15-af3d-ebdd411d92d6" />

## 6-1. 이제 코랩이나 runpod로 넘어가서, segformer 전이학습 코드 실습해보기. (gpt 이용)
[colab에서 진행한 코드](0812_Roboflow_Segformer.ipynb)
<img width="1702" height="940" alt="image" src="https://github.com/user-attachments/assets/b51dcda2-1a31-45dd-8489-b26a432f0bfd" />

- mask-semantic.zip 파일 업로드하고, 필요 라이브러리 설치하는 첫번재 셀에서 오류가있지만 작동은 하므로 넘어간다.
- 모든 전이학습이 그렇듯, 20 에포크 하는데에 25분 이상 소요되었다.

## 6-2. 좀더 속도가 빠른 runpod에서 전이학습까지 새로 마치고,<br> 결과영상까지 얻어보기. (gpt 이용)
[runpod에서 진행한 코드](0813_runpod_Roboflow_Segformer.ipynb)

<img width="2560" height="966" alt="image" src="https://github.com/user-attachments/assets/83a79e8f-7ad8-4637-93e5-86cb65633739" />

- 코랩과 동일하게 mask_semantic.zip파일을 업로드하고, 전이학습에 10분정도 소요됨.
- 도로주행 영상은 교수님의 영상을 업로드했고, 결과가 시각화된 영상까지 얻도록 함.
<img width="482" height="678" alt="image" src="https://github.com/user-attachments/assets/4e479003-26da-46c9-b006-a8e9498040b0" />

- 깃허브에 코드 사본저장을 위해 file -> download 를 눌러서 .ipynb 파일을 얻어서 백업했다.

<img width="1197" height="602" alt="image" src="https://github.com/user-attachments/assets/506ced9e-6e46-4cfa-965f-d83d22382d1c" />

**-> 불안불안하긴 하지만 점선, 실선, 중앙선을 표시해주고 있음. 파라미터 수정 or 횡단보도 가중치만 증가시켜서 다시하면 더 좋을듯.**
