# 4인 1조로 로보플로우 + SegFormer 프로젝트

## 로보플로우 내에서 한일
[사용한 AI HUB 이미지 자료 링크](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%8F%84%EB%A1%9C&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM003&aihubDataSe=data&dataSetSn=71625)

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

6. 이제 코랩이나 runpod로 넘어가서, segformer 전이학습 코드 실습해보기. (gpt 이용)
<img width="1702" height="940" alt="image" src="https://github.com/user-attachments/assets/b51dcda2-1a31-45dd-8489-b26a432f0bfd" />

- mask-semantic.zip 파일 업로드하고, 필요 라이브러리 설치하는 첫번재 셀에서 오류가있지만 작동은 하므로 넘어간다.
- 모든 전이학습이 그렇듯, 20 에포크 하는데에 25분 이상 소요되었다.

7. 좀더 속도가 빠른 runpod에서 전이학습까지 새로 마치고, 결과영상까지 얻어보기. (gpt 이용)
<img width="2523" height="873" alt="image" src="https://github.com/user-attachments/assets/a4be7b62-7e65-446e-a6d5-a2dc65f23106" />

- 코랩과 동일하게 mask_semantic.zip파일을 업로드하고, 10분정도 전이학습을 한 후, 결과영상까지 얻도록 실습 진행함.

<img width="2389" height="1079" alt="image" src="https://github.com/user-attachments/assets/03ef58c0-6ff3-4e8c-ac00-8637a60fff7b" />

**-> 불안불안하긴 하지만 점선, 실선, 횡단보도까지 표시해주고 있음. 파라미터 수정 or 색상을 구분하게 다시 학습하면 더 좋을듯.**
