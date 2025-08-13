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

7. runpod에서 전이학습까지 새로 마치고, 결과영상까지 얻어보기. (gpt 이용)


### 첫 코드에서 설치와, Semantic Segmentation Masks zip파일을 업로드했기에 오류가 떠도 일단 남겨두었다.

```python
# 0) 설치
!pip -q install transformers accelerate evaluate opencv-python-headless pillow

# 1) ZIP 업로드 (수동 업로드 창이 뜹니다)
from google.colab import files
up = files.upload()  # 방금 받은 Roboflow ZIP 선택
ZIP_PATH = "/content/" + list(up.keys())[0]

# 2) 압축 풀기
import os, zipfile, glob, shutil, re
EXTRACT_DIR = "/content/ds_rf"
if os.path.isdir(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as z: z.extractall(EXTRACT_DIR)
print("unzipped to", EXTRACT_DIR, "->", os.listdir(EXTRACT_DIR))

# 3) 데이터 구조 파악 (train/valid/test)
def find_split_dir(root, names=("train","valid","val","test")):
    found={}
    for n in names:
        p=os.path.join(root,n)
        if os.path.isdir(p): found["valid" if n in ("valid","val") else n]=p
    return found
splits = find_split_dir(EXTRACT_DIR)
if not splits: raise RuntimeError("train/valid/test 폴더를 찾지 못함. ZIP 내용 확인")

# 4) 학습 설정
COLLAPSE_TO_BINARY = True  # True: 모든 non-zero를 'lane(1)'로 합치기
if COLLAPSE_TO_BINARY:
    CLASS_NAMES = ["background","lane"]
else:
    # 예) 멀티클래스: background + lane 계열
    CLASS_NAMES = ["background","lane","lane-dot","lane-mid","lane_crosswalk"]

id2label = {i:n for i,n in enumerate(CLASS_NAMES)}
label2id = {n:i for i,n in id2label.items()}
NUM_LABELS = len(CLASS_NAMES)

# 5) 데이터셋 클래스 (Roboflow 'Semantic Segmentation Masks' 구조 자동 대응)
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

def normalize_stem(s):
    s=os.path.splitext(os.path.basename(s))[0]
    s=re.sub(r'(_mask|-mask)$','',s)
    return s

def index_mask_array(mask_img, collapse_to_binary=True):
    # 팔레트 PNG/그레이스케일 모두 지원
    m = np.array(mask_img.convert("L"), dtype=np.uint8)
    if collapse_to_binary:
        m = (m>0).astype(np.uint8)  # 0/1
    else:
        # 0..K 인덱스 그대로 사용 (배경은 0이어야 함)
        pass
    return m

class RFSegFolder(Dataset):
    def __init__(self, split_dir, processor):
        self.img_dir = os.path.join(split_dir, "images")
        # 마스크 폴더 후보
        cand = ["masks","labels","annotations","masks_png","labels_png"]
        self.mask_dirs = [os.path.join(split_dir,c) for c in cand if os.path.isdir(os.path.join(split_dir,c))]
        if not self.mask_dirs:
            # 일부 버전에선 images와 같은 폴더에 있을 수 있음(드물지만)
            self.mask_dirs = [split_dir]
        self.processor = processor

        # 마스크 인덱스 구축
        mask_map = {}
        for md in self.mask_dirs:
            for p in glob.glob(os.path.join(md, "*.png")):
                mask_map[normalize_stem(p)] = p

        # 이미지-마스크 페어 만들기
        self.items=[]
        for ip in sorted(glob.glob(os.path.join(self.img_dir, "*.*"))):
            st = normalize_stem(ip)
            mp = mask_map.get(st)
            if mp and os.path.exists(mp):
                self.items.append((ip, mp))
        if not self.items:
            raise RuntimeError(f"No (image,mask) pairs in {split_dir}. 마스크 폴더명이 'masks/labels/annotations' 중 하나인지 확인")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        image = Image.open(ip).convert("RGB")
        mask  = Image.open(mp)
        mask  = index_mask_array(mask, COLLAPSE_TO_BINARY)
        enc = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

# 6) 프로세서/모델
from transformers import SegformerForSemanticSegmentation
import torch, evaluate

CKPT = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(CKPT, reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(
    CKPT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 7) 데이터 로더 구성
train_dir = splits.get("train")
valid_dir = splits.get("valid") or splits.get("val") or train_dir  # valid 없으면 train 재사용(데모용)
train_ds = RFSegFolder(train_dir, processor)
val_ds   = RFSegFolder(valid_dir, processor)

# 8) 학습
from transformers import TrainingArguments, Trainer
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(
        predictions=preds, references=labels,
        num_labels=NUM_LABELS, ignore_index=255, reduce_labels=False
    )

args = TrainingArguments(
    output_dir="segformer-lane",
    learning_rate=5e-5,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
trainer.train()

# 9) 저장
trainer.save_model("segformer-lane/best")
processor.save_pretrained("segformer-lane/best")
print("✅ Saved to segformer-lane/best")
```

```python
# === 패치: RFSegFolder를 더 관대한 버전으로 재정의 ===
import os, glob, re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 이미지/마스크 파일명 매칭을 위해 뒤에 붙는 접미어들을 제거
_SUFFIX_RE = re.compile(r'(_|-)(mask|masks|label|labels|seg|segment|segmentation)$', re.I)

def _stem_no_suffix(path):
    s = os.path.splitext(os.path.basename(path))[0]
    s = _SUFFIX_RE.sub('', s)   # ..._mask, -labels 등 제거
    return s

def _is_img(name): 
    return name.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))

class RFSegFolder(Dataset):
    def __init__(self, split_dir, processor):
        # 1) 이미지 폴더 탐색: 'images/'가 있으면 거기, 없으면 split 루트에서 바로 찾기
        img_cands = [os.path.join(split_dir, "images"), split_dir]
        self.img_dir = None
        for d in img_cands:
            if os.path.isdir(d) and any(_is_img(f) for f in os.listdir(d)):
                self.img_dir = d
                break
        if self.img_dir is None:
            raise RuntimeError(f"No images found in {split_dir}")

        # 2) 마스크 폴더 후보: labels/masks/annotations/… 없으면 split 루트까지 포함
        mask_cands = ["masks","labels","annotations","masks_png","labels_png","mask","Labels","Masks"]
        self.mask_dirs = [os.path.join(split_dir, c) for c in mask_cands if os.path.isdir(os.path.join(split_dir, c))]
        if not self.mask_dirs:
            # 마지막 수단: split 디렉토리 안에서 PNG가 있는 모든 폴더를 스캔(이미지 폴더 제외)
            self.mask_dirs = []
            for root, dirs, files in os.walk(split_dir):
                if os.path.abspath(root) == os.path.abspath(self.img_dir):
                    continue
                if any(f.lower().endswith(".png") for f in files):
                    self.mask_dirs.append(root)
            if not self.mask_dirs:
                # 정말 없으면 루트도 후보에 포함(아주 드문 케이스)
                self.mask_dirs = [split_dir]

        self.processor = processor

        # 3) 마스크 인덱스 구축 (동일 stem 매칭)
        mask_map = {}
        for md in self.mask_dirs:
            for p in glob.glob(os.path.join(md, "*.png")):
                mask_map[_stem_no_suffix(p)] = p

        # 4) 이미지-마스크 페어 만들기
        self.items = []
        for ip in sorted(glob.glob(os.path.join(self.img_dir, "*.*"))):
            if not _is_img(ip):
                continue
            st = _stem_no_suffix(ip)
            mp = mask_map.get(st)
            if mp and os.path.exists(mp):
                self.items.append((ip, mp))

        if not self.items:
            # 디버깅 도움: 폴더 안에 뭐가 있는지 조금 찍어줌
            print("[DEBUG] img_dir:", self.img_dir)
            print("[DEBUG] mask_dirs:", self.mask_dirs[:3], "…", f"({sum(len(glob.glob(os.path.join(d,'*.png'))) for d in self.mask_dirs)} masks png)")
            raise RuntimeError(f"No (image,mask) pairs in {split_dir}. "
                               f"이미지/마스크 파일명이 서로 매칭되는지(예: abc.jpg ↔ abc_mask.png) 확인해주세요.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        image = Image.open(ip).convert("RGB")
        # 팔레트/그레이스케일 모두 지원: 0=배경, 1+=전부 차선으로 뭉치기(이진)
        m = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
        m = (m > 0).astype(np.uint8)  # 이진 세팅 (여러 클래스를 쓰려면 여기 로직 바꿔도 됨)
        enc = processor(images=image, segmentation_maps=m, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}
```

```python
from transformers import TrainingArguments, Trainer
import numpy as np, evaluate, torch

metric = evaluate.load("mean_iou")

def _to_py(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (N, C, h, w), labels: (N, H, W)
    if isinstance(logits, tuple):
        logits = logits[0]
    lt = torch.from_numpy(logits)
    yt = torch.from_numpy(labels)

    # 라벨 크기에 맞춰 업샘플(크기 불일치 방지)
    lt_up = torch.nn.functional.interpolate(
        lt, size=yt.shape[-2:], mode="bilinear", align_corners=False
    )
    preds = lt_up.argmax(dim=1).cpu().numpy()

    res = metric.compute(
        predictions=preds,
        references=labels,
        num_labels=getattr(model.config, "num_labels", 2),
        ignore_index=255,
        reduce_labels=False,
    )
    # ✅ JSON 직렬화 가능하도록 변환
    return {k: _to_py(v) for k, v in res.items()}

args = TrainingArguments(
    output_dir="segformer-lane",
    learning_rate=5e-5,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()  # 체크포인트에서 이어하려면: trainer.train(resume_from_checkpoint=True)
```
