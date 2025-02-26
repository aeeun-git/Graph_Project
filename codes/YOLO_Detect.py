import os
from ultralytics import YOLO

# YOLOv11x 모델 로드
# YOLOv11은 다양한 가중치가 있으므로, 원하는 가중치를 선택할 수 있습니다.
model = YOLO('yolo11x.pt')

# 프로젝트 폴더 경로 설정 (Graph_Project 기준)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 입력 이미지 경로와 출력 라벨 경로 설정 (절대경로 사용)
image_folder = os.path.join(BASE_DIR, 'output/images/')
label_folder = os.path.join(BASE_DIR, 'output/labels/')
os.makedirs(label_folder, exist_ok=True)  # 출력 폴더 생성

# 클래스 ID 설정 (YOLO에서 'Person' 클래스 ID가 0번)
PERSON_CLASS_ID = 0

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 라벨링 작업 시작
for image_file in image_files:
    # 이미지 경로
    image_path = os.path.join(image_folder, image_file)

    # 이미지에서 객체 탐지 실행
    results = model(image_path)

    # 라벨 데이터 파일 저장 경로
    txt_file_name = os.path.splitext(image_file)[0] + '.txt'
    txt_file_path = os.path.join(label_folder, txt_file_name)

    # 라벨 데이터 저장
    with open(txt_file_path, 'w') as f:
        # 결과 가져오기
        for result in results:
            boxes = result.boxes  # Bounding box 정보
            for box in boxes:
                # Box 정보 추출
                x_center, y_center, width, height = box.xywhn[0].tolist()  # Normalized 값
                class_id = int(box.cls[0])  # 클래스 ID

                # Person 클래스만 저장
                if class_id == PERSON_CLASS_ID:
                    # YOLO txt 형식: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 아무것도 검출되지 않아도 빈 파일 생성
    print(f"Processed: {image_file}, Saved: {txt_file_path}")

print("모든 이미지 처리 완료.")
