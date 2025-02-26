import sys
import os
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QWidget, QFileDialog, QMessageBox, QScrollArea, QSlider,
    QListWidget, QInputDialog, QDialog
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QClipboard
from PySide6.QtCore import Qt, QPoint, QRect

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self.start_point = None
        self.end_point = None
        self.rectangles = []
        self.image_size = (1, 1)
        self.clipboard = QApplication.clipboard()

    def reset_labels(self):
        """라벨링 데이터 초기화"""
        self.rectangles = []
        self.start_point = None
        self.end_point = None
        self.update()

    def set_image_size(self, width, height):
        self.image_size = (width, height)

    def convert_to_image_coords(self, rect):
        label_width = self.width()
        label_height = self.height()
        img_width, img_height = self.image_size

        x_scale = img_width / label_width
        y_scale = img_height / label_height

        x_min = rect.left() * x_scale
        y_min = rect.top() * y_scale
        x_max = rect.right() * x_scale
        y_max = rect.bottom() * y_scale

        return QRect(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def get_yolo_format(self, rect, class_id):
        """YOLO 포맷으로 좌표 변환"""
        img_width, img_height = self.image_size
        
        x_min = rect.left()
        y_min = rect.top()
        x_max = rect.right()
        y_max = rect.bottom()

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()

            rect_in_image_coords = self.convert_to_image_coords(rect)

            class_id, ok = QInputDialog.getInt(
                self, "클래스 ID 입력", "클래스 ID를 입력하세요:", minValue=0
            )
            if ok:
                self.rectangles.append((rect_in_image_coords, class_id))
                
                # YOLO 포맷으로 변환하여 클립보드에 복사
                yolo_format = self.get_yolo_format(rect_in_image_coords, class_id)
                self.clipboard.setText(yolo_format)

            self.start_point = None
            self.end_point = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)

        for rect, class_id in self.rectangles:
            x_scale = self.width() / self.image_size[0]
            y_scale = self.height() / self.image_size[1]
            label_rect = QRect(
                int(rect.left() * x_scale),
                int(rect.top() * y_scale),
                int(rect.width() * x_scale),
                int(rect.height() * y_scale)
            )
            painter.drawRect(label_rect)
            painter.drawText(label_rect.topLeft(), str(class_id))

        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

    def append_yolo_format(self, file_path):
        """기존 라벨을 유지하면서 새로운 라벨 추가"""
        # 기존 라벨 읽기
        existing_labels = []
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                existing_labels = file.readlines()
        
        # 새로운 라벨 추가
        with open(file_path, "w") as file:
            # 기존 라벨 쓰기
            for label in existing_labels:
                file.write(label.strip() + "\n")
            
            # 새로운 라벨 추가
            for rect, class_id in self.rectangles:
                yolo_format = self.get_yolo_format(rect, class_id)
                file.write(yolo_format + "\n")


class LabelingDialog(QDialog):
    def __init__(self, image_path, label_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이미지 라벨링")
        self.setModal(True)
        self.resize(900, 700)

        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: lightgray;")
        self.image_label.setFixedSize(800, 600)

        # 이미지 로드
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.set_image_size(pixmap.width(), pixmap.height())

        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(lambda: self.save_labels(label_path))

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_labels(self, label_path):
        self.image_label.append_yolo_format(label_path)
        QMessageBox.information(self, "알림", "라벨이 저장되었습니다.")
        self.accept()


class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window 설정
        self.setWindowTitle("YOLO 라벨링 도구")
        self.setGeometry(100, 100, 1000, 800)

        # 이미지 및 라벨 경로
        self.image_folder = ''
        self.label_folder = ''
        self.current_image_index = 0
        self.image_files = []
        self.deleted_labels = []

        # UI 요소 설정
        self.init_ui()

    def init_ui(self):
        # 중앙 위젯과 레이아웃 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃: 수평 레이아웃
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 왼쪽 레이아웃 (이미지 뷰와 설정)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        # 오른쪽 레이아웃 (이미지 리스트)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)

        # 경로 입력 필드
        path_layout = QHBoxLayout()
        left_layout.addLayout(path_layout)

        # 이미지 폴더 필드
        self.image_path_field = QLineEdit()
        self.image_path_field.setPlaceholderText("이미지 폴더 경로")
        path_layout.addWidget(self.image_path_field)

        image_browse_button = QPushButton("이미지 선택")
        image_browse_button.clicked.connect(self.browse_image_folder)
        path_layout.addWidget(image_browse_button)

        # 라벨 폴더 필드
        self.label_path_field = QLineEdit()
        self.label_path_field.setPlaceholderText("라벨 폴더 경로")
        path_layout.addWidget(self.label_path_field)

        label_browse_button = QPushButton("라벨 선택")
        label_browse_button.clicked.connect(self.browse_label_folder)
        path_layout.addWidget(label_browse_button)

        # 이미지 표시용 ScrollArea
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("이미지를 불러오세요")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        left_layout.addWidget(self.scroll_area)

        # 스크롤 슬라이더 추가
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.scroll_images)
        left_layout.addWidget(self.slider)

        # 오른쪽 레이아웃: 이미지 리스트와 라벨 데이터
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.image_list_clicked)
        right_layout.addWidget(QLabel("이미지 리스트"))
        right_layout.addWidget(self.image_list_widget)

        self.label_data_widget = QListWidget()
        right_layout.addWidget(QLabel("라벨 데이터"))
        right_layout.addWidget(self.label_data_widget)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)

        load_button = QPushButton("불러오기")
        load_button.clicked.connect(self.load_images)
        button_layout.addWidget(load_button)

        open_button = QPushButton("txt파일 열기")
        open_button.clicked.connect(self.open_txt)
        button_layout.addWidget(open_button)

        insert_all_button = QPushButton("txt파일 일괄 삽입")
        insert_all_button.clicked.connect(self.insert_all_txt)
        button_layout.addWidget(insert_all_button)

        labeling_button = QPushButton("현재 파일 라벨링")
        labeling_button.clicked.connect(self.open_labeling_dialog)
        button_layout.addWidget(labeling_button)

    def browse_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder:
            self.image_path_field.setText(folder)

    def browse_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택")
        if folder:
            self.label_path_field.setText(folder)

    def load_images(self):
        # 이미지 및 라벨 폴더 읽기
        self.image_folder = self.image_path_field.text()
        self.label_folder = self.label_path_field.text()

        # 이미지 파일 로드
        if not os.path.exists(self.image_folder):
            QMessageBox.warning(self, "경고", "유효한 이미지 폴더를 선택하세요!")
            return

        self.image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()

        if not self.image_files:
            QMessageBox.warning(self, "경고", "이미지 폴더에 파일이 없습니다!")
            return

        # 슬라이더 설정
        self.slider.setMaximum(len(self.image_files) - 1)
        self.slider.setValue(0)

        # 이미지 리스트 업데이트
        self.image_list_widget.clear()
        self.image_list_widget.addItems(self.image_files)

        # 첫 번째 이미지 표시
        self.current_image_index = 0
        self.show_image_with_labels()

    def show_image_with_labels(self):
        if not self.image_files:
            return

        # 현재 이미지 파일 경로
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_folder, image_file)

        # 이미지 로드
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "경고", f"이미지를 불러올 수 없습니다: {image_file}")
            return

        # 이미지 크기 축소 (ScrollArea의 크기에 맞춤)
        scroll_area_width = self.scroll_area.width()
        scroll_area_height = self.scroll_area.height()
        pixmap = pixmap.scaled(scroll_area_width - 20, scroll_area_height - 20, 
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 라벨 파일 경로
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_file)

        # 라벨 데이터 표시 초기화
        self.label_data_widget.clear()

        # 라벨 표시
        painter = QPainter(pixmap)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Bounding box를 픽셀 좌표로 변환
                    x_center *= pixmap.width()
                    y_center *= pixmap.height()
                    width *= pixmap.width()
                    height *= pixmap.height()
                    x = int(x_center - width / 2)
                    y = int(y_center - height / 2)

                    # 박스 그리기
                    pen = QPen(Qt.red, 2)
                    painter.setPen(pen)
                    painter.drawRect(x, y, int(width), int(height))

                    # 클래스 이름 표시
                    painter.setPen(Qt.blue)
                    painter.drawText(x, y - 5, f"Class: {int(class_id)}")

                    # 라벨 데이터를 리스트에 추가
                    self.label_data_widget.addItem(
                        f"Class: {int(class_id)}, x: {x_center:.2f}, y: {y_center:.2f}, w: {width:.2f}, h: {height:.2f}"
                    )

        painter.end()
        self.image_label.setPixmap(pixmap)

    def scroll_images(self, value):
        self.current_image_index = value
        self.show_image_with_labels()
        
        # 현재 이미지를 리스트에서 선택 상태로 만들기
        self.image_list_widget.setCurrentRow(self.current_image_index)
    
    def image_list_clicked(self, item):
        # 리스트에서 선택한 이미지의 인덱스 가져오기
        self.current_image_index = self.image_files.index(item.text())
        self.slider.setValue(self.current_image_index)
        self.show_image_with_labels()

    def open_txt(self):
        if not self.image_files:
            QMessageBox.warning(self, "경고", "먼저 이미지를 불러오세요!")
            return

        # 현재 이미지 파일 이름
        image_file = self.image_files[self.current_image_index]
        
        # 해당 이미지의 txt 파일 경로
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_file)

        # 파일이 존재하는지 확인
        if not os.path.exists(label_path):
            QMessageBox.warning(self, "경고", "해당 이미지의 라벨 파일이 존재하지 않습니다!")
            return

        # 운영 체제에 따라 메모장 열기
        if sys.platform.startswith('win'):  # Windows
            os.startfile(label_path)
        elif sys.platform.startswith('darwin'):  # macOS
            subprocess.call(('open', label_path))
        else:  # Linux
            subprocess.call(('xdg-open', label_path))

    def insert_all_txt(self):
        # 이미지 폴더와 라벨 폴더가 설정되었는지 확인
        if not self.image_folder or not self.label_folder:
            QMessageBox.warning(self, "경고", "먼저 이미지와 라벨 폴더를 선택하세요!")
            return

        # 삽입할 텍스트 파일 선택
        insert_file, _ = QFileDialog.getOpenFileName(self, "삽입할 txt 파일 선택", "", "Text Files (*.txt)")
        
        if not insert_file:
            return

        # 삽입할 텍스트 읽기
        with open(insert_file, 'r') as f:
            insert_text = f.read().strip()

        # 모든 라벨 파일에 대해 처리
        for image_file in self.image_files:
            # 해당 이미지의 라벨 파일 경로
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_file)

            # 기존 라벨 내용 읽기
            existing_labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    existing_labels = f.readlines()

            # 새로운 내용 추가
            with open(label_path, 'w') as f:
                # 기존 라벨들 먼저 쓰기
                f.writelines(existing_labels)
                
                # 줄바꿈 추가 (기존 내용이 있다면)
                if existing_labels and not existing_labels[-1].endswith('\n'):
                    f.write('\n')
                
                # 새로운 라벨 내용 추가
                f.write(insert_text + '\n')

        # 작업 완료 메시지
        QMessageBox.information(self, "완료", "모든 라벨 파일에 내용을 추가했습니다.")

    def open_labeling_dialog(self):
        if not self.image_files or not self.label_folder:
            QMessageBox.warning(self, "경고", "이미지와 라벨 폴더를 먼저 선택하세요!")
            return

        # 현재 이미지 파일 경로
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_folder, image_file)

        # 라벨 파일 경로
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_file)

        # 라벨링 다이얼로그 열기
        dialog = LabelingDialog(image_path, label_path, self)
        if dialog.exec_() == QDialog.Accepted:
            # 라벨링 후 이미지 업데이트
            self.show_image_with_labels()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tool = LabelingTool()
    tool.show()
    sys.exit(app.exec())