import sys
import json
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QGraphicsView, QGraphicsScene, QListWidget)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont

class NodeItem(QGraphicsScene):
    def __init__(self, node_data, parent=None):
        super().__init__(parent)
        self.node_data = node_data
        self.setSceneRect(0, 0, 800, 600)
        self.node_items = {}
        self.edge_items = []    
        self.draw_node()
        self.draw_edges()
    
    def draw_node(self):
        # 노드 색상 매핑
        color_map = {
            "person": QColor(255, 100, 100),  # 빨간색
            "rock": QColor(100, 100, 255),    # 파란색
            "tree": QColor(100, 255, 100)     # 초록색
        }
        
        # 노드 그리기
        for node in self.node_data["nodes"]:
            x = node["x"] * 8  # 화면 크기에 맞게 스케일 조정
            y = node["y"] * 6
            
            # 노드 생성
            node_color = color_map.get(node["class"], QColor(200, 200, 200))
            ellipse = self.addEllipse(x-20, y-20, 40, 40, 
                                    QPen(Qt.black), 
                                    QBrush(node_color))
            
            # 노드 텍스트 추가
            text = self.addText(f"{node['class']}\nID: {node['id']}")
            text.setDefaultTextColor(Qt.black)
            text.setPos(x-20, y-35)
            
            self.node_items[node["id"]] = (ellipse, text)
    
    def draw_edges(self):
        # 엣지 그리기
        for edge in self.node_data["edges"]:
            source = self.get_node_center(edge["source"])
            target = self.get_node_center(edge["target"])
            
            if source and target:
                # 엣지 선 그리기
                line = self.addLine(source[0], source[1], 
                                  target[0], target[1], 
                                  QPen(Qt.gray, 1, Qt.DashLine))
                
                # 엣지 레이블 추가
                mid_x = (source[0] + target[0]) / 2
                mid_y = (source[1] + target[1]) / 2
                text = self.addText(edge["position"])
                text.setDefaultTextColor(Qt.darkGray)
                text.setPos(mid_x, mid_y)
                
                self.edge_items.append((line, text))
    
    def get_node_center(self, node_id):
        if node_id in self.node_items:
            ellipse = self.node_items[node_id][0]
            rect = ellipse.rect()
            return (rect.x() + rect.width()/2, rect.y() + rect.height()/2)
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Visualization Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 위젯 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 왼쪽 패널
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 파일 선택 버튼
        self.file_btn = QPushButton("Open JSON File")
        self.file_btn.clicked.connect(self.open_file)
        left_layout.addWidget(self.file_btn)
        
        # 파일 목록
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_file)
        left_layout.addWidget(self.file_list)
        
        # 정보 표시 레이블
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)
        
        # 오른쪽 패널 (그래프 표시 영역)
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # 레이아웃 설정
        layout.addWidget(left_panel, 1)
        layout.addWidget(self.view, 4)
        
        self.current_dir = ""
    
    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open JSON File", "", "JSON Files (*.json)")
        
        if file_name:
            self.current_dir = os.path.dirname(file_name)
            self.update_file_list()
            self.load_json_file(file_name)
    
    def update_file_list(self):
        self.file_list.clear()
        if self.current_dir:
            json_files = [f for f in os.listdir(self.current_dir) 
                         if f.endswith('.json')]
            self.file_list.addItems(json_files)
    
    def load_selected_file(self, item):
        """리스트에서 선택된 파일을 로드합니다."""
        if self.current_dir and item:
            file_path = os.path.join(self.current_dir, item.text())
            self.load_json_file(file_path)
    
    def load_json_file(self, file_name):
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
                self.show_graph(data)
                
                # 정보 업데이트
                info_text = f"Image: {data['Image']}\n"
                info_text += f"Label: {data['label']}\n"
                info_text += f"Frame: {data['frame_no']}\n"
                info_text += f"Nodes: {len(data['nodes'])}\n"
                info_text += f"Edges: {len(data['edges'])}"
                self.info_label.setText(info_text)
        except Exception as e:
            self.info_label.setText(f"Error loading file: {str(e)}")
    
    def show_graph(self, data):
        """JSON 데이터로 그래프를 그립니다."""
        try:
            scene = NodeItem(data)
            self.view.setScene(scene)
            # 뷰 크기에 맞게 조정
            self.view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        except Exception as e:
            self.info_label.setText(f"Error displaying graph: {str(e)}")
    
    def resizeEvent(self, event):
        """창 크기가 변경될 때 그래프 크기도 조정합니다."""
        super().resizeEvent(event)
        if self.view.scene():
            self.view.fitInView(self.view.scene().sceneRect(), Qt.KeepAspectRatio)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()