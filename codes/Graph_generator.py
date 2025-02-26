import os
import json
import math
import numpy as np
from glob import glob
from enum import Enum
from typing import Dict, Optional, Tuple
from PIL import Image
from skimage.feature import graycomatrix, graycoprops  # greycomatrix에서 graycomatrix로 변경
import cv2

class SpatialRelation(Enum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    FRONT_LEFT = 4
    FRONT_RIGHT = 5
    BACK_LEFT = 6
    BACK_RIGHT = 7
    NONE = 8

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def determine_spatial_relation(rel_x, rel_y, distance, distance_threshold=30, angle_threshold=45):
    """
    두 객체 간의 상대적 위치를 기반으로 공간적 관계를 결정
    Missing person의 경우 항상 BACK 관계 반환
    """
    if distance > distance_threshold:
        return SpatialRelation.NONE

    angle = math.degrees(math.atan2(rel_y, rel_x))
    angle = (angle + 360) % 360

    if (45 - angle_threshold) <= angle < (45 + angle_threshold):
        return SpatialRelation.FRONT
    elif (225 - angle_threshold) <= angle < (225 + angle_threshold):
        return SpatialRelation.BACK
    elif (315 - angle_threshold) <= angle < (315 + angle_threshold):
        return SpatialRelation.RIGHT
    elif (135 - angle_threshold) <= angle < (135 + angle_threshold):
        return SpatialRelation.LEFT
    elif (315 + angle_threshold) <= angle or angle < (45 - angle_threshold):
        return SpatialRelation.FRONT_RIGHT
    elif (45 + angle_threshold) <= angle < (135 - angle_threshold):
        return SpatialRelation.FRONT_LEFT
    elif (135 + angle_threshold) <= angle < (225 - angle_threshold):
        return SpatialRelation.BACK_LEFT
    elif (225 + angle_threshold) <= angle < (315 - angle_threshold):
        return SpatialRelation.BACK_RIGHT
    else:
        return SpatialRelation.NONE


class PersonTracker:
    def __init__(self):
        self.prev_nearest_obj: Optional[Dict] = None
        self.prev_nearest_dist: float = float('inf')
        self.prev_positions = {}  # 이전 프레임의 위치 저장
        self.prev_time = None
        self.frame_rate = 30.0  # 초당 프레임 수 (필요에 따라 조정)
    
    def calculate_motion(self, current_pos, node_id, current_time):
        """
        객체의 속도와 방향을 계산합니다.
        
        Args:
            current_pos: (x, y) 현재 위치
            node_id: 객체 ID
            current_time: 현재 프레임 번호
            
        Returns:
            speed: 속도 (pixels/second)
            heading: 이동 방향 (도(degree) 단위, 0-360)
        """
        if node_id not in self.prev_positions or self.prev_time is None:
            self.prev_positions[node_id] = current_pos
            self.prev_time = current_time
            return 0, 0
        
        prev_pos = self.prev_positions[node_id]
        
        # 위치 변화 계산
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # 시간 간격 계산 (초 단위)
        dt = (current_time - self.prev_time) / self.frame_rate
        if dt == 0:
            return 0, 0
            
        # 속도 계산 (pixels/second)
        distance = math.sqrt(dx**2 + dy**2)
        speed = distance / dt if dt > 0 else 0
        
        # 방향 계산 (라디안 -> 도)
        heading = math.degrees(math.atan2(dy, dx))
        heading = (heading + 360) % 360  # 0-360도 범위로 변환
        
        # 현재 위치를 다음 계산을 위해 저장
        self.prev_positions[node_id] = current_pos
        self.prev_time = current_time
        
        return speed, heading
    
    def update(self, nodes: list, has_person: bool, current_time: int):
        """
        객체 추적 정보를 업데이트합니다.
        
        Args:
            nodes: 현재 프레임의 모든 노드 리스트
            has_person: person 객체 존재 여부
            current_time: 현재 프레임 번호
        """
        if has_person:
            person_node = next((node for node in nodes if node["class"] == "person"), None)
            if person_node:
                min_dist = float('inf')
                nearest_obj = None
                
                for node in nodes:
                    if node["class"] != "person":
                        dist = calculate_distance(
                            person_node["x"], person_node["y"],
                            node["x"], node["y"]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_obj = node.copy()
                
                self.prev_nearest_obj = nearest_obj
                self.prev_nearest_dist = min_dist
        
        return self.prev_nearest_obj

def extract_glcm_features(image: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    이미지 패치에서 GLCM 특징을 추출합니다.
    
    Args:
        image: 그레이스케일 이미지 패치
        distances: GLCM 계산을 위한 거리 값들
        angles: GLCM 계산을 위한 각도 값들
    
    Returns:
        dict: GLCM 특징들 (contrast, dissimilarity, homogeneity, energy, correlation)
    """
    # 이미지가 너무 작으면 리사이즈
    if image.shape[0] < 8 or image.shape[1] < 8:
        image = cv2.resize(image, (8, 8))
    
    # GLCM 계산
    glcm = graycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    
    # GLCM 특징 추출
    features = {
        'contrast': float(graycoprops(glcm, 'contrast').mean()),
        'dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
        'homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
        'energy': float(graycoprops(glcm, 'energy').mean()),
        'correlation': float(graycoprops(glcm, 'correlation').mean())
    }
    
    return features

def get_object_patch(image: np.ndarray, x_center: float, y_center: float, width: float, height: float):
    """
    이미지에서 객체의 바운딩 박스 영역을 추출합니다.
    
    Args:
        image: 전체 이미지
        x_center, y_center: 객체의 중심 좌표 (normalized)
        width, height: 객체의 너비와 높이 (normalized)
    
    Returns:
        numpy.ndarray: 추출된 객체 이미지 패치
    """
    img_height, img_width = image.shape[:2]
    
    # 정규화된 좌표를 픽셀 좌표로 변환
    x_center_px = int(x_center * img_width)
    y_center_px = int(y_center * img_height)
    width_px = int(width * img_width)
    height_px = int(height * img_height)
    
    # 바운딩 박스 좌표 계산
    x1 = max(0, x_center_px - width_px // 2)
    y1 = max(0, y_center_px - height_px // 2)
    x2 = min(img_width, x_center_px + width_px // 2)
    y2 = min(img_height, y_center_px + height_px // 2)
    
    # 이미지 패치 추출
    patch = image[y1:y2, x1:x2]
    
    return patch

def yolo_to_json(image_path, label_path, output_path, distance_threshold=30):
    class_mapping = {
        0: {"class": "person", "state": "dynamic"},
        1: {"class": "rock", "state": "static"},
        2: {"class": "tree", "state": "static"},
        3: {"class": "stonewall", "state": "static"},
        4: {"class": "fence", "state": "static"},
        5: {"class": "pole", "state": "static"},
        6: {"class": "car", "state": "static"},
    }
    
    label_files = sorted(glob(os.path.join(label_path, "*.txt")))
    person_tracker = PersonTracker()
    
    for frame_idx, label_file in enumerate(label_files, 1):  # enumerate를 사용하여 프레임 번호 생성
        base_name = os.path.basename(label_file)
        image_name = os.path.splitext(base_name)[0] + ".png"
        image_file = os.path.join(image_path, image_name)
        
        # 이미지 로드
        try:
            image = cv2.imread(image_file)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_file}")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error loading image {image_file}: {str(e)}")
            continue
        
        nodes = []
        edges = []
        next_id = 1
        has_person = False
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                lines.sort(key=lambda x: float(x.strip().split()[0]) if x.strip() else 1)
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    try:
                        values = line.strip().split()
                        if len(values) != 5:
                            print(f"Warning: Skipping invalid line in {label_file}: {line}")
                            continue
                        
                        class_id, x_center, y_center, width, height = map(float, values)
                        class_id = int(class_id)
                        
                        if class_id not in class_mapping:
                            print(f"Warning: Skipping unknown class ID {class_id} in {label_file}")
                            continue
                        
                        # 좌표 변환
                        x = round(x_center * 100, 5)
                        y = round(y_center * 100, 5)
                        
                        object_patch = get_object_patch(gray_image, x_center, y_center, width, height)
                        glcm_features = extract_glcm_features(object_patch)
                        
                        if class_id == 0:
                            has_person = True
                            node_id = "0"
                            # person의 speed와 heading 계산
                            speed, heading = person_tracker.calculate_motion((x, y), node_id, frame_idx)
                        else:
                            node_id = str(next_id)
                            next_id += 1
                            # 정적 객체는 속도와 방향이 0
                            speed, heading = 0, 0
                        
                        node = {
                            "id": node_id,
                            "class": class_mapping[class_id]["class"],
                            "state": class_mapping[class_id]["state"],
                            "x": x,
                            "y": y,
                            "speed": speed,
                            "heading": heading,
                            "visual_features": glcm_features
                        }
                        nodes.append(node)
                        
                    except ValueError as e:
                        print(f"Warning: Error parsing line in {label_file}: {line}")
                        print(f"Error message: {str(e)}")
                        continue
            
            # missing person 처리
            if not has_person:
                nearest_obj = person_tracker.update(nodes, has_person, frame_idx)
                if nearest_obj:
                    # missing person의 speed와 heading도 계산
                    speed, heading = person_tracker.calculate_motion(
                        (nearest_obj["x"], nearest_obj["y"]), "0", frame_idx)
                    
                    missing_person = {
                        "id": "0",
                        "class": "person",
                        "state": "missing",
                        "x": nearest_obj["x"],
                        "y": nearest_obj["y"],
                        "speed": speed,
                        "heading": heading,
                        "visual_features": nearest_obj.get("visual_features", {})
                    }
                    nodes.insert(0, missing_person)
                    nearest_id = nearest_obj["id"]
            else:
                person_tracker.update(nodes, has_person, frame_idx)
                nearest_id = None
            
                    # FC 엣지 생성
            edge_id = 1
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        source_node = nodes[i]
                        target_node = nodes[j]
                        
                        # missing person 관련 엣지 처리
                        if not has_person and (source_node["id"] == "0" or target_node["id"] == "0"):
                            if (source_node["id"] == "0" and target_node["id"] == nearest_id):
                                # person -> nearest object: BACK 관계
                                spatial_relation = SpatialRelation.BACK
                            elif (target_node["id"] == "0" and source_node["id"] == nearest_id):
                                # nearest object -> person: FRONT 관계
                                spatial_relation = SpatialRelation.FRONT
                            else:
                                # 다른 객체들과는 NONE 관계
                                spatial_relation = SpatialRelation.NONE
                        else:
                            # 일반적인 경우의 공간적 관계 계산
                            rel_x = target_node["x"] - source_node["x"]
                            rel_y = target_node["y"] - source_node["y"]
                            distance = calculate_distance(
                                source_node["x"], source_node["y"],
                                target_node["x"], target_node["y"]
                            )
                            spatial_relation = determine_spatial_relation(
                                rel_x, rel_y, distance, distance_threshold)
                        
                        edges.append({
                            "id": f"e{edge_id}",
                            "source": source_node["id"],
                            "target": target_node["id"],
                            "position": spatial_relation.name.lower()
                        })
                        edge_id += 1
        
            # [JSON 저장 부분은 동일]
            
            json_data = {
                "Image": image_name,
                "label": base_name,
                "frame_no": str(frame_idx).zfill(4),  # 4자리 숫자로 포맷팅 (예: "0001", "0002", ...)
                "nodes": nodes,
                "edges": edges
            }
            
            output_file = os.path.join(output_path, f"{os.path.splitext(base_name)[0]}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
                
            print(f"Successfully processed {label_file}")
            
        except Exception as e:
            print(f"Error processing file {label_file}: {str(e)}")
            continue

if __name__ == "__main__":
    image_path = "./images"
    label_path = "./labels"
    output_path = "./output"
    distance_threshold = 30
    
    os.makedirs(output_path, exist_ok=True)
    yolo_to_json(image_path, label_path, output_path, distance_threshold)