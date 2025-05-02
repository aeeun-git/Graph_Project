# Graph_Project: Spatial Relationship Prediction with GAT

<img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/PyG-29BEB0?style=for-the-badge&logo=PyTorch-Geometric&logoColor=white"/>

이 저장소는 **PyTorch Geometric** 기반의 그래프 신경망을 이용해 동적 노드 간의 공간적 관계를 예측하는 파이프라인을 포함합니다.  
핵심 스크립트('Spatial Relationship Prediction.py')에서 GATConv 임베딩과 관계 예측을 수행하고,  
('Walkthroughs.ipynb')을 통해 데이터 전처리, 학습, 평가, 시각화를 단계별로 확인할 수 있습니다.

---

## 📂 디렉토리 구조

```

Graph\_Project/
├── codes/                                 # 보조 스크립트 및 예제 코드
├── Spatial Relationship Prediction.py     # GAT 기반 관계 예측 모델 구현 및 학습·평가 파이프라인
├── Walkthroughs.ipynb                     # 단계별 데모 및 결과 시각화 노트북
└── README.md                              # 프로젝트 설명 (이 파일)

````

---

## 🚀 주요 기능

- **고유 위치 추출 & 매핑**  
  JSON 파일 내 'position' 필드를 스캔해 고유 라벨 생성  
- **커스텀 데이터셋 클래스**  
  'torch_geometric.data.Dataset' 상속 → Data 객체 변환, 노드 특징 정규화  
- **GAT 기반 그래프 모델**  
  두 단계 GATConv로 노드 임베딩 학습 후, 엣지 쌍 임베딩 결합하여 관계 예측  
- **학습·검증·테스트 루프**  
  에폭별 손실 계산, 최적화, 성능 지표(정확도/정밀도/재현율/F1) 제공  
- **배치 콜레이트**  
  'Batch.from_data_list' 활용, 엣지 라벨 배치 단위 출력  
- **시각화 도구**  
  matplotlib으로 예측 결과 및 학습 곡선 플롯

---

## ⚙️ 설치 및 실행

1. **환경 준비**  

   ```bash
   python3.8 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. **의존성 설치**

   ```bash
   pip install torch torchvision torchaudio
   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv  # PyG 설치 가이드 참조
   pip install scikit-learn matplotlib jupyter
   ```

3. **데이터 준비**

   * 'Spatial Relationship Prediction.py' 상단의 'data_root' 경로를
     본인의 JSON 데이터셋 폴더 경로로 변경합니다.

4. **모델 학습 및 평가**

   ```bash
   python "Spatial Relationship Prediction.py"
   ```

5. **노트북 데모 실행**

   ```bash
   jupyter notebook Walkthroughs.ipynb
   ```

---

## 📚 참고 문헌

* 김영진 외, “Spatial Temporal Graph Attention Network for Dynamic Relationship Prediction”, *KIPS Conference*, 2021.
  [https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050137](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050137)
* 이수민 외, “Graph Neural Networks in Scene Understanding”, *DBpia*, 2022.
  [https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050314](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050314)

---

## 📄 학술대회 논문

* "ROS2 기반 TurtleBot3 로봇 제어 및 실습 프레임워크 설계"
  [https://www.manuscriptlink.com/society/kips/conference/ack2024/file/downloadSoConfManuscript/abs/KIPS\_C2024B0256]

---

## 🏷️ 라이선스

MIT © [aeeun-git](https://github.com/aeeun-git)

---

https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050137
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12050314
