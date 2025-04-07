# -----------------------------
# 관계 예측 :
# - 동적 노드를 타겟으로 하고, 타겟을 예측하기 위한 모델로 GAT 임베딩 + 관계 예측을 수행함
#
# 데이터 디렉토리
# data_root = '/content/drive/MyDrive/MyProj/04-ST-GAT/data'
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import json
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1. 고유한 position 값 추출 및 매핑 생성 함수
# -----------------------------
def get_unique_positions(root):
    """
    데이터셋의 모든 JSON 파일을 읽어 고유한 position 값을 추출함.

    Parameters:
    - root (str): 데이터셋이 저장된 루트 디렉토리 경로임.

    Returns:
    - Set[str]: 고유한 position 값의 집합임.
    """
    unique_positions = set()
    for file in os.listdir(root):
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data_json = json.load(f)
            for edge in data_json['edges']:
                position = edge.get('position', 'none').replace('-', '_').upper()
                unique_positions.add(position)
    return unique_positions

def create_relation_mapping(unique_positions):
    """
    고유한 position 값을 기반으로 관계 매핑을 생성함.

    Parameters:
    - unique_positions (Set[str]): 고유한 position 값의 집합임.

    Returns:
    - Dict[str, int]: position 문자열을 정수 라벨로 매핑한 딕셔너리임.
    """
    sorted_positions = sorted(unique_positions)  # 일관성을 위해 정렬
    relation_mapping = {position: idx for idx, position in enumerate(sorted_positions)}
    return relation_mapping

# -----------------------------
# 2. 실제 데이터셋 클래스 구현
# -----------------------------
class RealGraphDataset(Dataset):
    """
    실제 데이터를 읽어와 torch_geometric의 Data 객체로 변환하는 데이터셋 클래스임.
    """
    def __init__(self, root, relation_mapping, transform=None, pre_transform=None):
        """
        데이터셋 초기화 함수임.

        Parameters:
        - root (str): 데이터셋이 저장된 루트 디렉토리 경로임.
        - relation_mapping (Dict[str, int]): position 문자열을 정수 라벨로 매핑한 딕셔너리임.
        - transform (callable, optional): 데이터 변환 함수임.
        - pre_transform (callable, optional): 사전 변환 함수임.
        """
        super().__init__(root, transform, pre_transform)
        self.data_files = self._load_data_files()
        self.relation_mapping = relation_mapping

        # 노드 특징 정규화를 위해 스케일러 초기화
        self.scaler = StandardScaler()
        all_features = []
        for file in self.data_files:
            with open(file, 'r') as f:
                data_json = json.load(f)
            for node in data_json['nodes']:
                features = [
                    node['visual_features']['contrast'],
                    node['visual_features']['dissimilarity'],
                    node['visual_features']['homogeneity'],
                    node['visual_features']['energy'],
                    node['visual_features']['correlation']
                ]
                all_features.append(features)
        self.scaler.fit(all_features)

    def _load_data_files(self):
        """
        데이터 디렉토리 내의 모든 JSON 파일을 로드함.

        Returns:
        - List[str]: 데이터 파일 경로 리스트임.
        """
        data_files = []
        for file in os.listdir(self.root):
            if file.endswith('.json'):
                data_files.append(os.path.join(self.root, file))
        # 파일 이름을 프레임 번호 순으로 정렬함
        data_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        return data_files

    def len(self):
        """
        데이터셋의 길이를 반환함.
        """
        return len(self.data_files)

    def get(self, idx):
        """
        주어진 인덱스에 해당하는 데이터를 반환함.

        Parameters:
        - idx (int): 데이터 인덱스임.

        Returns:
        - Data: torch_geometric의 Data 객체임.
        """
        file_path = self.data_files[idx]
        with open(file_path, 'r') as f:
            data_json = json.load(f)

        nodes = data_json['nodes']
        edges = data_json['edges']

        # 노드 특징 추출 (시각적 특징만 사용함)
        node_features = []
        node_id_map = {}  # ID를 인덱스로 매핑함
        for i, node in enumerate(nodes):
            node_id_map[node['id']] = i
            features = [
                node['visual_features']['contrast'],
                node['visual_features']['dissimilarity'],
                node['visual_features']['homogeneity'],
                node['visual_features']['energy'],
                node['visual_features']['correlation']
            ]
            node_features.append(features)

        # 노드 특징 정규화
        x = self.scaler.transform(node_features)
        x = torch.tensor(x, dtype=torch.float)  # [num_nodes, feature_dim]

        # 엣지 인덱스와 관계 라벨 추출
        edge_index = []
        edge_relations = []
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            source_idx = node_id_map[source_id]
            target_idx = node_id_map[target_id]

            # 엣지의 position 값을 직접 사용하여 관계 라벨 설정
            position = edge['position'].replace('-', '_').upper()
            relation_label = self.relation_mapping.get(position, self.relation_mapping.get('NONE', -1))

            # 관계가 'NONE'이 아닌 경우에만 엣지를 추가
            if relation_label != self.relation_mapping.get('NONE', -1):
                edge_index.append([source_idx, target_idx])
                edge_relations.append(relation_label)

        if not edge_index:
            # 엣지가 하나도 없을 경우, 더미 엣지를 추가함
            edge_index = [[0, 0]]
            edge_relations = [self.relation_mapping.get('NONE', -1)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_relations = torch.tensor(edge_relations, dtype=torch.long)          # [num_edges]

        data = Data(x=x, edge_index=edge_index, edge_label=edge_relations)

        return data

# -----------------------------
# 3. 커스텀 콜레이트 함수 수정
# -----------------------------
def custom_collate_fn(batch):
    """
    배치를 만들 때 각 샘플의 데이터를 어떻게 합칠지 정의하는 함수임.

    Parameters:
    - batch (List[Data]): Data 객체 리스트임.

    Returns:
    - dict:
        - 'batch_graph': Batched Data object from torch_geometric임.
        - 'relation_labels': Tensor, [total_edges] 엣지의 관계 라벨임.
    """
    batch_graph = Batch.from_data_list(batch)
    relation_labels = batch_graph.edge_label  # [total_edges]
    return {
        'batch_graph': batch_graph,
        'relation_labels': relation_labels
    }

# -----------------------------
# 4. Temporal Graph Network 모델 클래스
# -----------------------------
class TemporalGraphNetworkGAT_EdgeRelation(nn.Module):
    """
    GAT을 기반으로 한 그래프 네트워크 모델 클래스임.
    엣지 간의 관계를 예측함.
    """
    def __init__(self, feature_dim, hidden_dim, heads=4, num_relations=9):
        """
        모델의 초기화 함수임.

        Parameters:
        - feature_dim (int): 각 노드의 입력 특징 차원임.
        - hidden_dim (int): GAT 레이어의 히든 차원임.
        - heads (int): GAT 레이어의 헤드 수임.
        - num_relations (int): 예측할 관계의 종류 수임.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 첫 번째 GAT 레이어: 입력 차원 -> 히든 차원임.
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=False)
        # 두 번째 GAT 레이어: 히든 차원 -> 히든 차원임.
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

        # 관계 예측을 위한 선형 레이어임.
        self.relation_predictor = nn.Linear(hidden_dim * 2, num_relations)  # 관계 종류 수 만큼 출력함.

    def forward(self, batch_graph):
        """
        모델의 순전파 함수임.

        Parameters:
        - batch_graph (Batch): torch_geometric의 Batched Data 객체임.

        Returns:
        - relation_preds (Tensor): [total_edges, num_relations] 예측된 관계 확률임.
        """
        # GAT 레이어를 통과시켜 노드 임베딩을 계산함.
        x = self.gat1(batch_graph.x, batch_graph.edge_index)  # [total_nodes, hidden_dim]임.
        x = torch.relu(x)
        x = self.gat2(x, batch_graph.edge_index)             # [total_nodes, hidden_dim]임.
        x = torch.relu(x)

        # 관계 예측
        # 소스 노드와 타겟 노드의 임베딩을 결합하여 관계를 예측함.
        src, dst = batch_graph.edge_index  # [2, total_edges]임.
        src_emb = x[src]                   # [total_edges, hidden_dim]임.
        dst_emb = x[dst]                   # [total_edges, hidden_dim]임.
        relation_input = torch.cat([src_emb, dst_emb], dim=1)  # [total_edges, hidden_dim * 2]임.
        relation_preds = self.relation_predictor(relation_input)  # [total_edges, num_relations]임.

        return relation_preds

# -----------------------------
# 5. 학습, 검증, 테스트 함수
# -----------------------------
def train_epoch(model, dataloader, relation_criterion, optimizer, device):
    """
    한 에폭 동안 모델을 학습시키는 함수임.

    Parameters:
    - model (nn.Module): 학습할 모델임.
    - dataloader (DataLoader): 학습 데이터 로더임.
    - relation_criterion (nn.Module): 관계 예측 손실 함수임.
    - optimizer (torch.optim.Optimizer): 옵티마이저임.
    - device (torch.device): 학습 디바이스 (CPU 또는 GPU)임.

    Returns:
    - tuple: (평균 관계 손실, )
    """
    model.train()  # 모델을 학습 모드로 설정함.
    total_relation_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # 배치 데이터를 디바이스로 이동시킴.
        batch_graph = batch['batch_graph'].to(device)               # Batched Data object임.
        relation_labels = batch['relation_labels'].to(device)       # [total_edges]임.

        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화함.

        # 모델에 입력하여 예측값을 얻음.
        relation_preds = model(batch_graph)                        # [total_edges, num_relations]임.

        # 관계 예측 손실 계산함.
        loss_relation = relation_criterion(relation_preds, relation_labels)

        # 손실 역전파 수행함.
        loss_relation.backward()
        optimizer.step()  # 옵티마이저를 통해 가중치 업데이트함.

        # 손실 누적함.
        total_relation_loss += loss_relation.item()

    # 에폭당 평균 손실 계산함.
    avg_relation_loss = total_relation_loss / len(dataloader)
    return avg_relation_loss,

def validate(model, dataloader, relation_criterion, device):
    """
    모델을 검증 모드로 설정하고 검증 데이터를 통해 손실을 계산하는 함수임.

    Parameters:
    - model (nn.Module): 검증할 모델임.
    - dataloader (DataLoader): 검증 데이터 로더임.
    - relation_criterion (nn.Module): 관계 예측 손실 함수임.
    - device (torch.device): 검증 디바이스 (CPU 또는 GPU)임.

    Returns:
    - tuple: (평균 관계 손실, )
    """
    model.eval()  # 모델을 평가 모드로 설정함.
    total_relation_loss = 0

    with torch.no_grad():  # 역전파 계산 비활성화함.
        for batch_idx, batch in enumerate(dataloader):
            # 배치 데이터를 디바이스로 이동시킴.
            batch_graph = batch['batch_graph'].to(device)
            relation_labels = batch['relation_labels'].to(device)

            # 모델에 입력하여 예측값을 얻음.
            relation_preds = model(batch_graph)  # [total_edges, num_relations]임.

            # 관계 예측 손실 계산함.
            loss_relation = relation_criterion(relation_preds, relation_labels)

            # 손실 누적함.
            total_relation_loss += loss_relation.item()

    # 에폭당 평균 손실 계산함.
    avg_relation_loss = total_relation_loss / len(dataloader)
    return avg_relation_loss,

def test_model_final(model, dataloader, relation_criterion, device):
    """
    모델을 테스트 모드로 설정하고 테스트 데이터를 통해 손실을 계산하는 함수임.

    Parameters:
    - model (nn.Module): 테스트할 모델임.
    - dataloader (DataLoader): 테스트 데이터 로더임.
    - relation_criterion (nn.Module): 관계 예측 손실 함수임.
    - device (torch.device): 테스트 디바이스 (CPU 또는 GPU)임.

    Returns:
    - tuple:
        - (평균 관계 손실, )
    """
    model.eval()  # 모델을 평가 모드로 설정함.
    total_relation_loss = 0

    with torch.no_grad():  # 역전파 계산 비활성화함.
        for batch_idx, batch in enumerate(dataloader):
            # 배치 데이터를 디바이스로 이동시킴.
            batch_graph = batch['batch_graph'].to(device)
            relation_labels = batch['relation_labels'].to(device)

            # 모델에 입력하여 예측값을 얻음.
            relation_preds = model(batch_graph)  # [total_edges, num_relations]임.

            # 관계 예측 손실 계산함.
            loss_relation = relation_criterion(relation_preds, relation_labels)

            # 손실 누적함.
            total_relation_loss += loss_relation.item()

    # 에폭당 평균 손실 계산함.
    avg_relation_loss = total_relation_loss / len(dataloader)
    return (avg_relation_loss, )

# -----------------------------
# 6. 추가 평가 지표 함수
# -----------------------------
def evaluate_metrics(model, dataloader, device):
    """
    모델의 성능을 평가하기 위한 추가적인 지표를 계산하는 함수임.

    Parameters:
    - model (nn.Module): 평가할 모델임.
    - dataloader (DataLoader): 평가 데이터 로더임.
    - device (torch.device): 평가 디바이스 (CPU 또는 GPU)임.

    Returns:
    - Dict[str, float]: 정확도, 정밀도, 재현율, F1 스코어
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch_graph = batch['batch_graph'].to(device)
            relation_labels = batch['relation_labels'].to(device)
            relation_preds = model(batch_graph)
            preds = torch.argmax(relation_preds, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(relation_labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
def visualize_predictions(model, dataloader, relation_mapping, device):
    """
    모델의 예측 결과를 시각적으로 출력하는 함수입니다.
    
    Parameters:
    - model: 학습된 모델
    - dataloader: 테스트용 데이터로더
    - relation_mapping: position 문자열과 정수 라벨 간의 매핑 딕셔너리
    - device: 모델이 있는 디바이스 (CPU 또는 GPU)
    """
    # relation_mapping을 역매핑하여 인덱스->관계이름으로 변환할 수 있게 함
    reverse_mapping = {v: k for k, v in relation_mapping.items()}
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    confidence_scores = []
    
    print("\n=== 예측 결과 상세 분석 ===\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_graph = batch['batch_graph'].to(device)
            true_labels = batch['relation_labels']
            
            # 모델 예측
            relation_preds = model(batch_graph)
            probabilities = torch.softmax(relation_preds, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            
            # 결과 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.numpy())
            confidence_scores.extend(confidence.cpu().numpy())
            
            # 배치별 상세 결과 출력
            print(f"\n배치 {batch_idx + 1} 분석:")
            print("-" * 50)
            print(f"{'실제 관계':<15} {'예측 관계':<15} {'신뢰도':>10}")
            print("-" * 50)
            
            for true, pred, conf in zip(true_labels, predictions.cpu(), confidence):
                true_relation = reverse_mapping[true.item()]
                pred_relation = reverse_mapping[pred.item()]
                correct = "✓" if true.item() == pred.item() else "✗"
                
                print(f"{true_relation:<15} {pred_relation:<15} {conf.item():>10.2%} {correct}")
    
    # 전체 통계 계산
    correct_predictions = sum(1 for true, pred in zip(all_true_labels, all_predictions) if true == pred)
    total_predictions = len(all_true_labels)
    accuracy = correct_predictions / total_predictions
    
    # 관계별 성능 분석
    relation_performance = {}
    for relation_name, relation_id in relation_mapping.items():
        mask = [label == relation_id for label in all_true_labels]
        if not any(mask):  # 해당 관계가 테스트셋에 없는 경우
            continue
            
        relation_predictions = [pred == relation_id for pred in all_predictions]
        relation_true = sum(1 for m, pred in zip(mask, all_predictions) if m and pred == relation_id)
        relation_total = sum(mask)
        relation_accuracy = relation_true / relation_total if relation_total > 0 else 0
        
        relation_performance[relation_name] = {
            "accuracy": relation_accuracy,
            "total_samples": relation_total,
            "correct_predictions": relation_true
        }
    
    # 종합 결과 출력
    print("\n=== 종합 성능 분석 ===")
    print(f"\n전체 정확도: {accuracy:.2%}")
    print(f"평균 신뢰도: {sum(confidence_scores) / len(confidence_scores):.2%}")
    
    print("\n관계별 성능:")
    print("-" * 60)
    print(f"{'관계 유형':<15} {'정확도':>10} {'샘플 수':>10} {'정답 수':>10}")
    print("-" * 60)
    
    for relation, stats in relation_performance.items():
        print(f"{relation:<15} {stats['accuracy']:>10.2%} {stats['total_samples']:>10d} {stats['correct_predictions']:>10d}")

# -----------------------------
# 7. 메인 함수
# -----------------------------
def main():
    """
    전체적인 데이터 로딩부터 학습, 평가, 시각화까지 담당하는 메인 함수임.
    """
    # 디바이스 설정 (GPU가 사용 가능하면 GPU 사용, 아니면 CPU 사용함.)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 하이퍼파라미터 설정함.
    feature_dim = 5                # 각 노드의 특징 차원임 (visual_features만 사용함).
    hidden_dim = 128               # GAT 레이어의 히든 차원임.
    heads = 4                      # GAT 레이어의 헤드 수임.
    batch_size = 32                # 배치 크기임. 데이터 양에 따라 조정함.
    num_epochs = 80                # 학습 에폭 수임.
    learning_rate = 0.001          # 학습률임.

    # 데이터 디렉토리 경로 설정
    data_root = './root_dir'

    # 고유한 position 값 추출
    unique_positions = get_unique_positions(data_root)
    print(f"Unique positions found: {unique_positions}")

    # 관계 매핑 생성
    relation_mapping = create_relation_mapping(unique_positions)
    print(f"Relation mapping: {relation_mapping}")

    # 실제 데이터셋 초기화함.
    dataset = RealGraphDataset(root=data_root, relation_mapping=relation_mapping)
    print(f"Total samples: {len(dataset)}")

    # 데이터 분할: 훈련, 검증, 테스트 (예: 80%, 10%, 10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # 클래스 불균형 처리: 가중치 계산
    all_relation_labels = []
    for data in dataset:
        all_relation_labels.extend(data.edge_label.tolist())
    label_counts = Counter(all_relation_labels)
    total = sum(label_counts.values())
    num_relations = len(relation_mapping)
    class_weights = [total / label_counts.get(i, 1) for i in range(num_relations)]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 데이터 로더 생성함.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    # 모델 초기화함.
    model = TemporalGraphNetworkGAT_EdgeRelation(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_relations=num_relations
    ).to(device)  # 모델을 디바이스로 이동시킴.

    # 손실 함수 및 옵티마이저 설정함.
    relation_criterion = nn.CrossEntropyLoss(weight=class_weights)         # 관계 예측 손실: Cross Entropy Loss 사용함.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저 사용함.

    # 학습 루프 시작함.
    print("Starting training...")
    # 손실을 기록할 리스트 초기화함.
    train_relation_loss_history = []
    val_relation_loss_history = []

    for epoch in range(num_epochs):
        # 한 에폭 동안 모델을 학습시킴.
        train_rel_loss, = train_epoch(
            model, train_loader, relation_criterion, optimizer, device
        )
        # 한 에폭 동안 모델을 검증함.
        val_rel_loss, = validate(
            model, val_loader, relation_criterion, device
        )

        # 손실 기록함.
        train_relation_loss_history.append(train_rel_loss)
        val_relation_loss_history.append(val_rel_loss)

        # 주기적으로 현재 상태 출력함 (예: 5 에폭마다).
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Rel Loss = {train_rel_loss:.4f}, "
                  f"Val Rel Loss = {val_rel_loss:.4f}")

    print("Training completed!")

    # # 학습 및 검증 손실 히스토리 시각화함.
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_relation_loss_history, label='Train Relation Loss')
    # plt.plot(val_relation_loss_history, label='Validation Relation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Relation Prediction Loss History')
    # plt.legend()
    # plt.show()


    # 학습 및 검증 손실 히스토리 시각화함.
    plt.figure(figsize=(10, 6))
    plt.plot(train_relation_loss_history, label='Train Relation Loss')
    plt.plot(val_relation_loss_history, label='Validation Relation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Relation Prediction Loss History')
    plt.legend()


    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 이미지 저장 (result 폴더 안에 loss_history.png로 저장)
    save_path = os.path.join(result_dir, 'loss_history.png')
    plt.savefig(save_path, dpi=300)
    print(f"Loss history plot saved as '{save_path}'.")


    # 테스트 데이터셋에서 모델을 평가함.
    print("\nEvaluating on test set...")
    test_relation_loss, = test_model_final(
        model, test_loader, relation_criterion, device
    )
    print(f"Test Relation Loss: {test_relation_loss:.4f}")

    # 추가 평가 지표 계산
    metrics = evaluate_metrics(model, test_loader, device)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")

    # 모델 저장 (선택 사항)
    torch.save(model.state_dict(), 'relation_model.pth')
    print("Model saved as 'relation_model.pth'.")

    print("\n테스트 세트에 대한 예측 결과 분석:")
    visualize_predictions(model, test_loader, relation_mapping, device)

# -----------------------------
# 8. 스크립트 시작점
# -----------------------------
if __name__ == "__main__":
    main()
