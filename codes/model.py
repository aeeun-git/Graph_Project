import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from glob import glob

class TemporalGraphNetworkGAT(nn.Module):
    def __init__(self, feature_dim, hidden_dim, edge_dim, num_frames_predict=5, heads=4):
        super().__init__()
        self.num_frames_predict = num_frames_predict
        
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_frames_predict)
        )

    def forward(self, x, edge_index_seq, edge_feat_seq):
        batch_size = x.size(0)
        num_nodes = x.size(1)
        time_steps = x.size(2)
        
        spatial_embeddings = []

        for t in range(time_steps):
            batch_embeddings = []
            for b in range(batch_size):
                x_t = x[b, :, t]
                edge_index = edge_index_seq[b][t].to(x.device)
                edge_feat = edge_feat_seq[b][t].to(x.device)
                
                edge_feat_embedded = self.edge_embedding(edge_feat)
                h = self.gat1(x_t, edge_index)
                h = torch.relu(h)
                h = self.gat2(h, edge_index)
                batch_embeddings.append(h)
            
            timestep_embedding = torch.stack(batch_embeddings)
            spatial_embeddings.append(timestep_embedding)
        
        spatial_temporal = torch.stack(spatial_embeddings, dim=1)
        spatial_temporal = spatial_temporal.transpose(1, 2)
        spatial_temporal = spatial_temporal.reshape(batch_size * num_nodes, time_steps, -1)
        
        output, _ = self.gru(spatial_temporal)
        final_hidden = output[:, -1, :]
        
        predictions = self.predictor(final_hidden)
        predictions = predictions.view(batch_size, num_nodes, self.num_frames_predict, 2)
        
        return predictions

def load_all_trajectories(base_dir, trajectory_ids=[1, 2, 3, 4, 7, 8], sequence_length=10):
    """
    여러 trajectory 폴더의 그래프 시퀀스를 로드합니다.
    
    Args:
        base_dir: trajectory 폴더들이 있는 기본 디렉토리
        trajectory_ids: 처리할 trajectory 번호 리스트
        sequence_length: 입력으로 사용할 연속된 프레임 수
    """
    all_sequences = []
    
    for traj_id in trajectory_ids:
        print(f"Loading trajectory_{traj_id}...")
        graph_dir = os.path.join(base_dir, f"trajectory_{traj_id}", "graphs")
        
        # 해당 디렉토리가 존재하는지 확인
        if not os.path.exists(graph_dir):
            print(f"Warning: {graph_dir} does not exist. Skipping...")
            continue
            
        # 그래프 파일 목록 가져오기
        graph_files = sorted(glob(os.path.join(graph_dir, "*.gexf")))
        
        if not graph_files:
            print(f"Warning: No graph files found in {graph_dir}. Skipping...")
            continue
            
        print(f"Found {len(graph_files)} graph files in trajectory_{traj_id}")
        
        # 시퀀스 생성
        for i in range(len(graph_files) - sequence_length - 5):
            sequence = []
            future_sequence = []
            
            # 입력 시퀀스 로드
            for j in range(sequence_length):
                G = nx.read_gexf(graph_files[i + j])
                sequence.append(G)
                
            # 미래 프레임 로드
            for j in range(5):
                G = nx.read_gexf(graph_files[i + sequence_length + j])
                future_sequence.append(G)
                
            all_sequences.append({
                'trajectory_id': traj_id,
                'input_sequence': sequence,
                'future_sequence': future_sequence
            })
            
        print(f"Created {len(all_sequences)} sequences from trajectory_{traj_id}")
    
    print(f"\nTotal sequences created: {len(all_sequences)}")
    return all_sequences

def pad_sequence(features, max_nodes):
    """
    시퀀스를 지정된 최대 노드 수에 맞게 패딩합니다.
    """
    current_nodes = features.size(0)
    if current_nodes < max_nodes:
        # [node_num, timesteps, feature_dim] -> [max_nodes, timesteps, feature_dim]
        padding = torch.zeros(max_nodes - current_nodes, *features.size()[1:])
        return torch.cat([features, padding], dim=0)
    return features

def prepare_features(graph_sequence, max_nodes=15):
    sequence = graph_sequence['input_sequence']
    future = graph_sequence['future_sequence']
    
    features = []
    edge_indices = []
    edge_features = []
    
    # 첫 번째 그래프의 노드 수를 사용
    num_nodes = len(sequence[0].nodes())
    
    for G in sequence:
        node_feats = []
        for node in sorted(G.nodes()):
            node_data = G.nodes[node]
            try:
                if isinstance(node_data['visual_features'], str):
                    visual_features = eval(node_data['visual_features'])
                else:
                    visual_features = node_data['visual_features']
                visual_feat = np.array([
                    visual_features['contrast'],
                    visual_features['dissimilarity'],
                    visual_features['homogeneity'],
                    visual_features['energy'],
                    visual_features['correlation']
                ])
            except Exception as e:
                print(f"Error parsing visual features for node {node}: {str(e)}")
                visual_feat = np.zeros(5)
            
            pos = np.array([float(node_data['x'])/100, float(node_data['y'])/100])
            node_feats.append(np.concatenate([visual_feat, pos]))
        
        # 노드 특징을 미리 numpy 배열로 변환
        node_feats = np.array(node_feats)
        
        # 패딩 적용
        if len(node_feats) < max_nodes:
            padding = np.zeros((max_nodes - len(node_feats), node_feats.shape[1]))
            node_feats = np.vstack([node_feats, padding])
        
        features.append(torch.FloatTensor(node_feats))
        
        # 엣지 처리
        edge_idx = []
        edge_feat = []
        for e in G.edges(data=True):
            edge_idx.append([int(e[0]), int(e[1])])
            if e[2]['position'] == 'none':
                edge_feat.append([0] * 8)
            else:
                position_feat = [1 if e[2]['position'] == pos else 0 
                               for pos in ['front', 'back', 'left', 'right', 'front_left', 
                                         'front_right', 'back_left', 'back_right']]
                edge_feat.append(position_feat)
        
        edge_indices.append(torch.LongTensor(edge_idx).t())
        edge_features.append(torch.FloatTensor(edge_feat))
    
    future_positions = []
    for G in future:
        positions = []
        for node in sorted(G.nodes()):
            node_data = G.nodes[node]
            pos = [float(node_data['x']), float(node_data['y'])]
            positions.append(pos)
        
        # 패딩 적용
        while len(positions) < max_nodes:
            positions.append([0, 0])
        
        future_positions.append(positions)
    
    return {
        'features': torch.stack(features).transpose(0, 1),  # [nodes, timesteps, features]
        'edge_index': edge_indices,
        'edge_features': edge_features,
        'future_positions': torch.FloatTensor(future_positions).transpose(0, 1),  # [nodes, timesteps, 2]
        'num_nodes': torch.tensor(num_nodes)  # 스칼라 값
    }



def custom_collate_fn(batch):
    features = torch.stack([item['features'] for item in batch])
    edge_indices = [[tensor for tensor in item['edge_index']] for item in batch]
    edge_features = [[tensor for tensor in item['edge_features']] for item in batch]
    future_positions = torch.stack([item['future_positions'] for item in batch])  # [batch, nodes, timesteps, 1, 2]
    num_nodes = torch.stack([item['num_nodes'] for item in batch])
    
    return {
        'features': features,
        'edge_index': edge_indices,
        'edge_features': edge_features,
        'future_positions': future_positions,
        'num_nodes': num_nodes
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        edge_indices = batch['edge_index']
        edge_features = batch['edge_features']
        future_positions = batch['future_positions'].to(device)
        num_nodes = batch['num_nodes'].to(device)  # [batch_size]
        
        optimizer.zero_grad()
        predictions = model(features, edge_indices, edge_features)
        
        loss = 0
        for i in range(predictions.size(0)):  # batch 크기만큼 반복
            n = int(num_nodes[i])  # 각 배치의 실제 노드 수
            pred = predictions[i, :n]  # [n, num_frames_predict, 2]
            target = future_positions[i, :n]  # [n, num_frames_predict, 2]
            loss += criterion(pred, target)
        loss = loss / predictions.size(0)  # 배치 크기로 나누기
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            edge_indices = batch['edge_index']
            edge_features = batch['edge_features']
            future_positions = batch['future_positions'].to(device)
            num_nodes = batch['num_nodes'].to(device)
            
            predictions = model(features, edge_indices, edge_features)
            
            loss = 0
            for i in range(predictions.size(0)):
                n = int(num_nodes[i])
                pred = predictions[i, :n]
                target = future_positions[i, :n]
                loss += criterion(pred, target)
            loss = loss / predictions.size(0)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            edge_indices = batch['edge_index']
            edge_features = batch['edge_features']
            future_positions = batch['future_positions'].to(device)
            num_nodes = batch['num_nodes'].to(device)
            
            predictions = model(features, edge_indices, edge_features)
            
            # 패딩된 노드 제외하고 실제 노드만 저장
            batch_predictions = []
            batch_targets = []
            for i in range(len(num_nodes)):
                n = int(num_nodes[i].item())
                pred = predictions[i, :n, :, :]
                target = future_positions[i, :n, :, :]
                batch_predictions.append(pred)
                batch_targets.append(target)
                loss = criterion(pred, target)
                total_loss += loss.item()
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
    
    return (total_loss / len(dataloader), 
            torch.cat(all_predictions, dim=0), 
            torch.cat(all_targets, dim=0))

def visualize_trajectories(predictions, targets, num_samples=5):
    output_dir = "../output"  # 저장할 디렉토리 지정
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성
    
    for i in range(min(num_samples, predictions.shape[0])):
        plt.figure(figsize=(8, 8))
        plt.scatter(targets[i, :, 0].cpu().numpy(), targets[i, :, 1].cpu().numpy(), 
                    label="Ground Truth", color="blue", alpha=0.6)
        plt.scatter(predictions[i, :, 0].cpu().numpy(), predictions[i, :, 1].cpu().numpy(), 
                    label="Predictions", color="red", alpha=0.6)
        plt.title(f"Trajectory Prediction Sample {i+1}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        
        # 이미지 저장 (파일명에 index 추가)
        save_path = os.path.join(output_dir, f"trajectory_{i+1}.png")
        plt.savefig(save_path)
        # plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 전체 trajectory 데이터 로드
    print("Loading all trajectory sequences...")
    base_dir = "../"  # 현재 디렉토리 기준
    sequences = load_all_trajectories(base_dir)
    # 데이터가 없는 경우 처리
    if not sequences:
        print("No valid sequences found. Exiting...")
        return
        
    # 데이터 섞기 (다른 trajectory의 데이터가 고르게 분포하도록)
    np.random.shuffle(sequences)
    
    # 데이터셋 분할
    num_sequences = len(sequences)
    train_size = int(0.7 * num_sequences)
    val_size = int(0.15 * num_sequences)
    
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size+val_size]
    test_sequences = sequences[train_size+val_size:]
    
    print(f"\nDataset split:")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    train_data = [prepare_features(seq) for seq in train_sequences]
    val_data = [prepare_features(seq) for seq in val_sequences]
    test_data = [prepare_features(seq) for seq in test_sequences]
    
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    
    feature_dim = train_data[0]['features'].shape[-1]
    model = TemporalGraphNetworkGAT(
        feature_dim=feature_dim,
        hidden_dim=128,
        edge_dim=8,
        num_frames_predict=5,
        heads=4
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("../output/learning curve.png")
    # plt.show()
    
    # 최적 모델로 테스트
    print("\nTesting best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, predictions, targets = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # 예측 결과 시각화
    print("\nVisualizing predictions...")
    visualize_trajectories(predictions, targets)

if __name__ == "__main__":
    main()
