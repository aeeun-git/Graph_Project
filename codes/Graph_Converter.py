import json
import networkx as nx
from glob import glob
import os

def json_to_graph(json_file):
    """
    JSON 파일을 NetworkX 그래프로 변환합니다.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 방향성 그래프 생성
    G = nx.DiGraph()
    
    # 그래프 메타데이터 추가
    G.graph['image'] = data['Image']
    G.graph['label'] = data['label']
    G.graph['frame_no'] = data['frame_no']
    
    # 노드 추가
    for node in data['nodes']:
        node_attrs = {
            'class_name': node['class'],
            'state': node['state'],
            'x': node['x'],
            'y': node['y'],
            'speed': node['speed'],
            'heading': node['heading']
        }
        # 시각적 특징이 있는 경우 추가
        if 'visual_features' in node:
            node_attrs['visual_features'] = node['visual_features']
        
        G.add_node(node['id'], **node_attrs)
    
    # 엣지 추가
    for edge in data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            id=edge['id'],
            position=edge['position']
        )
    
    return G

def save_graph(G, output_dir, format='gexf'):
    """
    그래프를 지정된 형식으로 저장합니다.
    
    Args:
        G (nx.DiGraph): 저장할 그래프
        output_dir (str): 저장할 디렉토리 경로
        format (str): 저장 형식 ('gexf', 'graphml', 'gml', 'pkl' 중 하나)
    """
    frame_no = G.graph.get('frame_no', '0000')
    base_name = f"graph_{frame_no}"
    
    if format == 'gexf':
        nx.write_gexf(G, os.path.join(output_dir, f"{base_name}.gexf"))
    elif format == 'graphml':
        nx.write_graphml(G, os.path.join(output_dir, f"{base_name}.graphml"))
    elif format == 'gml':
        nx.write_gml(G, os.path.join(output_dir, f"{base_name}.gml"))
    elif format == 'pkl':
        nx.write_gpickle(G, os.path.join(output_dir, f"{base_name}.pkl"))
    else:
        raise ValueError(f"Unsupported format: {format}")

def process_json_files(json_dir, graph_dir, format='gexf'):
    """
    JSON 파일들을 처리하여 그래프로 변환하고 저장합니다.
    
    Args:
        json_dir (str): JSON 파일들이 있는 디렉토리 경로
        graph_dir (str): 그래프를 저장할 디렉토리 경로
        format (str): 그래프 저장 형식
    """
    # 출력 디렉토리 생성
    os.makedirs(graph_dir, exist_ok=True)
    
    # JSON 파일 목록 가져오기
    json_files = sorted(glob(os.path.join(json_dir, "*.json")))
    
    for json_file in json_files:
        try:
            # JSON을 그래프로 변환
            G = json_to_graph(json_file)
            
            # 그래프 저장
            save_graph(G, graph_dir, format)
            
            print(f"Successfully processed and saved graph for {json_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # 디렉토리 설정
    json_dir = "./output"  # JSON 파일이 있는 디렉토리
    graph_dir = "./graphs"  # 그래프를 저장할 디렉토리
    
    # JSON 파일들을 처리하여 그래프로 변환하고 저장
    process_json_files(json_dir, graph_dir, format='gexf')
    
    print(f"\nAll graphs have been saved to {graph_dir}")