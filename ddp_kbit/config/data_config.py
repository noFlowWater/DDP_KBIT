"""
Data Configuration Module

This module contains all data-related configurations including:
- Kafka settings
- Payload formats and schemas
- Data transformation settings
- MongoDB configurations
- Dataset split configurations

All configuration values are preserved exactly as they appear in the original notebook.
"""

import torch
import numpy as np
from PIL import Image
import io


# Transform Functions
def transform_RawMNISTData(image_bytes):
    """
    RawMNISTData를 텐서로 변환하는 함수.
    
    Args:
        image_bytes: Raw image bytes data
    
    Returns:
        torch.Tensor: 변환된 이미지 텐서.
    """
    from torchvision import transforms

    # 변환 파이프라인 정의
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 이미지 데이터를 바이트로부터 PIL 이미지로 변환
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 'L' 모드로 변환하여 grayscale로 읽기
    
    # transform 파이프라인 적용
    image_tensor = transform_pipeline(image)
    
    return image_tensor


def transform_mongodb_image(image_list):
    """
    MongoDB에서 온 이미지 리스트를 텐서로 변환하는 함수.
    
    Args:
        image_list (list): 이미지 데이터 리스트 (이미 정규화됨, 784개 값)
    
    Returns:
        torch.Tensor: 변환된 이미지 텐서 [1, 28, 28]
    """
    # 리스트를 numpy 배열로 변환하고 적절한 shape으로 변경
    image_array = np.array(image_list, dtype=np.float32)
    
    # 784개 값을 [1, 28, 28] 형태로 reshape
    image_tensor = torch.tensor(image_array).view(1, 28, 28)
    
    return image_tensor


# AVRO Schema Definitions
MNIST_AVRO_SCHEMA_V1 = {
    "type": "record",
    "name": "MNISTImage",
    "fields": [
        {"name": "data", "type": {"type": "array", "items": "float"}},
        {"name": "shape", "type": {"type": "array", "items": "int"}},
        {"name": "label", "type": "int"},
        {"name": "meta", "type": {
            "type": "record",
            "name": "Meta",
            "fields": [
                {"name": "dataset", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "normalization_mean", "type": "float"},
                {"name": "normalization_std", "type": "float"}
            ]
        }}
    ]
}

MNIST_AVRO_SCHEMA_V2 = {
    "type": "record",
    "name": "MNISTImage",
    "fields": [
        {"name": "data", "type": {"type": "array", "items": "float"}},
        {"name": "shape", "type": {"type": "array", "items": "int"}},
        {"name": "label", "type": "int"},
        {"name": "meta", "type": {
            "type": "record",
            "name": "Meta",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "partition", "type": "int"},
                {"name": "timestamp", "type": "long"},
                {"name": "format", "type": "string"},
                {"name": "compression", "type": "string"}
            ]
        }}
    ]
}

# Default schema (as set in the original notebook)
MNIST_AVRO_SCHEMA = MNIST_AVRO_SCHEMA_V2

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': ['155.230.35.200:32100', '155.230.35.213:32100', '155.230.35.215:32100'],
    'data_load_topic': 'kbit-p3r1',
}

# MongoDB Configuration
MONGO_CONFIG = {
    "connection_id": "my-mongo-1",
    "mongo_database": "kbit-db",
    "collection": "mnist_train_avro",  # 기존: "mnist_train"
}

# Payload Configurations
PAYLOAD_CONFIG_PROTOBUF = {
    'protobuf_msg_class': None,  # Would be RawMNISTData if imported
    'transform_data_fn': transform_RawMNISTData,
    'transform_label_fn': None
}

PAYLOAD_CONFIG_JSON = {
    "message_format": "none",  # JSON 형식 사용
    "data_field": "data",
    "label_field": "label",    # label 필드는 동일
    "transform_data_fn": transform_mongodb_image,
    "transform_label_fn": None
}

PAYLOAD_CONFIG_MONGODB_AVRO = {
    "message_format": "mongodb_avro",     # MongoDB Avro 형식 사용
    "avro_schema": MNIST_AVRO_SCHEMA,     # MNISTImage Avro 스키마
    "data_field": "avro_data",            # MongoDB에서 avro_data 필드 사용
    "label_field": "label",               # Avro 내부의 label 필드
    "transform_data_fn": transform_mongodb_image, # 이미지 데이터 변환 함수
    "transform_label_fn": None,
    "schema_fingerprint_field": "schema_fingerprint"  # 스키마 검증용
}

PAYLOAD_CONFIG_AVRO = {
    "message_format": "avro",
    "avro_schema": MNIST_AVRO_SCHEMA,
    "data_field": "data",
    "label_field": "label",
    "transform_data_fn": transform_mongodb_image,
    "transform_label_fn": None,
}

# Default payload config (as set in the original notebook)
PAYLOAD_CONFIG = {
    "message_format": "none",
    "data_field": "data",
    "label_field": "label",
    "transform_data_fn": transform_mongodb_image,
    "transform_label_fn": None,
}

# Data Loader Configuration
DATA_LOADER_CONFIG = {
    "data_loader_type": "kafka",  # "kafka" 또는 "local"
    # "local_data_path": "/root/processed_mnist",  # 로컬 데이터 경로 ( 각 워커의 파일시스템 PATH )
    
    # 만약 data_loader_config 에 "offsets_data", "offsets_data_topic" 키가 있으면, 해당 데이터를 사용하여 데이터셋을 생성
    # 만약 없으면, "api_config" 으로 데이터셋을 생성
    
    # Different topic configurations (commented alternatives from original)
    # "offsets_data": ['0:0:19999', '1:0:19999', '2:0:19999'], # MNIST ( AVRO Lz4 ) 데이터 오프셋 범위
    # "offsets_data_topic": "my-topic-4", # AVRO Lz4
    
    # "offsets_data": ['0:0:19999', '1:0:19999', '2:0:19999'], # MNIST ( AVRO none )  데이터 오프셋 범위
    # "offsets_data_topic": "my-topic-2", # AVRO none
    
    # "offsets_data": ['0:0:19999', '1:0:19999', '2:0:19999'],  # MNIST ( JSON Lz4 )  데이터 오프셋 범위
    # "offsets_data_topic": "my-topic-3",  # JSON Lz4
    
    # "offsets_data": ['0:0:19999', '1:0:19999', '2:0:19999'], # MNIST ( JSON none )  데이터 오프셋 범위
    # "offsets_data_topic": "my-topic", # JSON none
    
    "api_config": {  # 분산 전송 API Config => MongoDB 데이터 조회 방식
        "base_url": "http://155.230.36.25:3001",  # 분산 전송 API 서버 base_url
        "endpoint": "data/export",  # API endpoint
        "params": {  # 분산 전송 API에 필요한 Params들
            "connection_id": MONGO_CONFIG['connection_id'],
            "mongo_database": MONGO_CONFIG['mongo_database'],
            "collection": MONGO_CONFIG['collection'],
            "kafka_brokers": ','.join(KAFKA_CONFIG['bootstrap_servers']),
            "send_topic": KAFKA_CONFIG['data_load_topic'],
            # "drop_last": "F",
        }
    },
    "dataset_split_config": [
        {"rate": 0.857150},  # ``Train 데이터셋`` --> "fillup": True(Default)
        {"rate": 0.071425},  # ``Test 데이터셋`` {"rate": 0.143, "fillup": True},
        {"rate": 0.071425},  # ``Val 데이터셋`` # --> 딕셔너리 요소로 명시하지 않아도 자동으로 "rate": 0.143, "fillup" : True 으로 설정.
    ],
    "consumer_params": {
        'bootstrap_servers': KAFKA_CONFIG['bootstrap_servers'],  # bootstrap_servers옵션은 필수. 나머지는 선택.
        # Other Kafka Consumer Params ..
    },
    "payload_config": PAYLOAD_CONFIG
}

# Alternative Kafka configurations for experiments
ALTERNATIVE_KAFKA_CONFIG = {
    'bootstrap_servers': ['155.230.34.51:32100', '155.230.34.52:32100', '155.230.34.53:32100'],
    'topic': 'mnist-images'
}

# Experiment configurations for different data formats
EXPERIMENT_CONFIGS = [
    {
        "name": "AVRO + LZ4 압축",
        "topic": "my-topic-4",
        "payload_config": {
            "message_format": "avro",
            "avro_schema": MNIST_AVRO_SCHEMA,
            "data_field": "data",
            "label_field": "label",
            "transform_data_fn": transform_mongodb_image,
            "transform_label_fn": None,
        }
    },
    {
        "name": "AVRO + 압축 없음",
        "topic": "my-topic-2",
        "payload_config": {
            "message_format": "avro",
            "avro_schema": MNIST_AVRO_SCHEMA,
            "data_field": "data",
            "label_field": "label",
            "transform_data_fn": transform_mongodb_image,
            "transform_label_fn": None,
        }
    },
    {
        "name": "JSON + LZ4 압축",
        "topic": "my-topic-3",
        "payload_config": {
            "message_format": "none",
            "data_field": "data",
            "label_field": "label",
            "transform_data_fn": transform_mongodb_image,
            "transform_label_fn": None,
        }
    },
    {
        "name": "JSON + 압축 없음",
        "topic": "my-topic",
        "payload_config": {
            "message_format": "none",
            "data_field": "data",
            "label_field": "label",
            "transform_data_fn": transform_mongodb_image,
            "transform_label_fn": None,
        }
    }
]


def get_data_loader_config_for_experiment(experiment_name):
    """
    Get data loader configuration for a specific experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        dict: Data loader configuration for the experiment
    """
    base_config = DATA_LOADER_CONFIG.copy()
    
    for exp_config in EXPERIMENT_CONFIGS:
        if exp_config['name'] == experiment_name:
            base_config['data_loader_type'] = "kafka"
            base_config['offsets_data'] = ['0:0:19999', '1:0:19999', '2:0:19999']
            base_config['offsets_data_topic'] = exp_config['topic']
            base_config['payload_config'] = exp_config['payload_config']
            break
    
    return base_config


def get_data_loader_config_with_topic(topic, payload_config=None):
    """
    Get data loader configuration with specific topic and payload config.
    
    Args:
        topic (str): Kafka topic name
        payload_config (dict, optional): Payload configuration
        
    Returns:
        dict: Data loader configuration
    """
    config = DATA_LOADER_CONFIG.copy()
    config['offsets_data_topic'] = topic
    if payload_config:
        config['payload_config'] = payload_config
    return config