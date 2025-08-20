# DDP_KBIT 설치 및 사용 가이드

## 📋 시스템 요구사항

### 필수 요구사항
- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, Linux, macOS
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 최소 5GB 여유 공간

### 선택적 요구사항
- **GPU**: CUDA 11.0 이상 지원 GPU (NVIDIA)
- **Apache Kafka**: 실시간 데이터 스트리밍용
- **Apache Spark**: 분산 처리용

## 🚀 설치 방법

### 1. 소스코드 다운로드
```bash
# Git을 사용하는 경우
git clone <repository-url>
cd DDP_KBIT

# 또는 압축 파일을 다운로드하여 압축 해제
```

### 2. 가상환경 생성 (권장)
```bash
# Python venv 사용
python -m venv ddp_kbit_env

# Windows에서 가상환경 활성화
ddp_kbit_env\Scripts\activate

# Linux/macOS에서 가상환경 활성화
source ddp_kbit_env/bin/activate
```

### 3. 패키지 설치
```bash
# 개발 모드로 설치 (소스코드 수정 시 자동 반영)
pip install -e .

# 또는 의존성만 설치
pip install -r requirements.txt
```

### 4. 설치 확인
```bash
# 패키지 임포트 테스트
python -c "import DDP_KBIT; print('설치 성공!')"

# 명령줄 도구 확인
ddp-kbit --help
```

## 🔧 초기 설정

### 1. 샘플 설정 파일 생성
```bash
ddp-kbit --create_sample_config
```

이 명령은 `sample_config.json` 파일을 생성합니다.

### 2. 설정 파일 편집
생성된 `sample_config.json` 파일을 편집하여 환경에 맞게 설정:

```json
{
  "spark_config": {
    "master": "local[*]",
    "app_name": "DDP_KBIT_Sample",
    "executor_instances": 2,
    "executor_cores": 2,
    "executor_memory": "4g"
  },
  "training_config": {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001
  },
  "data_config": {
    "kafka_servers": ["localhost:9092"],
    "topic": "mnist_topic",
    "batch_size": 32
  }
}
```

## 📚 사용법

### 기본 사용법

#### 1. 훈련 모드
```bash
# 단일 노드 훈련
ddp-kbit --mode train --config_path sample_config.json

# 분산 훈련
ddp-kbit --mode train --distributed --config_path sample_config.json
```

#### 2. 실험 모드
```bash
# 단일 실험
ddp-kbit --mode experiment --experiment_type single

# 다중 실험 (통계 분석 포함)
ddp-kbit --mode experiment --experiment_type multiple --iterations 10
```

#### 3. 도움말 보기
```bash
# 전체 도움말
ddp-kbit --help

# 특정 모드 도움말
ddp-kbit --mode train --help
ddp-kbit --mode experiment --help
```

### Python 스크립트에서 사용

#### 1. 기본 임포트
```python
from DDP_KBIT.main import run_training_mode, run_experiment_mode
from DDP_KBIT.config import training_config, data_config, spark_config
```

#### 2. 훈련 실행
```python
# 기본 설정으로 훈련 실행
run_training_mode(None)

# 또는 설정 객체 생성
class Args:
    def __init__(self):
        self.config_path = "my_config.json"
        self.distributed = True

args = Args()
run_training_mode(args)
```

#### 3. 실험 실행
```python
# 단일 실험
run_experiment_mode(None)

# 다중 실험
class Args:
    def __init__(self):
        self.experiment_type = "multiple"
        self.iterations = 20

args = Args()
run_experiment_mode(args)
```

### Jupyter Notebook에서 사용

#### 1. 노트북 인터페이스 사용
```python
# DDP_KBIT 노트북 인터페이스 임포트
from DDP_KBIT.notebook_interface import setup_module_path, run_training_mode

# 모듈 경로 설정
setup_module_path()

# 훈련 실행
run_training_mode(None)
```

#### 2. 직접 모듈 사용
```python
# 필요한 모듈들 임포트
from DDP_KBIT.models.networks import create_cnn_model
from DDP_KBIT.training.trainer import main_fn
from DDP_KBIT.utils.spark_utils import create_spark_session

# Spark 세션 생성
spark = create_spark_session(app_name="My_Notebook_App")

# 모델 생성
model = create_cnn_model()
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. 임포트 오류
```bash
# 오류: ModuleNotFoundError: No module named 'DDP_KBIT'
# 해결: 패키지가 올바르게 설치되었는지 확인
pip list | grep DDP_KBIT

# 재설치
pip uninstall ddp-kbit
pip install -e .
```

#### 2. CUDA 메모리 부족
```bash
# 설정 파일에서 배치 크기 줄이기
"batch_size": 32  # 64에서 32로 줄이기

# 또는 GPU 사용 비활성화
"use_gpu": false
```

#### 3. Spark 설정 오류
```bash
# 로컬 모드로 실행
"master": "local[*]"

# 메모리 설정 조정
"executor_memory": "2g"  # 4g에서 2g로 줄이기
```

#### 4. Kafka 연결 오류
```bash
# Kafka 서버 주소 확인
"kafka_servers": ["localhost:9092"]

# 또는 Kafka 없이 실행 (로컬 데이터 사용)
```

### 로그 레벨 조정
```bash
# 디버그 정보 보기
ddp-kbit --mode train --log_level DEBUG

# 경고만 보기
ddp-kbit --mode train --log_level WARNING
```

## 📊 성능 최적화

### 1. GPU 가속
```bash
# CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"

# GPU 메모리 확인
nvidia-smi
```

### 2. Spark 설정 최적화
```json
{
  "spark_config": {
    "executor_instances": 4,
    "executor_cores": 4,
    "executor_memory": "8g",
    "driver_memory": "4g"
  }
}
```

### 3. 배치 크기 조정
```json
{
  "training_config": {
    "batch_size": 128,  # GPU 메모리에 따라 조정
    "gradient_accumulation_steps": 2
  }
}
```

## 🔧 고급 설정

### 1. 사용자 정의 모델
```python
# models/networks.py에 새 모델 추가
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 정의
        
    def forward(self, x):
        # 순전파 로직
        return x

# __init__.py에 등록
__all__ = ['MyCustomModel', 'create_my_custom_model']
```

### 2. 사용자 정의 데이터 변환
```python
# data/transforms.py에 새 변환 함수 추가
def my_custom_transform(data):
    # 변환 로직
    return transformed_data

# data_config.py에 등록
CUSTOM_TRANSFORMS = {
    'my_transform': my_custom_transform
}
```

### 3. 사용자 정의 실험
```python
# experiments/runner.py에 새 실험 함수 추가
def my_custom_experiment():
    # 실험 로직
    return results

# main.py에 등록
elif args.mode == "my_experiment":
    my_custom_experiment()
```

## 📝 개발 환경 설정

### 1. 개발 의존성 설치
```bash
pip install -r requirements.txt[dev]
```

### 2. 코드 포맷팅
```bash
# Black을 사용한 코드 포맷팅
black DDP_KBIT/

# Flake8을 사용한 린팅
flake8 DDP_KBIT/
```

### 3. 테스트 실행
```bash
# pytest를 사용한 테스트
pytest tests/

# 또는 특정 테스트 파일
pytest tests/test_training.py
```

## 🆘 지원 및 문의

### 문제 해결 순서
1. 이 가이드의 문제 해결 섹션 확인
2. 로그 레벨을 DEBUG로 설정하여 상세 정보 확인
3. 설정 파일 검증
4. 의존성 버전 호환성 확인

### 유용한 명령어
```bash
# 패키지 정보 확인
pip show ddp-kbit

# 설치된 패키지 목록
pip list

# Python 경로 확인
python -c "import sys; print(sys.path)"

# 현재 작업 디렉토리 확인
pwd  # Linux/macOS
cd   # Windows
```

### 추가 리소스
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Apache Spark 문서](https://spark.apache.org/docs/)
- [Apache Kafka 문서](https://kafka.apache.org/documentation/)
- [원본 노트북 참조](sparkDL_KBIT_gpu_lightning.ipynb)
