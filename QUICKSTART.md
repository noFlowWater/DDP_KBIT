# DDP_KBIT 빠른 시작 가이드

이 가이드는 DDP_KBIT를 처음 사용하는 사용자를 위한 빠른 시작 튜토리얼입니다.

## ⚡ 5분 만에 시작하기

### 1단계: 설치
```bash
# 저장소 클론
git clone <repository-url>
cd DDP_KBIT

# 패키지 설치
pip install -e .
```

### 2단계: 샘플 설정 생성
```bash
# 기본 설정 파일 생성
ddp-kbit --create_sample_config
```

### 3단계: 첫 번째 훈련 실행
```bash
# 기본 설정으로 훈련 시작
ddp-kbit --mode train --config_path sample_config.json
```

## 🎯 기본 사용 예제

### 예제 1: 단일 노드 훈련
```bash
# 가장 간단한 훈련 실행
ddp-kbit --mode train --config_path sample_config.json
```

**예상 출력:**
```
2024-01-01 12:00:00 - INFO - Starting training mode...
2024-01-01 12:00:00 - INFO - Creating Spark session...
2024-01-01 12:00:00 - INFO - Training started...
2024-01-01 12:00:01 - INFO - Epoch 1/5 - Loss: 2.302 - Accuracy: 0.112
...
```

### 예제 2: 실험 실행
```bash
# 다양한 데이터 형식으로 실험
ddp-kbit --mode experiment --experiment_type single
```

**예상 출력:**
```
2024-01-01 12:00:00 - INFO - Running single experiment...
2024-01-01 12:00:00 - INFO - Testing JSON format...
2024-01-01 12:00:01 - INFO - Testing Avro format...
2024-01-01 12:00:02 - INFO - Testing Protobuf format...
2024-01-01 12:00:03 - INFO - Experiment completed!
```

### 예제 3: 다중 실험 (통계 분석)
```bash
# 10번 반복하여 통계 분석
ddp-kbit --mode experiment --experiment_type multiple --iterations 10
```

## 📱 Jupyter Notebook에서 사용

### 노트북 셀 1: 환경 설정
```python
# DDP_KBIT 모듈 경로 설정
from DDP_KBIT.notebook_interface import setup_module_path
setup_module_path()
```

### 노트북 셀 2: 훈련 실행
```python
# 훈련 모드 실행
from DDP_KBIT.main import run_training_mode
run_training_mode(None)
```

### 노트북 셀 3: 실험 실행
```python
# 실험 모드 실행
from DDP_KBIT.main import run_experiment_mode
run_experiment_mode(None)
```

## 🔧 설정 커스터마이징

### 기본 설정 수정
`sample_config.json` 파일을 편집하여 설정을 조정:

```json
{
  "training_config": {
    "epochs": 10,           // 훈련 에포크 수 증가
    "batch_size": 32,       // 배치 크기 줄이기 (메모리 절약)
    "learning_rate": 0.0001 // 학습률 조정
  }
}
```

### Spark 설정 조정
```json
{
  "spark_config": {
    "executor_instances": 4,    // 실행자 인스턴스 수 증가
    "executor_memory": "8g",    // 메모리 증가
    "executor_cores": 4         // 코어 수 증가
  }
}
```

## 📊 결과 확인

### 훈련 결과
- 훈련 손실 및 정확도 그래프
- 체크포인트 파일 저장
- 로그 파일 생성

### 실험 결과
- 다양한 데이터 형식별 성능 비교
- 통계 분석 결과
- 시각화 차트

## 🚨 문제 해결

### 일반적인 오류와 해결책

#### 오류 1: "ModuleNotFoundError: No module named 'DDP_KBIT'"
```bash
# 해결: 패키지 재설치
pip uninstall ddp-kbit
pip install -e .
```

#### 오류 2: "CUDA out of memory"
```json
// 해결: 설정 파일에서 배치 크기 줄이기
{
  "training_config": {
    "batch_size": 16  // 64에서 16으로 줄이기
  }
}
```

#### 오류 3: "Spark configuration error"
```json
// 해결: 로컬 모드로 실행
{
  "spark_config": {
    "master": "local[*]"
  }
}
```

## 📚 다음 단계

### 기본 사용법을 익힌 후:

1. **고급 설정**: `INSTALL.md` 참조
2. **API 문서**: `README.md`의 API Reference 섹션
3. **사용자 정의 모델**: `models/networks.py` 수정
4. **새로운 실험**: `experiments/runner.py` 확장

### 유용한 명령어들
```bash
# 도움말 보기
ddp-kbit --help

# 특정 모드 도움말
ddp-kbit --mode train --help

# 로그 레벨 조정
ddp-kbit --mode train --log_level DEBUG

# 설정 파일 없이 실행
ddp-kbit --mode experiment --experiment_type single
```

## 🎉 축하합니다!

이제 DDP_KBIT의 기본 사용법을 익혔습니다. 

- ✅ 설치 완료
- ✅ 첫 번째 훈련 실행
- ✅ 실험 실행
- ✅ 설정 커스터마이징

더 자세한 내용은 `README.md`와 `INSTALL.md`를 참조하세요!
