# 테스트 전략

## 테스트 프레임워크
- pytest

## 테스트 구조
```
tests/
├── conftest.py          # 공통 fixture
├── test_*.py            # 유닛 테스트
└── integration/         # 통합 테스트
```

## 실행 방법
```bash
# 전체 테스트 실행
uv run pytest

# 특정 테스트 실행
uv run pytest tests/test_example.py -v

# 커버리지 포함 실행
uv run pytest --cov=src --cov-report=term-missing
```

## 커버리지 기준
- 비즈니스 로직: 80% 이상 목표
- 유틸리티/헬퍼: 필수 경로 테스트

## 테스트 원칙
- 테스트 간 독립성 유지 (공유 상태 금지)
- fixture를 적극 활용
- 외부 의존성은 mock 처리
