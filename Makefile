# Makefile for AI Voice Assistant project

.PHONY: protos test clean install dev

# Generate protobuf stubs
protos:
	python scripts/generate_protos.py

# Run tests
test:
	python -m pytest tests/ -v

# Run specific test files
test-proto:
	python -m pytest tests/test_proto.py -v

test-registry:
	python -m pytest tests/test_registry.py -v

test-config:
	python -m pytest tests/test_config.py -v

# Clean generated files
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find services/proto -name "*_pb2.py" -delete
	find services/proto -name "*_pb2_grpc.py" -delete
	find services/proto -name "*_pb2.pyi" -delete

# Install dependencies
install:
	uv sync

# Development setup
dev: install protos
	@echo "Development environment ready!"

# Lint code
lint:
	python -m flake8 services/ tests/ --max-line-length=100 --ignore=E203,W503

# Format code
format:
	python -m black services/ tests/ --line-length=100

# Type check
typecheck:
	python -m mypy services/ --ignore-missing-imports

# Run all checks
check: lint typecheck test

# Help
help:
	@echo "Available targets:"
	@echo "  protos     - Generate protobuf stubs"
	@echo "  test       - Run all tests"
	@echo "  test-*     - Run specific test files"
	@echo "  clean      - Clean generated files"
	@echo "  install    - Install dependencies"
	@echo "  dev        - Set up development environment"
	@echo "  lint       - Lint code"
	@echo "  format     - Format code"
	@echo "  typecheck  - Type check code"
	@echo "  check      - Run all checks"
	@echo "  help       - Show this help"