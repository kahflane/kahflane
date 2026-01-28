#!/bin/bash
# Run all tests with real services
#
# Usage:
#   ./scripts/run-tests.sh                  # Start services, run tests, stop services
#   ./scripts/run-tests.sh --no-down        # Keep services running after tests
#   ./scripts/run-tests.sh --only           # Run tests only (assumes services running)
#   ./scripts/run-tests.sh --cov            # Run with coverage report
#   ./scripts/run-tests.sh -- -k "redis"    # Pass extra args to pytest after --

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.test.yml"

# Detect docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif docker-compose --version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' found"
    exit 1
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
NO_DOWN=false
ONLY_TESTS=false
WITH_COV=false
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-down)
            NO_DOWN=true
            shift
            ;;
        --only)
            ONLY_TESTS=true
            shift
            ;;
        --cov)
            WITH_COV=true
            shift
            ;;
        --)
            shift
            PYTEST_ARGS+=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Start services if not --only
if [ "$ONLY_TESTS" = false ]; then
    log_info "Starting test services..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d

    log_info "Waiting for services to be healthy..."

    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL (port 5433)..."
    for i in {1..30}; do
        if $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T postgres-test pg_isready -U kahflane_test >/dev/null 2>&1; then
            log_info "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "PostgreSQL failed to start"
            exit 1
        fi
        sleep 1
    done

    # Wait for Redis
    log_info "Waiting for Redis (port 6380)..."
    for i in {1..30}; do
        if $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T redis-test redis-cli ping >/dev/null 2>&1; then
            log_info "Redis is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Redis failed to start"
            exit 1
        fi
        sleep 1
    done

    # Wait for Qdrant
    log_info "Waiting for Qdrant (port 6334)..."
    for i in {1..30}; do
        if curl -s http://localhost:6334/healthz >/dev/null 2>&1; then
            log_info "Qdrant is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Qdrant failed to start"
            exit 1
        fi
        sleep 1
    done

    log_info "All services are ready"
fi

# Set environment variables
export DB_HOST=localhost
export DB_PORT=5433
export DB_USER=kahflane_test
export DB_PASSWORD=kahflane_test
export DB_NAME=kahflane_test
export REDIS_HOST=localhost
export REDIS_PORT=6380
export QDRANT_HOST=localhost
export QDRANT_PORT=6334
export QDRANT_KEY=""

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Build pytest command
PYTEST_CMD=(pytest tests/ -v --tb=short)

if [ "$WITH_COV" = true ]; then
    PYTEST_CMD+=(--cov=app --cov-report=term-missing --cov-report=html)
fi

PYTEST_CMD+=("${PYTEST_ARGS[@]}")

log_info "Running tests: ${PYTEST_CMD[*]}"
"${PYTEST_CMD[@]}"
TEST_EXIT_CODE=$?

# Stop services unless --no-down
if [ "$NO_DOWN" = false ] && [ "$ONLY_TESTS" = false ]; then
    log_info "Stopping test services..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" down
fi

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_info "All tests passed!"
else
    log_error "Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

exit $TEST_EXIT_CODE
