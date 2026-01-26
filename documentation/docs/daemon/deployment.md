---
title: Production Deployment
description: Systemd, Docker, reverse proxy, monitoring, and security for production Mullama deployments
---

# Production Deployment

This guide covers deploying the Mullama daemon in production environments with proper process management, security, monitoring, and high availability.

---

## Systemd Service

### Service File

Create `/etc/systemd/system/mullama.service`:

```ini
[Unit]
Description=Mullama LLM Inference Daemon
Documentation=https://docs.neullabs.com/mullama/daemon/
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=mullama
Group=mullama
WorkingDirectory=/opt/mullama

# Binary and arguments
ExecStart=/opt/mullama/bin/mullama serve \
    --model llama3.2:1b \
    --model qwen2.5:7b \
    --http-port 8080 \
    --http-addr 127.0.0.1 \
    --gpu-layers 35 \
    --context-size 8192 \
    --socket ipc:///var/run/mullama/mullama.sock

# Graceful shutdown
ExecStop=/opt/mullama/bin/mullama daemon stop --socket ipc:///var/run/mullama/mullama.sock
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Environment
Environment=MULLAMA_CACHE_DIR=/opt/mullama/models
Environment=MULLAMA_LOG_LEVEL=info
Environment=HF_TOKEN=

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/mullama/models /var/run/mullama /var/log/mullama
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictNamespaces=true

# Resource limits
LimitNOFILE=65536
MemoryMax=32G
CPUQuota=800%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mullama

# Runtime directory
RuntimeDirectory=mullama
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
```

### Setup Commands

```bash
# Create user and directories
sudo useradd --system --shell /bin/false --home-dir /opt/mullama mullama
sudo mkdir -p /opt/mullama/{bin,models}
sudo mkdir -p /var/run/mullama
sudo mkdir -p /var/log/mullama

# Copy binary
sudo cp target/release/mullama /opt/mullama/bin/
sudo chown -R mullama:mullama /opt/mullama

# Pre-download models
sudo -u mullama MULLAMA_CACHE_DIR=/opt/mullama/models /opt/mullama/bin/mullama pull llama3.2:1b
sudo -u mullama MULLAMA_CACHE_DIR=/opt/mullama/models /opt/mullama/bin/mullama pull qwen2.5:7b

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mullama
sudo systemctl start mullama

# Check status
sudo systemctl status mullama
sudo journalctl -u mullama -f
```

### GPU Access

For NVIDIA GPU access, add the user to the appropriate group and allow device access:

```ini
# Add to [Service] section
SupplementaryGroups=video render
DeviceAllow=/dev/nvidia0 rw
DeviceAllow=/dev/nvidiactl rw
DeviceAllow=/dev/nvidia-uvm rw
DeviceAllow=/dev/nvidia-uvm-tools rw
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM rust:1.75-bookworm AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build mullama
WORKDIR /src
COPY . .
RUN git submodule update --init --recursive
RUN cargo build --release --features daemon

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --shell /bin/false mullama
RUN mkdir -p /opt/mullama/models && chown -R mullama:mullama /opt/mullama

COPY --from=builder /src/target/release/mullama /usr/local/bin/mullama

USER mullama
WORKDIR /opt/mullama

ENV MULLAMA_CACHE_DIR=/opt/mullama/models
ENV MULLAMA_LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["mullama"]
CMD ["serve", "--http-addr", "0.0.0.0", "--http-port", "8080"]
```

### Dockerfile with CUDA

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
COPY . .
RUN git submodule update --init --recursive

ENV LLAMA_CUDA=1
RUN cargo build --release --features daemon

# Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --shell /bin/false mullama
RUN mkdir -p /opt/mullama/models && chown -R mullama:mullama /opt/mullama

COPY --from=builder /src/target/release/mullama /usr/local/bin/mullama

USER mullama
WORKDIR /opt/mullama

ENV MULLAMA_CACHE_DIR=/opt/mullama/models
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["mullama"]
CMD ["serve", "--http-addr", "0.0.0.0", "--http-port", "8080", "--gpu-layers", "99"]
```

### Docker Run

```bash
# CPU only
docker build -t mullama .
docker run -d \
    --name mullama \
    -p 8080:8080 \
    -v mullama-models:/opt/mullama/models \
    mullama \
    serve --model llama3.2:1b --http-addr 0.0.0.0

# With NVIDIA GPU
docker build -f Dockerfile.cuda -t mullama-gpu .
docker run -d \
    --name mullama-gpu \
    --gpus all \
    -p 8080:8080 \
    -v mullama-models:/opt/mullama/models \
    mullama-gpu \
    serve --model llama3.2:1b --gpu-layers 35 --http-addr 0.0.0.0
```

---

## Docker Compose

### Basic Setup

```yaml
version: '3.8'

services:
  mullama:
    build: .
    container_name: mullama
    ports:
      - "8080:8080"
    volumes:
      - mullama-models:/opt/mullama/models
    environment:
      - MULLAMA_CACHE_DIR=/opt/mullama/models
      - MULLAMA_LOG_LEVEL=info
    command: serve --model llama3.2:1b --http-addr 0.0.0.0 --http-port 8080
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      start_period: 60s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'

volumes:
  mullama-models:
```

### With NVIDIA GPU

```yaml
version: '3.8'

services:
  mullama:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    container_name: mullama-gpu
    ports:
      - "8080:8080"
    volumes:
      - mullama-models:/opt/mullama/models
    environment:
      - MULLAMA_CACHE_DIR=/opt/mullama/models
      - MULLAMA_LOG_LEVEL=info
      - NVIDIA_VISIBLE_DEVICES=all
    command: >
      serve
        --model llama3.2:1b
        --model qwen2.5:7b
        --gpu-layers 35
        --context-size 8192
        --http-addr 0.0.0.0
        --http-port 8080
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 32G
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      start_period: 120s
      retries: 3

volumes:
  mullama-models:
    driver: local
```

### Full Stack (with Monitoring)

```yaml
version: '3.8'

services:
  mullama:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    container_name: mullama
    ports:
      - "8080:8080"
    volumes:
      - mullama-models:/opt/mullama/models
    environment:
      - MULLAMA_CACHE_DIR=/opt/mullama/models
      - MULLAMA_LOG_LEVEL=info
    command: >
      serve
        --model llama3.2:1b
        --gpu-layers 35
        --http-addr 0.0.0.0
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      start_period: 120s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  mullama-models:
  prometheus-data:
  grafana-data:
```

---

## Reverse Proxy

### Nginx (HTTPS Termination)

```nginx
upstream mullama_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name llm.example.com;

    ssl_certificate /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Request size limit (for base64 images in vision requests)
    client_max_body_size 50M;

    # Timeouts for long-running generation
    proxy_connect_timeout 10s;
    proxy_send_timeout 120s;
    proxy_read_timeout 300s;

    # API endpoints
    location / {
        proxy_pass http://mullama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    # Health check (no auth required)
    location /health {
        proxy_pass http://mullama_backend/health;
        access_log off;
    }

    # Metrics (restrict access)
    location /metrics {
        proxy_pass http://mullama_backend/metrics;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }

    # Optional: Basic auth for API
    # location /v1/ {
    #     auth_basic "Mullama API";
    #     auth_basic_user_file /etc/nginx/htpasswd;
    #     proxy_pass http://mullama_backend;
    #     proxy_buffering off;
    #     proxy_set_header Connection '';
    #     proxy_http_version 1.1;
    # }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name llm.example.com;
    return 301 https://$server_name$request_uri;
}
```

### Nginx (Load Balancing Multiple Instances)

```nginx
upstream mullama_pool {
    least_conn;
    server 127.0.0.1:8080 weight=1;
    server 127.0.0.1:8081 weight=1;
    server 127.0.0.1:8082 weight=1;
    keepalive 64;
}

server {
    listen 443 ssl http2;
    server_name llm.example.com;

    ssl_certificate /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;

    client_max_body_size 50M;
    proxy_read_timeout 300s;

    location / {
        proxy_pass http://mullama_pool;
        proxy_buffering off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
    }
}
```

---

## Load Balancing

### Multiple Instances

Run multiple Mullama instances with different models or configurations:

```bash
# Instance 1: Fast models (low latency)
mullama serve --model llama3.2:1b --http-port 8080 \
    --socket ipc:///var/run/mullama-1.sock --gpu-layers 35

# Instance 2: Large models (high quality)
mullama serve --model qwen2.5:7b --http-port 8081 \
    --socket ipc:///var/run/mullama-2.sock --gpu-layers 35

# Instance 3: Embedding models
mullama serve --model nomic-embed --http-port 8082 \
    --socket ipc:///var/run/mullama-3.sock
```

### HAProxy Configuration

```haproxy
global
    maxconn 4096
    log /dev/log local0

defaults
    mode http
    timeout connect 10s
    timeout client 300s
    timeout server 300s
    option httplog

frontend mullama_front
    bind *:443 ssl crt /etc/haproxy/certs/llm.pem
    default_backend mullama_back

    # Route embeddings to dedicated instance
    acl is_embeddings path_beg /v1/embeddings
    use_backend mullama_embed if is_embeddings

backend mullama_back
    balance leastconn
    option httpchk GET /health
    http-check expect status 200

    server mullama1 127.0.0.1:8080 check inter 10s fall 3 rise 2
    server mullama2 127.0.0.1:8081 check inter 10s fall 3 rise 2

backend mullama_embed
    server embed1 127.0.0.1:8082 check inter 10s fall 3 rise 2
```

---

## Monitoring

### Prometheus Scraping Configuration

Add to `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mullama'
    metrics_path: '/metrics'
    static_configs:
      - targets:
          - 'localhost:8080'
          - 'localhost:8081'
        labels:
          service: 'mullama'
    scrape_interval: 10s
```

### Key Metrics to Monitor

| Metric | Type | Alert Threshold | Description |
|--------|------|-----------------|-------------|
| `mullama_requests_active` | gauge | > 10 | Currently processing requests |
| `mullama_models_loaded` | gauge | = 0 | No models loaded |
| `mullama_memory_used_bytes` | gauge | > 90% system | Memory pressure |
| `mullama_tokens_per_second` | gauge | < 5 | Slow generation |
| `mullama_request_duration_seconds` | histogram | p99 > 30s | Slow requests |
| `mullama_uptime_seconds` | counter | resets | Unexpected restarts |

### Alerting Rules

```yaml
groups:
  - name: mullama_alerts
    rules:
      - alert: MullamaDown
        expr: up{job="mullama"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Mullama daemon is down"

      - alert: MullamaNoModels
        expr: mullama_models_loaded == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "No models loaded in Mullama"

      - alert: MullamaHighLatency
        expr: histogram_quantile(0.99, mullama_request_duration_seconds_bucket) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Mullama p99 latency exceeds 30s"

      - alert: MullamaHighMemory
        expr: mullama_memory_used_bytes / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Mullama using >90% system memory"
```

### Grafana Dashboard

Import the following dashboard JSON or create panels for:

- Request rate (requests/second by endpoint)
- Token generation speed (tokens/second by model)
- Active requests gauge
- Memory usage by model
- Request duration histogram
- Model load status
- Error rate

---

## Health Checks

### HTTP Health Check

```bash
# Simple health check
curl -sf http://localhost:8080/health || echo "UNHEALTHY"

# Detailed status check
curl -sf http://localhost:8080/api/system/status | jq '{
  status: "ok",
  models: .models_loaded,
  uptime: .uptime_secs,
  active: .stats.active_requests
}'
```

### Kubernetes Probes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: mullama
      image: mullama:latest
      ports:
        - containerPort: 8080
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 120
        periodSeconds: 30
        timeoutSeconds: 5
        failureThreshold: 3
      readinessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 60
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3
      startupProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
        failureThreshold: 30
```

### Script-Based Health Check

```bash
#!/bin/bash
# /opt/mullama/healthcheck.sh

set -e

ENDPOINT="${MULLAMA_HEALTH_URL:-http://localhost:8080/health}"
TIMEOUT=5

response=$(curl -sf --max-time $TIMEOUT "$ENDPOINT" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "CRITICAL: Mullama daemon not responding"
    exit 2
fi

status=$(echo "$response" | jq -r '.status')
if [ "$status" != "ok" ]; then
    echo "WARNING: Mullama status is $status"
    exit 1
fi

echo "OK: Mullama daemon healthy"
exit 0
```

---

## Log Rotation

### Logrotate Configuration

Create `/etc/logrotate.d/mullama`:

```
/var/log/mullama/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 mullama mullama
    sharedscripts
    postrotate
        systemctl reload mullama 2>/dev/null || true
    endscript
}
```

### Journald Configuration

For systemd-managed logging, configure `/etc/systemd/journald.conf`:

```ini
[Journal]
SystemMaxUse=2G
MaxRetentionSec=30day
```

---

## Security

### Running as Non-Root

Always run Mullama as a dedicated non-root user:

```bash
# Create dedicated user
sudo useradd --system --shell /bin/false --home-dir /opt/mullama mullama

# Set file ownership
sudo chown -R mullama:mullama /opt/mullama
```

### Firewall Rules (UFW)

```bash
# Allow only specific IPs to access the API
sudo ufw allow from 10.0.0.0/8 to any port 8080 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp

# Allow Prometheus scraping from monitoring network
sudo ufw allow from 10.0.10.0/24 to any port 8080 proto tcp

# Deny all other access to port 8080
sudo ufw deny 8080/tcp
```

### Firewall Rules (iptables)

```bash
# Allow from specific subnet
iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT

# Drop all other traffic to port 8080
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### API Key Authentication

Configure an API key requirement in the daemon:

```bash
# Set API key via environment variable
export MULLAMA_API_KEY="your-secret-api-key-here"
mullama serve --model llama3.2:1b
```

Clients must include the key in requests:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### TLS (Without Reverse Proxy)

For environments without a reverse proxy, the daemon can serve HTTPS directly (when built with TLS support):

```bash
mullama serve \
    --model llama3.2:1b \
    --tls-cert /etc/mullama/cert.pem \
    --tls-key /etc/mullama/key.pem \
    --http-port 8443
```

---

## Resource Limits

### Memory Management

```bash
# Set maximum memory for the systemd service
# In mullama.service [Service] section:
MemoryMax=32G
MemoryHigh=28G  # Trigger memory pressure handling at 28G

# Or via cgroups directly
echo "32G" > /sys/fs/cgroup/mullama/memory.max
```

### CPU Pinning

For consistent performance, pin the process to specific CPU cores:

```bash
# Using taskset
taskset -c 0-7 mullama serve --model llama3.2:1b --threads 8

# In systemd service
# [Service]
CPUAffinity=0 1 2 3 4 5 6 7
```

### GPU Memory

Monitor and limit GPU memory usage:

```bash
# Check GPU memory before loading
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# Limit GPU layers based on available VRAM
# ~500 MB per layer for 7B models at Q4_K_M
mullama serve --model qwen2.5:7b --gpu-layers 20  # ~10 GB VRAM
```

---

## Graceful Shutdown

The daemon handles graceful shutdown with the following behavior:

1. Stops accepting new requests
2. Waits for active requests to complete (up to timeout)
3. Unloads all models (frees GPU/CPU memory)
4. Closes IPC and HTTP sockets
5. Exits with code 0

```bash
# Graceful stop (waits for active requests)
mullama daemon stop

# Force stop after 10 seconds
mullama daemon stop --timeout 10

# Immediate kill (not recommended)
mullama daemon stop --force
```

For systemd, configure appropriate timeouts:

```ini
[Service]
TimeoutStopSec=60
KillMode=mixed
KillSignal=SIGTERM
SendSIGKILL=yes
```

---

## Backup and Recovery

### Model Cache Backup

```bash
# Backup model cache
tar -czf mullama-models-backup.tar.gz -C /opt/mullama models/

# Restore
tar -xzf mullama-models-backup.tar.gz -C /opt/mullama/
chown -R mullama:mullama /opt/mullama/models
```

### Session Backup

```bash
# Backup sessions and custom models
tar -czf mullama-config-backup.tar.gz \
    /opt/mullama/models/*.json \
    /home/mullama/.mullama/sessions/

# Restore
tar -xzf mullama-config-backup.tar.gz -C /
```

### Disaster Recovery Procedure

1. Install Mullama binary on new host
2. Restore model cache from backup (or re-pull from HuggingFace)
3. Restore custom model configurations
4. Start the daemon with the same configuration
5. Verify health: `curl http://localhost:8080/health`
6. Verify models loaded: `curl http://localhost:8080/api/models`

---

## Production Checklist

!!! success "Pre-Deployment Checklist"

    - [ ] Build with release optimizations: `cargo build --release --features daemon`
    - [ ] Pre-download all required models
    - [ ] Configure systemd service with resource limits
    - [ ] Set up reverse proxy with TLS termination
    - [ ] Configure firewall rules
    - [ ] Set up Prometheus metrics scraping
    - [ ] Configure alerting rules
    - [ ] Set up log rotation
    - [ ] Run as non-root user with minimal permissions
    - [ ] Test health check endpoint
    - [ ] Verify graceful shutdown behavior
    - [ ] Set appropriate GPU layers for available VRAM
    - [ ] Configure backup schedule for model cache
    - [ ] Document API key and distribute to authorized users
    - [ ] Load test with expected request patterns
    - [ ] Set up monitoring dashboard
