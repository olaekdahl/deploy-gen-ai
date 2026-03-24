# Demo 3 -- Kubernetes Deployment and Autoscaling

## What This Demo Teaches

- Deploying a containerized GenAI service to Kubernetes
- Kubernetes Deployment, Service, and Namespace resources
- Liveness, readiness, and startup probes for model-loading services
- Resource requests and limits for capacity planning
- Horizontal Pod Autoscaler (HPA) for CPU-based autoscaling
- How the HPA formula calculates desired replicas
- Generating test traffic to observe scaling in action

---

## Prerequisites

- Docker 24+ installed
- A local Kubernetes cluster: **minikube** or **kind**
- `kubectl` installed and configured
- `metrics-server` installed in the cluster (required for HPA)
- Python 3.11+ (for the load test script)

### Setting Up Minikube

```bash
# Start minikube with enough resources for the demo
minikube start --cpus=4 --memory=8192

# Enable the metrics-server addon (required for HPA)
minikube addons enable metrics-server

# Point Docker to minikube's Docker daemon
eval $(minikube docker-env)
```

---

## Building the Image

```bash
# Build inside minikube's Docker daemon
docker build -t genai-service:v3 .
```

---

## Deploying to Kubernetes

Apply the manifests in order:

```bash
# Create the namespace
kubectl apply -f k8s/namespace.yaml

# Deploy the service
kubectl apply -f k8s/deployment.yaml

# Expose the service via NodePort
kubectl apply -f k8s/service.yaml

# Enable autoscaling
kubectl apply -f k8s/hpa.yaml
```

### Verify the Deployment

```bash
# Check pods are running
kubectl get pods -n genai-demo

# Watch pods come up (model loading takes 1-2 minutes)
kubectl get pods -n genai-demo --watch

# Check the service
kubectl get svc -n genai-demo

# Check the HPA status
kubectl get hpa -n genai-demo
```

---

## Accessing the Service

### With Minikube

```bash
# Get the service URL
minikube service genai-service -n genai-demo --url
```

### With NodePort

```bash
# The service is exposed on port 30080 on any cluster node
curl http://$(minikube ip):30080/health
```

### Port Forwarding (alternative)

```bash
kubectl port-forward svc/genai-service -n genai-demo 8000:80
curl http://localhost:8000/health
```

---

## Testing the Service

```bash
# Health check
curl http://$(minikube ip):30080/health

# Generate text
curl -X POST http://$(minikube ip):30080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Kubernetes orchestration enables", "max_tokens": 30}'
```

---

## Observing Autoscaling

### Step 1: Watch the HPA

Open a terminal and run:

```bash
kubectl get hpa -n genai-demo --watch
```

### Step 2: Watch the pods

Open another terminal and run:

```bash
kubectl get pods -n genai-demo --watch
```

### Step 3: Generate load

```bash
pip install requests
python load_test.py --url http://$(minikube ip):30080 --concurrency 10 --duration 120
```

### What to Observe

1. The HPA shows increasing CPU utilization percentages.
2. When utilization exceeds 50%, the HPA increases the replica count.
3. New pods appear in the pod watch terminal.
4. After the load test ends, the HPA gradually scales down (after the 300s stabilization window).

### How Scaling Decisions Are Made

The HPA uses this formula:

```
desiredReplicas = ceil(currentReplicas * (currentCPU / targetCPU))
```

Example: 2 pods running at 80% CPU with a 50% target:

```
desired = ceil(2 * (80 / 50)) = ceil(3.2) = 4 replicas
```

The HPA respects:
- **minReplicas (2)**: never scales below this, even when idle
- **maxReplicas (8)**: never scales above this, even under extreme load
- **scaleUp stabilization (30s)**: waits before scaling up to avoid premature reactions
- **scaleDown stabilization (300s)**: waits 5 minutes before scaling down to avoid flapping

---

## Cleanup

```bash
kubectl delete namespace genai-demo
```

---

## File Structure

```
demo-3-kubernetes-deployment-and-autoscaling/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with /ready probe endpoint
│   ├── models.py
│   ├── inference.py
│   └── logging_config.py
├── k8s/
│   ├── namespace.yaml       # genai-demo namespace
│   ├── deployment.yaml      # 2 replicas with probes and resource limits
│   ├── service.yaml         # NodePort service on port 30080
│   └── hpa.yaml             # HPA targeting 50% CPU utilization
├── Dockerfile
├── .dockerignore
├── load_test.py             # Concurrent load generator
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Teaching Talking Points

1. **Readiness vs. Liveness Probes**: GenAI models take time to load.
   The startup probe gives the model time to initialize. The readiness
   probe prevents traffic from reaching pods that are still loading.
   The liveness probe restarts pods that become unresponsive.

2. **Resource Requests vs. Limits**: Requests are what the scheduler
   guarantees; limits are the ceiling. Under-requesting means pods
   compete for resources. Over-limiting means wasted capacity.

3. **HPA Scaling Formula**: Walk through the `ceil(current * actual/target)`
   formula with concrete numbers. Show that scaling is proportional, not
   binary.

4. **Stabilization Windows**: Explain why asymmetric stabilization
   (fast scale-up, slow scale-down) prevents oscillation and wasted
   resources from constant pod churn.

5. **CPU as a Proxy for GenAI Load**: On CPU-only deployments, CPU
   utilization correlates closely with inference demand. For GPU deployments,
   you would need custom metrics (GPU utilization, queue depth, or
   request latency).
