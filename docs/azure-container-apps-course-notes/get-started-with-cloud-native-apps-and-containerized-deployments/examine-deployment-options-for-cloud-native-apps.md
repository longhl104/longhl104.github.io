---
title: Examine Deployment Options for Cloud-Native Apps
parent: Get Started with Cloud-Native Apps and Containerized Deployments
grand_parent: Deploy Cloud-Native Apps Using Azure Container Apps
nav_order: 3
---

# Examine Deployment Options for Cloud-Native Apps
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Azure provides multiple hosting options for cloud-native applications. Each service is optimized for different scenarios and use cases. This guide helps you understand which Azure service best fits your containerized application needs.

## Azure Container Services Comparison

| Service | Best For | Key Strength | Management Level |
|---------|----------|--------------|------------------|
| **Container Apps** | Microservices & event-driven apps | Serverless containers with auto-scaling | Fully managed |
| **App Service** | Web applications & APIs | Integrated web hosting platform | Fully managed |
| **Container Instances (ACI)** | Simple container tasks | Quickest container deployment | Minimal management |
| **Kubernetes Service (AKS)** | Complex orchestration needs | Full Kubernetes control | Self-managed cluster |
| **Functions** | Event-triggered code | Function-as-a-Service (FaaS) | Fully managed |
| **Spring Apps** | Java Spring applications | Spring-optimized platform | Fully managed |
| **Red Hat OpenShift** | Enterprise OpenShift workloads | Full OpenShift experience | Managed OpenShift |

## Detailed Service Descriptions

### Azure Container Apps

**What it is**: Serverless platform for running containerized microservices and jobs.

**Key Features**:

- Built on Kubernetes + open-source technologies (Dapr, KEDA, Envoy)
- Automatic scaling based on HTTP traffic, events, or queues
- Scale to zero when idle (pay nothing for unused capacity)
- Service discovery and traffic splitting built-in
- Supports scheduled and event-driven jobs

**When to use**:

- ✅ Building microservices architectures
- ✅ Event-driven applications
- ✅ Want Kubernetes benefits without managing clusters
- ❌ Need direct Kubernetes API access (use AKS instead)

**Why teams choose it**: Best balance of power and simplicity for containerized microservices.

---

### Azure App Service

**What it is**: Fully managed platform for hosting web applications and APIs.

**Key Features**:

- Deploy from code or containers
- Integrated with Azure ecosystem (Functions, Container Apps, databases)
- Built-in authentication, scaling, and monitoring
- Support for .NET, Java, Node.js, Python, PHP

**When to use**:

- ✅ Traditional web applications
- ✅ REST APIs
- ✅ Need integrated web hosting features
- ❌ Complex microservices (use Container Apps instead)

**Why teams choose it**: Simplest path for deploying standard web applications.

---

### Azure Container Instances (ACI)

**What it is**: On-demand container execution without orchestration.

**Key Features**:

- Single containers or container groups
- Hyper-V isolation for security
- Per-second billing
- No cluster management required

**Key Limitation**: No built-in load balancing, auto-scaling, or certificate management.

**When to use**:

- ✅ Simple batch jobs
- ✅ Task automation
- ✅ Build agents
- ✅ Testing and development
- ❌ Production microservices (use Container Apps instead)

**Why teams choose it**: Simplest building block for running a single container quickly.

---

### Azure Kubernetes Service (AKS)

**What it is**: Fully managed Kubernetes with complete API access.

**Key Features**:

- Full Kubernetes control plane access
- Run any Kubernetes workload
- Complete control over cluster configuration
- Integration with Azure services and monitoring

**When to use**:

- ✅ Need Kubernetes API access
- ✅ Complex orchestration requirements
- ✅ Existing Kubernetes expertise
- ✅ Migrate from on-premises Kubernetes
- ❌ Want to avoid cluster management (use Container Apps instead)

**Why teams choose it**: Maximum flexibility and control for Kubernetes workloads.

---

### Azure Functions

**What it is**: Serverless compute for event-driven code execution.

**Key Features**:

- Function-as-a-Service (FaaS) programming model
- Automatic scaling and event binding
- Deploy as code or containers
- Pay per execution (consumption plan)
- Rich triggers and bindings (HTTP, timers, queues, databases)

**When to use**:

- ✅ Event-driven functions
- ✅ Scheduled tasks
- ✅ Webhooks and APIs
- ✅ Data processing pipelines
- ❌ Long-running processes (use Container Apps instead)

**Why teams choose it**: Fastest way to deploy event-driven code without infrastructure management.

---

### Azure Spring Apps

**What it is**: Managed service specifically for Java Spring Framework applications.

**Key Features**:

- Optimized for Spring Boot and Spring Cloud
- Built-in Spring configuration management
- Service discovery, circuit breakers, and config servers
- Blue-green deployments
- Comprehensive monitoring and diagnostics

**When to use**:

- ✅ Java Spring applications
- ✅ Spring Boot microservices
- ✅ Spring Cloud architectures
- ❌ Non-Spring applications (use other services)

**Why teams choose it**: Best experience for Spring developers with infrastructure managed for you.

---

### Azure Red Hat OpenShift

**What it is**: Managed OpenShift platform jointly operated by Red Hat and Microsoft.

**Key Features**:

- Full OpenShift experience on Azure
- Integrated product support from both Microsoft and Red Hat
- Choose your own registry, networking, storage, CI/CD
- Built-in source code management and container builds
- Automated scaling and health management

**When to use**:

- ✅ Existing OpenShift investment
- ✅ Enterprise OpenShift requirements
- ✅ Need Red Hat support and ecosystem
- ❌ Starting fresh (consider AKS or Container Apps)

**Why teams choose it**: Seamless migration path for organizations already using OpenShift.

---

## Decision Tree

**Start here**: What are you deploying?

1. **Web application or API** → **App Service**
2. **Event-driven functions** → **Azure Functions**
3. **Java Spring applications** → **Azure Spring Apps**
4. **Existing OpenShift workloads** → **Azure Red Hat OpenShift**
5. **Containerized applications**:
   - Need Kubernetes API access? → **AKS**
   - Simple single container task? → **Container Instances**
   - Microservices or event-driven? → **Container Apps** ⭐ (Recommended starting point)

## Key Takeaways

- **Azure Container Apps** is the recommended starting point for most containerized microservices
- **App Service** remains the best choice for traditional web applications
- **AKS** provides full Kubernetes control when you need it
- **Functions** excels at event-driven, short-lived operations
- **Container Instances** is the simplest option for running a single container
- Specialized services exist for **Spring** (Java) and **OpenShift** workloads
