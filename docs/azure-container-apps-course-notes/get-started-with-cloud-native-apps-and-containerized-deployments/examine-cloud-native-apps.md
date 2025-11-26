---
title: Examine Cloud-Native Apps
parent: Get Started with Cloud-Native Apps and Containerized Deployments
grand_parent: Deploy Cloud-Native Apps Using Azure Container Apps
nav_order: 2
last_modified_date: 2025-11-26 20:43:00
---

# Examine Cloud-Native Apps
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

![Side-by-side comparison illustrating traditional versus cloud-native infrastructure approaches: Left side labeled TRADITIONAL (PETS) shows a person standing next to a server rack thinking about prod-server-01 with text below stating Monolithic & Manual and a red X mark; Right side labeled CLOUD-NATIVE (CATTLE) displays a cloud icon connected to multiple service instances (Service-01 through Service-04) with a CI/CD pipeline, accompanied by icons representing Modern Design, Containers, Backing Services, Automation, and DevOps Culture, with text below stating Microservices, Containers, Automation, Scalable and a green checkmark](/assets/images/azure-container-apps/what-is-cloud-native.png)

## What Is Cloud-Native?

![Cloud infrastructure diagram showing a cloud containing icons representing a Cloud Native App built from multiple microservices, with arrows connecting to surrounding concepts: Modern Design, Microservices, Containers on the right side, and Backing Services and Automation on the left side](/assets/images/azure-container-apps/cloud-native-app-diagram.png)

The speed and agility of cloud-native derives from many factors. Foremost is cloud infrastructure, but there are five other foundational pillars that provide the bedrock for cloud-native systems:

1. **Modern design** - Microservices architecture
2. **Containers** - Lightweight, portable packaging
3. **Backing services** - Managed cloud services
4. **Automation** - CI/CD and infrastructure as code
5. **DevOps culture** - Collaboration and shared responsibility

## Cloud Infrastructure Service Model

Cloud-native systems take full advantage of the cloud service model. They're designed to thrive in a dynamic, virtualized cloud environment, making extensive use of Platform as a Service (PaaS) compute infrastructure and managed services.

### Key Principle: Disposable Infrastructure

Cloud-native treats the underlying infrastructure as **disposable** - provisioned in minutes and resized, scaled, or destroyed on demand via automation.

### Pets vs. Cattle: A Comparison

The difference between traditional and cloud-native infrastructure can be understood through the "pets vs. cattle" analogy:

| Aspect | Pets (Traditional) | Cattle (Cloud-Native) |
|--------|-------------------|----------------------|
| **Identity** | Named servers (e.g., "prod-server-01") | System identifiers (e.g., "Service-01") |
| **Care** | Manually maintained and repaired | Automatically replaced when unhealthy |
| **Scaling** | Vertical (add resources to same machine) | Horizontal (add more instances) |
| **Failure Impact** | Everyone notices, requires intervention | Automatic replacement, minimal impact |
| **Updates** | Modified and patched in place | Destroyed and replaced with new version |
| **Infrastructure** | Mutable (changed over time) | Immutable (never modified after creation) |

### The Cattle Model in Practice

In the commodities (cattle) model:

- Each instance is provisioned as a virtual machine or container
- All instances are identical and interchangeable
- Instances receive system identifiers rather than meaningful names
- Failed or outdated instances are destroyed and replaced automatically
- The application continues running regardless of individual instance lifecycle

**Azure Support**: The Azure cloud platform supports this highly elastic infrastructure with automatic scaling, self-healing, and monitoring capabilities.

## Benefits of Cloud-Native Applications

Cloud-native applications are built to take advantage of cloud computing models to increase speed, flexibility, and quality while reducing deployment risks.

### Core Benefits

| Benefit | Description |
|---------|-------------|
| **Resilient** | Designed to be loosely coupled and distributed - if one component fails, the application continues to function |
| **Elastic** | Scale out to meet demand, scale in to reduce costs, or scale to zero when not in use |
| **Observable** | Built-in monitoring capabilities for health and performance tracking |
| **Automated** | Build, test, and deploy quickly and reliably through automation |

### Deployment and Operations

| Benefit | Description |
|---------|-------------|
| **Portable** | Run in the cloud, on-premises, or in hybrid environments |
| **Secure** | Built-in security practices protect data and customers |
| **Composable** | Built from modular components that can be reused across applications |
| **Managed** | Focus on building applications instead of managing infrastructure |

### Development and Team Benefits

| Benefit | Description |
|---------|-------------|
| **Modern** | Leverage the latest technologies and best practices |
| **Open** | Use open-source software and avoid vendor lock-in |
| **Collaborative** | Enable team-based development and shared ownership |
| **Agile** | Respond quickly to business changes and customer needs |
| **Innovative** | Adopt cutting-edge technologies and development practices |

### Business Value

| Benefit | Description |
|---------|-------------|
| **Cost-effective** | Pay only for resources you use |
| **Sustainable** | Reduce environmental impact through efficient resource usage |
| **Data-driven** | Use data to make informed decisions and improve applications |
| **Inclusive** | Build accessible applications for everyone |

## Key Takeaways

- Cloud-native applications embrace **disposable infrastructure** rather than long-lived servers
- The **cattle model** (horizontal scaling of identical instances) replaces the traditional **pets model** (vertical scaling of named servers)
- Cloud-native provides **17 key benefits** across resilience, operations, development, and business value
- Azure provides the platform capabilities needed for cloud-native success: automatic scaling, self-healing, and comprehensive monitoring
