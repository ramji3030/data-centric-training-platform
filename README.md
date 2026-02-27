# Data-Centric Training Platform

**Production-Ready Infrastructure for Data-Centric LLM Training**

A comprehensive platform that optimizes training data instead of focusing on model architectures. Inspired by Andrew Ng's data-centric AI movement, this system provides production-grade infrastructure for deduplication, quality scoring, active learning, and expert feedback integration.

## Why Data-Centric Training?

- **80/20 Rule**: 80% of model quality comes from data, 20% from the model
- **Scalable**: Improve models without retraining from scratch
- **Cost-Effective**: Focus resources on data, not massive models
- **Market Gap**: Few open-source, production-ready systems exist

## Core Features

### 1. Data Deduplication
- **Semantic Deduplication**: Find near-duplicates using embeddings
- **Exact Matching**: Hash-based duplicate detection
- **Configurable Thresholds**: Adjust sensitivity per domain
- **Batch Operations**: Process millions of documents efficiently

### 2. Quality Scoring Pipeline
- **Perplexity Scoring**: Measure text complexity and coherence
- **Toxicity Detection**: Flag harmful or biased content
- **Domain Relevance Scoring**: Custom scorers per domain
- **Language Quality**: Grammar, spelling, readability metrics
- **Source Credibility**: Track document provenance
- **Composable Scorers**: Chain multiple quality metrics

### 3. Active Learning System
- **Uncertainty Sampling**: Select ambiguous examples for labeling
- **Diversity Sampling**: Ensure representative data distribution
- **Query-by-Committee**: Ensemble-based active learning
- **Budget Management**: Optimize annotation budget allocation
- **Batch Selection**: Select optimal batches for training

### 4. Expert Feedback Integration
- **Web UI for Annotation**: Experts mark "good/bad" examples
- **Dynamic Sampling Weights**: System updates distribution based on feedback
- **Feedback Loop Closure**: Changes immediately reflected in training
- **Audit Trail**: Track all expert decisions and rationales
- **Multi-Expert Consensus**: Handle disagreements gracefully

### 5. Synthetic Data Generation with Safeguards
- **Controlled Generation**: LLM-based synthetic data with filters
- **Feedback Loop Prevention**: Detect and prevent reward hacking
- **Distribution Matching**: Ensure synthetic data matches real distribution
- **Diversity Enforcement**: Avoid mode collapse
- **Validation Pipeline**: Automatic quality checks for synthetic samples

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Data Ingestion & Storage Layer                  │
│   (Databases, Vector Stores, File Systems)              │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│      Data Processing Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Deduplication│→ │ Quality      │→ │ Active       │   │
│  │              │  │ Scoring      │  │ Learning     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│      Expert Feedback & Augmentation Layer               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Expert UI    │→ │ Feedback     │→ │ Synthetic    │   │
│  │              │  │ Integration  │  │ Generation   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│      Training Data Output & Monitoring                  │
│  (Datasets, Metrics, Dashboards)                        │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/ramji3030/data-centric-training-platform.git
cd data-centric-training-platform
pip install -r requirements.txt
```

### Basic Usage

```python
from dctp.pipeline import DataCentricPipeline
from dctp.scorers import PerplexityScorer, ToxicityScorer

# Initialize pipeline
pipeline = DataCentricPipeline(
    db_url="postgresql://localhost/dctp"
)

# Add quality scorers
pipeline.add_scorer(PerplexityScorer())
pipeline.add_scorer(ToxicityScorer())

# Process training data
results = pipeline.process(
    documents=my_documents,
    dedup_threshold=0.95,
    min_quality_score=0.7
)
```

## Project Roadmap

### Phase 1: Core Infrastructure
- [x] Data deduplication (exact & semantic)
- [x] Quality scoring pipeline
- [x] Database schema & APIs
- [x] Configuration management

### Phase 2: Active Learning (In Progress)
- Uncertainty sampling strategies
- Diversity-based sampling
- Query-by-committee ensemble
- Batch selection optimization

### Phase 3: Expert Feedback System
- Web UI for annotation
- Feedback persistence & analytics
- Dynamic weight adjustment
- Consensus mechanisms

### Phase 4: Synthetic Data & Safeguards
- LLM-based generation
- Feedback loop detection
- Distribution validation
- Reward hacking prevention

### Phase 5: Production & Monitoring
- End-to-end pipeline orchestration
- Metrics dashboards
- Cost tracking
- Multi-tenant support

## API Overview

### Deduplication
```python
from dctp.dedup import SemanticDeduplicator

dedup = SemanticDeduplicator(model="sentence-transformers/all-MiniLM-L6-v2")
clusters = dedup.find_duplicates(documents, threshold=0.95)
```

### Quality Scoring
```python
from dctp.quality import QualityScorer

scorer = QualityScorer()
scores = scorer.score_batch(documents)
# Returns: perplexity, toxicity, domain_relevance, etc.
```

### Active Learning
```python
from dctp.active_learning import UncertaintySampler

sampler = UncertaintySampler(model=your_model, budget=100)
selected_indices = sampler.select_batch(unlabeled_data, predictions)
```

## Configuration

See `.env.example` for all configuration options:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
black dctp/ tests/

# Type checking
mypy dctp/
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this platform in research, please cite:

```bibtex
@software{dctp2026,
  title={Data-Centric Training Platform},
  author={Your Name},
  year={2026},
  url={https://github.com/ramji3030/data-centric-training-platform}
}
```

## License

MIT - See [LICENSE](LICENSE) for details

## References

- Ng, A. (2021). "Data-centric AI vs. Model-centric AI"
- Ratner, A., et al. (2016). Data Programming: Creating Large Training Sets Quickly
- Monarch, R. M. (2021). Human-in-the-Loop Machine Learning
- Settles, B. (2009). Active Learning Literature Survey

## Contact

For questions and feedback, please open an issue or contact the maintainers.
