# Autonomous Green Lung Urban AI: Multi-Agent Reinforcement Learning for Dynamic Atmospheric Mitigation

A decentralized cyber-physical urban mitigation architecture powered by Multi-Agent Reinforcement Learning for real-time PM2.5 control in industrial corridors.

## Overview

The Autonomous Green Lung (AGL) framework represents a paradigm shift in urban air quality management. Rather than relying on static regulatory measures and passive green infrastructure, AGL introduces an intelligent, self-regulating urban ecosystem that dynamically adapts to atmospheric conditions and emission patterns.

### Key Innovations

- **Decentralized MARL Control**: Distributed agents coordinate Bio-Digital Curtains and smart traffic controllers without centralized oversight
- **Cyber-Physical Integration**: Seamless fusion of computational intelligence with physical mitigation assets
- **Multi-Objective Optimization**: Balances public health (PM2.5 reduction) against economic throughput (traffic efficiency)
- **Real-Time Adaptation**: Responds dynamically to Shamal winds, traffic patterns, and meteorological changes
- **Scalable Architecture**: Proven effective in the Sitra–Manama Industrial Corridor, applicable to other urban centers

## System Architecture

### Core Components

```
Autonomous Green Lung Framework
├── Multi-Agent Reinforcement Learning Engine
│   ├── Filtration Agents (Bio-Digital Curtains)
│   ├── Traffic Agents (Smart Controllers)
│   └── Coordination Module
├── Cyber-Physical Assets
│   ├── Bio-Digital Curtains (Photobioreactors + Mechanical Filters)
│   ├── Smart Traffic Controllers
│   └── Distributed Sensor Network
├── Environmental Modeling
│   ├── Computational Fluid Dynamics (CFD)
│   ├── Pollutant Dispersion Simulation
│   └── Meteorological Integration
└── Optimization Engine
    ├── Multi-Objective Reward Function
    ├── Decentralized Execution
    └── Real-Time Decision Making
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.12+
- OpenFOAM (for CFD simulations)
- NumPy 1.21+
- Pandas 1.3+
- Stable-Baselines3 1.6+
- Ray 2.0+

### Setup

```bash
git clone https://github.com/yourusername/AGL-Research.git
cd AGL-Research
pip install -r requirements.txt
```

## Project Structure

```
AGL-Research/
├── agl/
│   ├── __init__.py
│   ├── core/
│   │   ├── marl_engine.py
│   │   ├── environment.py
│   │   └── reward_function.py
│   ├── agents/
│   │   ├── filtration_agent.py
│   │   ├── traffic_agent.py
│   │   └── agent_coordinator.py
│   ├── assets/
│   │   ├── bio_digital_curtain.py
│   │   ├── smart_traffic_controller.py
│   │   └── sensor_network.py
│   ├── simulation/
│   │   ├── cfd_environment.py
│   │   ├── pollutant_dispersion.py
│   │   └── meteorological_model.py
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── data_processing.py
├── experiments/
│   ├── train_marl.py
│   ├── evaluate_performance.py
│   ├── compare_baselines.py
│   └── config/
│       └── default_config.yaml
├── simulation_data/
│   ├── cfd_domain.py
│   ├── meteorological_data.py
│   └── emission_sources.py
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_agent_training.ipynb
│   └── 03_performance_analysis.ipynb
├── tests/
│   ├── test_agents.py
│   ├── test_environment.py
│   ├── test_reward_function.py
│   └── test_cfd_integration.py
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### Training MARL Agents for Air Quality Control

```python
from agl.core import MARLEngine, UrbanEnvironment
from agl.agents import FiltrationAgent, TrafficAgent
from agl.simulation import CFDEnvironment

cfd_env = CFDEnvironment(
    domain_size=(10000, 5000, 500),
    resolution=10,
    shamal_wind_enabled=True
)

env = UrbanEnvironment(
    cfd_environment=cfd_env,
    n_filtration_agents=20,
    n_traffic_agents=15
)

marl_engine = MARLEngine(
    environment=env,
    n_filtration_agents=20,
    n_traffic_agents=15,
    learning_rate=3e-4,
    privacy_epsilon=1.0
)

training_results = marl_engine.train(
    total_timesteps=1000000,
    eval_freq=10000,
    n_eval_episodes=10
)
```

### Evaluating Real-Time Performance

```python
from agl.core import MARLEngine
from agl.utils import PerformanceMetrics

engine = MARLEngine.load('trained_model.pkl')
metrics = PerformanceMetrics()

results = engine.evaluate(
    n_episodes=100,
    track_metrics=True
)

print(f"Average PM2.5 Reduction: {results['pm25_reduction']:.1%}")
print(f"Peak Event Reduction: {results['peak_event_reduction']:.1%}")
print(f"Traffic Latency Impact: {results['traffic_latency_change']:.1%}")
print(f"Energy Consumption: {results['energy_consumption']:.0f} kWh/day")
```

## Experimental Results

The AGL framework achieves substantial improvements in urban air quality management:

| Metric | No Intervention | Static Mitigation | AGL (Proposed) | Improvement |
|--------|-----------------|-------------------|----------------|-------------|
| Avg. Residential PM2.5 (µg/m³) | 28.4 | 23.1 | 18.6 | -34.5% |
| Peak PM2.5 Events (>50 µg/m³) | 14 days/yr | 9 days/yr | 3 days/yr | -78.6% |
| Avg. Traffic Latency (min) | 12.5 | 12.8 | 13.2 | +5.6% |
| BDC Energy Consumption (kWh/day) | 0 | 150 | 210 | - |

## System Components

### Bio-Digital Curtains (BDCs)

Hybrid vegetative-mechanical filtration units deployed at strategic locations:

- **Photobioreactors**: Micro-algae (*Chlorella vulgaris*, *Spirulina*) for passive CO2 sequestration and PM2.5 capture
- **Mechanical Filtration**: HEPA filters with variable fan speeds for active air purification
- **Smart Control**: Embedded sensors and actuators for real-time optimization
- **Capacity**: Each unit can filter 500-1000 m³/hour of air

### Smart Traffic Controllers (STCs)

Intelligent intersection management systems:

- **Local Sensing**: Vehicle detection, queue monitoring, average speed measurement
- **Dynamic Timing**: Real-time signal adjustment based on traffic and air quality
- **Route Optimization**: Suggestions for traffic diversion to reduce localized emissions
- **Communication**: Inter-controller coordination for corridor-wide optimization

### Computational Fluid Dynamics (CFD) Environment

High-fidelity simulation of atmospheric dynamics:

- **Domain**: 10 km × 5 km × 500 m covering Sitra–Manama corridor
- **Resolution**: 10-meter grid cells (refined to 1-meter near sources)
- **Physics**: RANS equations with k-epsilon turbulence modeling
- **Meteorology**: Shamal wind patterns, temperature inversions, diurnal cycles
- **Emissions**: Industrial stacks, vehicular sources, fugitive emissions

## MARL Algorithm

The framework employs **Multi-Agent Proximal Policy Optimization (MAPPO)** with:

- **Centralized Training**: Global critic with access to full state for efficient learning
- **Decentralized Execution**: Local actor networks for scalable deployment
- **Multi-Objective Reward**: Balances PM2.5 exposure, traffic latency, and energy consumption
- **Privacy Protection**: Differential Privacy on agent updates

### Reward Function

$$R(S_t, A_t) = - (\alpha \cdot PWE(S_t) + \beta \cdot \Delta L(S_t, A_t) + \gamma \cdot E(A_t))$$

Where:
- **PWE**: Population-Weighted PM2.5 Exposure
- **ΔL**: Traffic Latency Deviation
- **E**: Energy Consumption
- **α, β, γ**: Policy-tunable weighting factors

## Performance Metrics

The AGL framework is evaluated on multiple dimensions:

- **Air Quality**: Average PM2.5, peak events, exposure distribution
- **Economic Impact**: Traffic latency, throughput, congestion patterns
- **Energy Efficiency**: BDC power consumption, operational costs
- **Robustness**: Performance under Shamal winds, unexpected emissions
- **Scalability**: Performance with varying numbers of agents

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{taha2026agl,
  title={Autonomous Green Lung Urban AI: Multi-Agent Reinforcement Learning for Dynamic Atmospheric Mitigation in the Sitra--Manama Industrial Corridor},
  author={Taha, Mohammed},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ministry of Health, Kingdom of Bahrain
- Nasser Centre for Science and Technology (NCST)
- Environmental Protection Agency, Kingdom of Bahrain

## Contact

For questions or inquiries, please contact:
- Mohammed Taha: nv23014@ncst.edu.bh
- Department of Computer Science, NCST

## References

For detailed technical information, please refer to the accompanying research paper: "Autonomous Green Lung Urban AI: Multi-Agent Reinforcement Learning for Dynamic Atmospheric Mitigation in the Sitra–Manama Industrial Corridor"
