# Getting Started with Autonomous Green Lung (AGL)

Welcome, fellow innovator! You're here because you believe in a future where our cities don't just exist, but thrive. This guide is your first step into the Autonomous Green Lung project—a mission to reclaim our breath and build truly intelligent urban ecosystems.

## Before You Dive In: The Essentials

To get started, you'll need a few tools in your belt:

*   **Python 3.8+**: Our primary language. Make sure it's installed and ready.
*   **OpenFOAM**: The powerhouse behind our CFD simulations. If you're serious about atmospheric modeling, this is non-negotiable.
*   **Git**: For version control. You know the drill.
*   **A Brain for Reinforcement Learning**: Basic understanding of RL and Python will make this journey much smoother.

## Setting Up Your Workspace

Let's get your environment ready. It's straightforward, I promise.

### 1. Clone the Mission Briefing (Repository)

First, grab the code. This is where the magic lives.

```bash
git clone https://github.com/yourusername/autonomous-green-lung-ai.git
cd autonomous-green-lung-ai
```

### 2. Create Your Sanctuary (Virtual Environment)

Keep your project dependencies clean and isolated. It's good practice.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, it's `venv\Scripts\activate`
```

### 3. Arm Your Arsenal (Install Dependencies)

Install all the necessary Python libraries. We've kept `requirements.txt` lean and mean.

```bash
pip install -r requirements.txt
```

## Project Blueprint: Where Everything Lives

Here's a quick map of the repository. We've structured it for clarity and scalability.

```
autonomous-green-lung-ai/
├── README.md                    # Your mission statement. Read it.
├── LICENSE                      # Open-source. Freedom to innovate.
├── CONTRIBUTING.md              # Want to help? Start here.
├── CODE_OF_CONDUCT.md           # Our community values. Be excellent.
├── requirements.txt             # All the Python packages you'll need.
├── docs/                        # The knowledge base.
│   ├── RESEARCH_PAPER.md       # The full story. The science. The breakthrough.
│   ├── GETTING_STARTED.md      # You're reading it.
│   ├── ARCHITECTURE.md         # Deep dive into the system design (coming soon).
│   └── API_REFERENCE.md        # How to talk to our systems (coming soon).
├── src/                         # The core intelligence.
│   ├── agents/                 # Our intelligent agents: BDCs, STCs, and the Coordinator.
│   ├── environment/            # The simulated world: CFD models, urban geometry, meteorology.
│   ├── training/               # Where the agents learn to breathe life into cities.
│   ├── utils/                  # Handy tools and configurations.
├── tests/                       # Ensuring everything works as it should.
├── data/                        # The raw fuel and processed insights.
└── notebooks/                   # Your sandbox for exploration and analysis.
```

## Your First Breath: Quick Start Examples

Let's get a taste of what AGL can do.

### Running a Simple Simulation

This snippet shows how to get a basic CFD simulation running with an agent.

```python
from src.environment.cfd_simulator import CFDSimulator
from src.agents.filtration_agent import FiltrationAgent

# Spin up the CFD simulator
simulator = CFDSimulator(domain_size=(10000, 5000, 500))

# Create a single filtration agent
agent = FiltrationAgent(agent_id=1)

# Let's run it for a bit
for step in range(100):
    observation = simulator.get_observation(agent.position)
    action = agent.select_action(observation)
    next_obs, reward = simulator.step(action)
    agent.update(observation, action, reward, next_obs)

print("Simple simulation complete. Agent took 100 steps.")
```

### Training Our Intelligent Agents

This is where the MARL magic happens. Get ready to teach our agents how to make cities breathe.

```python
from src.training.mappo_trainer import MAPPOTrainer

# Initialize the trainer with our agent counts
trainer = MAPPOTrainer(
    num_filtration_agents=20,
    num_traffic_agents=15,
    learning_rate=1e-4
)

# Train for a good number of episodes
trainer.train(num_episodes=10000)

# Save your hard-earned knowledge
trainer.save_models("checkpoints/trained_agents")

print("MARL agents trained and saved!")
```

## Fine-Tuning the Lungs: Configuration

All the critical parameters are in `src/utils/config.py`. Tweak them, experiment, and push the boundaries.

## Proving It Works: Running Tests

We believe in robust code. Run our tests to ensure everything is solid.

```bash
pytest tests/
```

## Seeing Is Believing: Visualization and Analysis

Our `notebooks/` directory is packed with Jupyter notebooks to help you visualize data, analyze results, and understand the impact of AGL.

## Troubleshooting: When Things Get Hazy

Encountering issues? Don't worry, it happens. Here are a few common ones:

### Issue: OpenFOAM isn't playing nice

**Solution**: Ensure OpenFOAM is installed correctly and your `FOAM_INST_DIR` environment variable points to its home.

```bash
export FOAM_INST_DIR=/path/to/OpenFOAM
```

### Issue: My machine is running out of breath during training

**Solution**: Training MARL agents and CFD simulations can be resource-intensive. Try reducing the batch size or the domain resolution in `src/utils/config.py`.

## What's Next for You?

*   **Read the [RESEARCH_PAPER.md](RESEARCH_PAPER.md)**: Get the full scientific narrative. It's a deep dive.
*   **Explore the `src/` directory**: This is where you can start building, experimenting, and innovating.
*   **Join the conversation**: We're building a community. Your insights are invaluable.

Let's make our cities truly breathe. Together.

## Need a Hand?

*   Check our (soon-to-be-created) FAQ.
*   Open an issue on GitHub. We're here to help.
*   Drop a line to Mohammed Taha at nv23014@ncst.edu.bh.

Happy innovating!
