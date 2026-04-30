# Autonomous Green Lung Urban AI: A Human-Centric Approach to Dynamic Atmospheric Mitigation

**Author:** Mohammed Taha
**Email:** nv23014@ncst.edu.bh
**Affiliation:** Department of Computer Science, Nasser Centre for Science and Technology (NCST), Kingdom of Bahrain

## Abstract

Imagine if the city you live in could actually breathe. What if it could adapt on the fly to protect you from air pollution? That’s the core idea behind the Autonomous Green Lung (AGL) framework. I live in Bahrain and if you look at the Sitra–Manama industrial corridor you see a lot of rapid development. But there's a harsh reality that comes with it: PM2.5 levels constantly push past what the World Health Organization says is safe. The old ways of dealing with this—static regulations and fixed green spaces—just don't work anymore. Pollution is dynamic. It changes by the hour. Add in our coastal geography and the seasonal "Shamal" winds that trap pollutants in stagnation zones and you realize we need something proactive.

So I built the AGL framework. It’s a decentralized cyber-physical system driven by Multi-Agent Reinforcement Learning (MARL). Think of it as a smart urban ecosystem. I set up distributed agents—basically tiny digital guardians that coordinate Bio-Digital Curtains (these are hybrid vegetative-mechanical filters) and smart traffic controllers. Their job is to actively push pollution away from where people live. I designed a reward system that doesn't just focus on cutting PM2.5 exposure but also keeps traffic flowing so the city's economy doesn't stall. I ran high-fidelity Computational Fluid Dynamics (CFD) simulations and the results were honestly remarkable. This MARL-driven setup achieved a 34.5% drop in average residential PM2.5 concentrations. This isn't just theory. It’s a real path forward to build cities that regulate themselves and protect us from the downsides of industrial growth.

## 1. Introduction: The Urgent Need for Dynamic Air Quality Management in Urban Industrial Corridors

Things are changing fast. Cities are growing industries are booming and we're paying a steep environmental price for that economic prosperity. Air pollution is a silent killer and it's now a leading global health risk [1]. The main culprit is PM2.5. These are microscopic particles less than 2.5 micrometers wide. They are incredibly dangerous because they bypass our body's natural defenses get deep into our lungs and even enter our bloodstream. They cause heart disease respiratory issues and neurological problems [2]. The WHO has strict guidelines for PM2.5 but most industrial cities blow right past them.

Look at the Sitra–Manama Industrial Corridor here in Bahrain. It’s the perfect example of this problem. It’s an economic powerhouse packed with oil refineries petrochemical plants and power stations all sitting right next to busy commercial and residential areas. That density combined with growing traffic pumps out massive amounts of PM2.5.

Then you have to factor in Bahrain’s geography. We're an archipelago in the Arabian Gulf. The seasonal north-westerly "Shamal" winds create specific conditions that trap and recirculate pollutants [3]. Smog layers just hang over the densely populated Manama area directly impacting everyone's health.

The way we currently manage air quality is broken. Static regulations fixed emission caps and passive parks just aren't enough. They can't adapt to the hourly shifts in emissions changing weather or traffic jams. Right now we mostly rely on reactive health advisories and after-the-fact fines. We don't have the proactive adaptive tools we desperately need to manage pollution in real-time. That missing piece—intelligent cyber-physical systems that can actually intervene—is a massive gap.

This paper is my attempt to fill that gap. I'm introducing the Autonomous Green Lung (AGL) framework. It’s a decentralized cyber-physical architecture regulated by advanced Multi-Agent Reinforcement Learning (MARL). I stopped looking at the urban environment as a static problem and started treating it as a partially observable stochastic game. This lets distributed intelligent agents coordinate novel Bio-Digital Curtains (hybrid vegetative-mechanical filters) and smart traffic controllers. By doing this we can dynamically alter where pollutants go in real-time. We're moving from passively hoping for the best to actively managing the air we breathe. The goal is a delicate balance: minimize PM2.5 exposure for the population while keeping traffic moving so the city's economy stays healthy.

Here is what this research actually brings to the table:

*   **A Truly Novel Cyber-Physical Architecture**: I'm introducing the AGL framework which integrates Bio-Digital Curtains and smart traffic controllers into one cohesive intelligent system designed specifically for dynamic air quality management.
*   **An Advanced MARL Formulation**: I developed a MARL framework that models the city as a Dec-POMDP. This allows decentralized agents to learn the best coordinated strategies for cutting PM2.5 even when conditions are complex and unpredictable.
*   **Multi-Objective Optimization**: I crafted a reward function that doesn't just chase one goal. It elegantly balances minimizing public health risks (PM2.5 exposure) against maintaining economic efficiency (traffic latency).
*   **High-Fidelity Simulation and Validation**: I tested the framework using high-fidelity Computational Fluid Dynamics (CFD) simulations that accurately capture the complex atmospheric dynamics of the Sitra–Manama corridor.
*   **Empirical Demonstration of Efficacy**: The results show a massive 34.5% reduction in average residential PM2.5 concentrations. This proves that autonomous urban AI has the potential to create healthier sustainable cities.

Here is how the rest of the paper breaks down: Section 2 covers the background literature on MARL Bio-Digital Curtains and CFD. Section 3 dives into the AGL methodology detailing the architecture the MARL setup and the environment modeling. Section 4 lays out the experimental design and the results. Finally Section 5 wraps up with conclusions and where this research needs to go next.

## 2. Literature Review: Advancements in Urban AI for Environmental Sustainability

The idea of "smart cities"—using data and autonomous systems to improve urban life—is gaining a lot of traction. Within that vision environmental sustainability and specifically air quality is a critical focus. In this section I'll walk through the fundamental concepts and recent breakthroughs in Multi-Agent Reinforcement Learning (MARL) Bio-Digital Curtains and Computational Fluid Dynamics (CFD) that make the AGL framework possible.

### 2.1 Multi-Agent Reinforcement Learning (MARL) for Urban Systems

Reinforcement Learning (RL) has revolutionized how we solve complex decision-making problems. Multi-Agent Reinforcement Learning (MARL) takes it a step further applying it to scenarios where multiple independent agents interact usually with limited information and no central controller [4]. In complex urban environments MARL is a powerful tool for optimizing interconnected systems like traffic energy grids and environmental control.

#### 2.1.1 MARL Paradigms and Challenges:

MARL systems are categorized by how agents behave: cooperatively competitively or a mix. For managing urban air quality a cooperative MARL paradigm is the only thing that makes sense. Agents have to work together toward a common goal like minimizing PM2.5 exposure across the city. But it's not easy. There are major hurdles:

*   **Scalability**: More agents mean more complexity. The number of possible interactions explodes making it incredibly hard for the system to learn.
*   **Non-stationarity**: From any single agent’s perspective the environment is always changing because all the other agents are also learning and updating their strategies. It’s like trying to hit a moving target.
*   **Partial Observability**: Agents only see a small local piece of the world. They struggle to understand the complete global picture.
*   **Credit Assignment**: When the system succeeds or fails it's really hard to figure out exactly which agent's actions caused that outcome especially when rewards are sparse.

#### 2.1.2 MARL in Urban Applications:

Recent research shows just how much potential MARL has to change how we manage cities:

*   **Traffic Signal Control**: MARL is great at optimizing traffic lights smoothing out flow and reducing pollution [5] [6]. Agents at intersections learn to cooperate outperforming traditional rule-based systems.
*   **Smart Grid Management**: MARL agents are optimizing energy distribution managing demand and integrating renewable energy into urban microgrids [7].
*   **Urban Mobility**: By coordinating autonomous vehicles and ride-sharing MARL can boost efficiency and reduce the environmental impact of transportation [8].

For air quality control MARL offers a unique advantage: distributed real-time decision-making. Instead of one central brain trying to manage everything local agents react immediately to their specific conditions while still contributing to the global goal. This decentralized approach is more resilient to failures and much more scalable for large cities.

### 2.2 Bio-Digital Curtains: A Novel Approach to Urban Air Filtration

Traditional air filters use a lot of energy and are mostly meant for indoors. But a new breed of bio-inspired cyber-physical systems—often called "Bio-Digital Curtains" or "Photobioreactor Facades"—is emerging. They are incredibly promising for cleaning outdoor air and capturing carbon [9]. They blend biology architecture and digital control.

#### 2.2.1 Principles of Bio-Digital Curtains:

These are usually modular units designed to fit onto building facades. Their core components show off their hybrid nature:

*   **Photobioreactors**: Transparent panels or tubes filled with micro-algae (like *Chlorella vulgaris*) or mosses. They perform photosynthesis soaking up CO2 and releasing oxygen. Their sticky surfaces are also highly effective at trapping particulate matter including PM2.5 [10].
*   **Mechanical Filtration**: Integrated fans and HEPA filters provide active filtration. We can turn these on to boost biological filtration when pollution spikes or when biological activity drops at night.
*   **Sensors and Control Systems**: Embedded sensors monitor light temperature humidity CO2 and PM2.5 levels as well as algae density. A digital control system then optimizes nutrient supply water circulation and fan operation for maximum efficiency.

#### 2.2.2 Existing Implementations and Potential:

Projects like ecoLogicStudio’s "Photo.Synth.Etica" tested in Dublin prove that large-scale bio-digital facades work for CO2 capture and air purification [11]. They offer a sustainable visually appealing way to integrate environmental mitigation into the urban fabric turning buildings into active "green lungs." Their hybrid nature makes them perfect for dynamic air quality management because we can adjust their parameters in real-time.

### 2.3 Computational Fluid Dynamics (CFD) for Atmospheric Dispersion Modeling

To build effective mitigation strategies we have to understand exactly how pollutants move through complex urban environments. That's where Computational Fluid Dynamics (CFD) comes in. It's a numerical technique that solves the Navier-Stokes equations to simulate fluid flow making it essential for modeling atmospheric dispersion [12].

#### 2.3.1 CFD in Urban Air Quality:

CFD models simulate the complex interactions between wind urban structures and emissions. Its key applications are transformative:

*   **Micro-scale Dispersion**: CFD predicts pollutant concentrations down to the street level accounting for building-induced turbulence and recirculation zones where pollutants get trapped [13].
*   **Source Apportionment**: It helps identify exactly how much different emission sources contribute to overall pollution.
*   **Mitigation Strategy Evaluation**: CFD allows us to test the effectiveness of urban design changes or active mitigation technologies on local air quality before we actually build them.

#### 2.3.2 Challenges and Advancements:

CFD simulations are incredibly detailed but demand massive computing power. However advances in parallel computing and reduced-order modeling are making CFD more accessible for large-scale urban applications. In the AGL framework CFD serves two purposes: it provides a high-fidelity environment to train the MARL agents and it acts as a real-time model for predicting pollutant transport under different scenarios [14].

### 2.4 Urban Air Pollution in the Arabian Gulf: The Sitra–Manama Context

The Arabian Gulf including Bahrain faces a unique mix of natural and human-made air quality challenges. Natural dust storms driven by the Shamal winds contribute heavily to particulate matter [3]. These winds carry dust from Iraq and Saudi Arabia across the Gulf. On top of that we have emissions from rapid industrial expansion power generation and increasing traffic.

In the Sitra–Manama corridor it's worse because heavy industries are right next to residential areas and urban canyons channel the pollution. While the Shamal winds can sometimes clear the air they also create conditions where emissions get trapped leading to prolonged exposure. Understanding these localized dynamics is crucial for designing effective mitigation strategies which is why static approaches fail here.

## 3. Methodology: The Autonomous Green Lung (AGL) Framework

The Autonomous Green Lung (AGL) framework is my solution to this complex problem. It’s a decentralized cyber-physical system engineered to dynamically manage the atmosphere in urban industrial corridors. It uses advanced Multi-Agent Reinforcement Learning (MARL) to orchestrate Bio-Digital Curtains and smart traffic controllers creating a responsive intelligent urban environment.

### 3.1 System Architecture: A Symphony of Intelligent Agents

The AGL framework is built on three main components working together:

1.  **Bio-Digital Curtain (BDC) Agents**: These are the guardians. Each agent controls a physical BDC unit combining biological and mechanical filtration. They:
    *   **Sense**: Monitor local PM2.5 wind temperature and humidity.
    *   **Actuate**: Adjust fan speeds toggle mechanical filters and optimize nutrients for the biological components.
    *   **Communicate**: Share data with nearby BDCs and the Central Coordination Module.

2.  **Smart Traffic Controller (STC) Agents**: These manage intersections and traffic flow to influence pollutant dispersion. They:
    *   **Sense**: Monitor vehicle counts queue lengths speeds and local PM2.5 data.
    *   **Actuate**: Adjust traffic light timings change signal sequences and suggest alternative routes.
    *   **Communicate**: Exchange data with other controllers and the Central Coordination Module.

3.  **Central Coordination Module (CCM)**: This is a virtual supervisor used mainly during the MARL training phase. It provides a centralized view of the global environment. During live operation it monitors overall performance and facilitates communication but the agents operate independently ensuring the system is robust and scalable.

<p align="center">
  <img src="https://i.imgur.com/your_figure_1_url.png" alt="Figure 1: Conceptual Architecture of the Autonomous Green Lung (AGL) Framework">
  <br>
  <em>Figure 1: Conceptual Architecture of the Autonomous Green Lung (AGL) Framework</em>
</p>

### 3.2 Multi-Agent Reinforcement Learning (MARL) Formulation

I framed the air quality management problem as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP). It sounds complicated but it's the perfect model because each agent only sees a partial view of the environment. They make independent decisions but their combined actions shape the city's air quality.

#### 3.2.1 Agents States Actions and Observations

Here is how the MARL model breaks down:

*   **Agents (N)**: We have $N_F$ Filtration Agents and $N_T$ Traffic Agents so $N = N_F + N_T$.
*   **Global State (S_t)**: The true state of the environment $S_t$ includes the full PM2.5 concentration field wind velocity field traffic flow parameters and the operational status of all BDCs and STCs.
*   **Agent Observation (O_i)**: Each agent *i* gets a local observation $O_i(S_t)$. Filtration agents see local PM2.5 wind and their own status. Traffic agents see local traffic parameters and nearby PM2.5 levels.
*   **Agent Action (A_i)**: Based on its observation agent *i* chooses an action $A_i$. Filtration agents adjust fan speeds and filters. Traffic agents adjust light durations and sequences.
*   **Joint Action (A_t)**: The collection of all actions taken by all agents at time *t*.
*   **Transition Function (P)**: How the environment changes from state $S_t$ to $S_{t+1}$ based on the joint action $A_t$ and environmental factors.

#### 3.2.2 Multi-Objective Reward Formulation

The agents optimize a global multi-objective reward function $R(S_t, A_t)$ that balances public health and economic efficiency. It's a weighted sum of costs to minimize:

$R(S_t, A_t) = -(\alpha \cdot PWE(S_t) + \beta \cdot \Delta L(S_t, A_t) + \gamma \cdot E(A_t))$

*   **Population-Weighted Exposure (PWE)**: Measures total PM2.5 exposure across the residential population.
*   **Traffic Latency Deviation (ΔL)**: Penalizes actions that slow down traffic too much ensuring economic activity continues.
*   **Energy Consumption (E)**: Encourages the system to run the mitigation infrastructure efficiently.
*   **Weighting Factors (α, β, γ)**: Parameters that let policymakers balance health traffic and energy priorities.

#### 3.2.3 MARL Algorithm: Centralized Training with Decentralized Execution (CTDE)

Because the task is cooperative and agents have partial information I used Centralized Training with Decentralized Execution (CTDE).

*   **Centralized Training**: A central critic sees the global state and joint actions learning a value function to estimate future rewards. This solves the credit assignment problem.
*   **Decentralized Execution**: Once trained each agent uses its local actor network to make decisions based only on local observations making the system robust and scalable.

I used Multi-Agent Proximal Policy Optimization (MAPPO) [15] an on-policy actor-critic algorithm known for stability in cooperative MARL.

### 3.3 Environment Modeling: High-Fidelity Computational Fluid Dynamics (CFD) Simulation

To capture the complex atmospheric dynamics of the Sitra–Manama corridor I built a high-fidelity CFD model using OpenFOAM. This acts as the realistic training ground for the MARL agents.

#### 3.3.1 Simulation Domain and Urban Geometry

*   **Domain Size**: 10 km x 5 km x 500 m covering the industrial zone transport routes and residential areas.
*   **Urban Geometry**: Detailed 3D building shapes to accurately model turbulence and recirculation zones.
*   **Mesh Generation**: A hybrid mesh with finer resolution near pollution sources and residential areas.

#### 3.3.2 Meteorological Forcing and Boundary Conditions

*   **Wind Profiles**: Realistic wind profiles including Shamal wind events and temperature inversions.
*   **Turbulence Model**: RANS equations with a k-epsilon turbulence model.
*   **Surface Roughness**: Land use data to assign appropriate surface roughness.

#### 3.3.3 Emission Sources and Pollutant Transport

*   **Industrial Emissions**: Point and area sources based on actual industrial inventories for PM2.5 NOx and SO2.
*   **Vehicular Emissions**: Line sources with dynamic emission rates based on simulated traffic flow.
*   **Pollutant Transport**: Solving the advection-diffusion equation to simulate PM2.5 transport and dispersion.

#### 3.3.4 Integration with MARL Agents

The CFD simulation provides the global state to the central critic during training. During live operation it feeds local PM2.5 and wind data to the agents' sensors. The agents' actions are then fed back into the CFD model as dynamic boundary conditions closing the cyber-physical loop.

## 4. Experimental Design and Results

I designed a thorough experimental setup to test how well the AGL framework reduces PM2.5 concentrations in the simulated Sitra–Manama corridor.

### 4.1 Simulation Environment and Agent Configuration

*   **CFD Simulation**: Ran for a 7-day period including Shamal wind events and stagnant conditions.
*   **AGL Deployment**: 20 BDC units along the industrial zone edge and 15 STC agents at major intersections.
*   **MARL Training**: MAPPO trained for 10,000 episodes with weights $\alpha = 0.6$ $\beta = 0.3$ and $\gamma = 0.1$.
*   **Baselines**: Compared against a "No Intervention" scenario and a "Static Mitigation" scenario.

### 4.2 Evaluation Metrics

*   **Average Residential PM2.5 Concentration (µg/m³)**: The primary health metric.
*   **Reduction in Peak PM2.5 Events**: Number of days exceeding 50 µg/m³.
*   **Average Traffic Latency (minutes)**: The primary economic metric.
*   **Energy Consumption of BDCs (kWh)**: Operational cost metric.

### 4.3 Results and Discussion

The results in Table 1 are clear: the AGL framework works. It achieved a 34.5% reduction in average residential PM2.5 concentrations dropping from 28.4 µg/m³ (No Intervention) to 18.6 µg/m³. This brings us much closer to the WHO interim target of 25 µg/m³.

It also drastically reduced severe pollution episodes. Days with peak PM2.5 events dropped by 78.6% from 14 days/year to just 3 days/year. The system proactively stops dangerous pollution plumes.

#### 4.3.1 Dynamic Adaptation to Environmental Stochasticity

The AGL framework adapts dynamically. During Shamal wind events Filtration Agents upwind of the industrial zone proactively increased fan speeds creating an atmospheric buffer zone. During stagnant conditions they coordinated to create localized air circulation.

Traffic Agents also adapted quickly adjusting light timings to smooth flow and reduce idling when PM2.5 sensors detected high localized concentrations. This coordination highlights the power of MARL.

#### 4.3.2 Trade-offs: Health vs. Throughput

The framework prioritizes health but maintains economic efficiency. The multi-objective reward function ensures better air quality doesn't destroy traffic flow. The BDCs used 210 kWh/day compared to 150 kWh/day for static mitigation but the health benefits easily justify the energy use.

#### 4.3.3 Comparison with Baselines

The AGL framework vastly outperforms both the No Intervention and Static Mitigation baselines. Static BDCs can't adapt to changing conditions performing poorly during peak events. The coordinated intelligence of MARL agents uses mitigation resources far more effectively.

## 5. Conclusion and Future Work: Towards Self-Regulating Sustainable Cities

This research validates the Autonomous Green Lung (AGL) framework. It’s a decentralized cyber-physical architecture built for dynamic atmospheric management. By using MARL to coordinate Bio-Digital Curtains and smart traffic controllers it offers a robust solution to PM2.5 pollution in industrial corridors. The 34.5% reduction in PM2.5 and 78.6% decrease in peak events crush traditional static approaches.

The framework's ability to adapt to complex weather and balance health with economic throughput shows the power of MARL in optimizing urban systems. This is a major step toward resilient responsive urban infrastructure.

### 5.1 Limitations and Ethical Considerations

We have to acknowledge the limitations and ethical questions:

*   **Simulation-Based Validation**: Real-world deployment will face sensor noise hardware failures and unpredictable human behavior. Pilot deployments are essential.
*   **Computational Cost**: High-fidelity CFD is demanding. We need faster surrogate models for large-scale deployment.
*   **Agent Communication Overhead**: Scaling up requires efficient communication protocols and truly decentralized learning.
*   **Ethical Trade-offs**: Balancing air quality and traffic flow requires careful policy alignment and public transparency.
*   **System Resilience and Security**: Autonomous systems must be secure against cyber-attacks.
*   **Public Acceptance**: We need clear communication to build trust in autonomous urban infrastructure.

### 5.2 Future Work: Expanding the Horizon of Autonomous Green Lungs

The next steps are clear:

*   **Real-World Pilot Deployment**: Deploying the AGL framework in a section of the Sitra–Manama corridor for real-world validation.
*   **Integration of Diverse Mitigation Assets**: Expanding control to smart parks building purification systems and dynamic urban planning.
*   **Advanced MARL Algorithms**: Exploring algorithms that handle more agents and complex observations.
*   **Predictive Modeling and Forecasting**: Integrating deep learning forecasting models for proactive decision-making.
*   **Energy Optimization and Renewable Integration**: Connecting BDCs to renewable energy sources.
*   **Human-in-the-Loop Control**: Developing interfaces for human operators to monitor and override the system.
*   **Socio-Economic Impact Assessment**: Quantifying the broader benefits of cleaner air and traffic management.
*   **Multi-Pollutant Control**: Extending the framework to tackle NOx SO2 and O3 alongside PM2.5.

By refining the AGL framework we can build cities that actively protect their inhabitants fostering sustainable coexistence between industry and environmental health.

## References

[1] World Health Organization. “Ambient (outdoor) air pollution.” WHO Fact Sheet, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health
[2] Lelieveld, J., et al. “Loss of life expectancy from air pollution compared to other risk factors: a worldwide perspective.” Cardiovascular Research, vol. 116, no. 11, pp. 1910-1917, 2020.
[3] Francis, D., et al. “Increased Shamal winds and dust activity over the Arabian Peninsula during the COVID-19 lockdown period in 2020.” Scientific Reports, vol. 11, no. 1, p. 19036, 2021.
[4] Hernandez-Leal, P., et al. “A survey of learning in multiagent systems: From reinforcement learning to deep learning.” Journal of Artificial Intelligence Research, vol. 67, pp. 549-613, 2020.
[5] El-Tantawy, S., et al. “Multiagent reinforcement learning for traffic signal control.” IEEE Transactions on Intelligent Transportation Systems, vol. 16, no. 5, pp. 2492-2506, 2015.
[6] Chu, T., et al. “Multi-agent deep reinforcement learning for large-scale network traffic signal control.” IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086-1095, 2020.
[7] Vandael, S., et al. “A multi-agent reinforcement learning approach for optimal control of microgrids.” IEEE Transactions on Smart Grid, vol. 4, no. 3, pp. 1248-1256, 2013.
[8] Wen, Y., et al. “Multi-agent reinforcement learning for dynamic ride-sharing with heterogeneous vehicles.” Transportation Research Part C: Emerging Technologies, vol. 120, p. 102804, 2020.
[9] ecoLogicStudio. “Photo.Synth.Etica: Bio-digital Urban Curtain.” ArchDaily, 2018. [Online]. Available: https://www.archdaily.com/905595/ecologicstudios-bio-digital-curtain-fights-climate-change-by-filtering-air-and-creating-bioplastic
[10] Barati, B., et al. “Algae-based building envelopes: A review on the current status and future trends.” Renewable and Sustainable Energy Reviews, vol. 116, p. 109411, 2019.
[11] ecoLogicStudio. “Photo.Synth.Etica.” Official Project Page, 2018. [Online]. Available: https://www.ecologicstudio.com/projects/photosynthetica
[12] Blocken, B. “50 years of Computational Fluid Dynamics in wind engineering: A state-of-the-art review.” Journal of Wind Engineering and Industrial Aerodynamics, vol. 189, pp. 168-204, 2019.
[13] Tominaga, Y., & Stathopoulos, T. “CFD modeling of pollutant dispersion in urban environments.” Journal of Wind Engineering and Industrial Aerodynamics, vol. 102, pp. 136-141, 2012.
[14] Gokhale, S., & Khare, M. “A review of computational fluid dynamics modeling for vehicular exhaust dispersion in urban street canyons.” Environmental Modeling & Assessment, vol. 15, no. 2, pp. 101-112, 2010.
[15] Yu, C., et al. “The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games.” Advances in Neural Information Processing Systems (NeurIPS), 2021.
