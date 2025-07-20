# Distributed ML System

A distributed machine learning system that implements a coordinator-compute node architecture for training and validating machine learning models across multiple nodes.

## Project Overview

This project implements a distributed machine learning system with the following key components:

- **Coordinator Node**: Manages task distribution and load balancing across compute nodes
- **Compute Nodes**: Individual nodes that perform ML training tasks
- **Client**: Initiates training requests and manages the overall training process
- **ML Model**: A custom machine learning implementation for letter recognition

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │ Coordinator │    │ Compute     │
│             │◄──►│   Node      │◄──►│ Node 1      │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │ Compute     │
                   │ Node 2      │
                   └─────────────┘
```

## Features

- **Distributed Training**: Train ML models across multiple compute nodes
- **Load Balancing**: Multiple scheduling policies for optimal resource utilization
- **Fault Tolerance**: System continues operation even if some nodes fail
- **Scalable Architecture**: Easy to add or remove compute nodes
- **Letter Recognition**: ML model trained on letter classification dataset

## Prerequisites

- Python 3.x
- pip (Python package installer)
- Network connectivity between nodes (for distributed setup)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd distributed-ml-system
   ```

2. **Navigate to the ML directory**
   ```bash
   cd src/phase2/ML
   ```

3. **Create and activate virtual environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Steps to Execute

### 1. Unzip the Folder
a. Extract the zipped folder (e.g., `distributed_training.zip`) into a local directory of your choice.
b. Navigate to `phase2/ML/` within the unzipped folder.

### 2. Create Virtual Environment and Install Requirements
a. Create virtual environment:
   ```bash
   python3 -m venv myenv
   ```
b. Activate virtual environment:
   ```bash
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
c. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
d. **Important**: Execute the compute nodes, coordinator node, and client in the virtual environment.

### 3. Configure Compute Nodes
a. Open the file `compute_nodes.txt` (if used) and ensure it lists the host and port for each compute node you plan to run.

**Example:**
```
localhost,9091
localhost,9092
localhost,9093
```

### 4. Start the Compute Nodes
Open additional terminals—one for each compute node.

```bash
python3 compute_node.py <port> <load_probability>
```

**Examples:**
```bash
python3 compute_node.py 9091 0.2
python3 compute_node.py 9092 0.4
```

Each node will start listening on its respective port and be ready to accept tasks from the coordinator.

**Parameters:**
- `port`: The port number for the compute node
- `load_probability`: Probability of node being busy (0.0 to 1.0)

### 5. Start the Coordinator
```bash
python3 coordinator_node.py <port> <scheduling_policy>
```

**Example:**
```bash
python3 coordinator_node.py 9095 2
```

Here, `9095` is the coordinator's port, and `2` represents the load-balancing policy.

**Parameters:**
- `port`: The port number for the coordinator
- `scheduling_policy`: 
  - `1`: Random
  - `2`: Load-balancing

### 6. Run the Client
Once the coordinator and compute nodes are running, open another terminal for the client.

```bash
python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs> <h> <eta>
```

**Example:**
```bash
python3 client.py localhost 9095 letters 3 75 24 0.0001
```

**Parameters:**
- `coordinator_ip`: IP address of the coordinator node
- `coordinator_port`: Port of the coordinator node
- `dir_path`: Directory containing training data (e.g., "letters")
- `rounds`: Number of training rounds
- `epochs`: Number of epochs per round
- `h`: Hidden layer size
- `eta`: Learning rate

### 7. View Results
You can find the final validation error after training completes.

## Data Format

The system expects training data in the following format:
- Training files: `train_letters1.txt`, `train_letters2.txt`, etc.
- Validation file: `validate_letters.txt`

Each file should contain letter classification data with features and labels.

## Project Structure

```
distributed-ml-system/
├── README.md                 # This file
├── reports/                  # Project documentation and reports
│   ├── Distributed ML System Design Document.pdf
│   ├── phase_1_document.pdf
│   └── metadata.yml
└── src/
    ├── ML/                   # Phase 1 implementation
    │   ├── ML.cpp
    │   ├── ML.hpp
    │   ├── ML.py
    │   └── letters/          # Training data
    └── phase2/               # Phase 2 implementation
        └── ML/
            ├── client.py
            ├── compute_node.py
            ├── coordinator_node.py
            ├── compute_node.thrift
            ├── coordinator.thrift
            ├── gen-py/        # Generated Thrift code
            ├── letters/       # Training data
            ├── ML.cpp
            ├── ML.hpp
            ├── ML.py
            ├── requirements.txt
            └── README.txt
```

## Technologies Used

- **Python**: Main programming language
- **Apache Thrift**: RPC framework for distributed communication
- **NumPy**: Numerical computing library
- **C++**: Core ML algorithm implementation

## Scheduling Policies

1. **Round-robin**: Distributes tasks sequentially across nodes
2. **Load-balancing**: Sends tasks to the least busy node
3. **Random**: Randomly selects a compute node

## Monitoring and Debugging

- Check terminal output for each component for status messages
- The client will display final validation error after training
- Use `timing.txt` for performance analysis

## Troubleshooting

1. **Connection Issues**: Ensure all nodes are on the same network and ports are not blocked
2. **Import Errors**: Make sure virtual environment is activated and dependencies are installed
3. **Data Issues**: Verify training data files are in the correct format and location

## Performance Considerations

- Adjust `load_probability` to simulate different node capacities
- Use appropriate `epochs` and `rounds` values for your dataset
- Monitor system performance using the timing information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of a Distributed Systems course assignment.

## Author

**Prajwal Umesha** 
**Poorna B S**

## Acknowledgments

- University of Minnesota - Distributed Systems Course
- Apache Thrift for RPC framework
- NumPy community for numerical computing tools
