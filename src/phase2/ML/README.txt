# README


1. Unzip the Folder
Extract the zipped folder into a local directory of your choice.


2. Navigate to the ML Directory
Move into the `phase2/ML/` directory within the unzipped folder.


3. Create Virtual Environment and Install Dependencies
Run the following commands to set up a virtual environment and install the required dependencies:

python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt



4. Configure Compute Nodes
Open the `compute_nodes.txt` file (if used) and ensure it lists the host and port for each compute node you plan to run.

Example:
localhost,9091
localhost,9092
localhost,9093


5. Start the Compute Nodes
Open a terminal for each compute node and execute the following command:
python3 compute_node.py <port> <load_probability>

Example:
python3 compute_node.py 9091 0.2
python3 compute_node.py 9092 0.4

Each node will start listening on its respective port and be ready to accept tasks from the coordinator.


6. Start the Coordinator
Run the following command to start the coordinator:
python3 coordinator_node.py <port> <scheduling_policy>

Example:
python3 coordinator_node.py 9095 2

Here, `9095` is the coordinator’s port, and `2` represents the load-balancing policy.


7. Run the Client
Once the coordinator and compute nodes are running, start the client using:
python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs> <h> <eta>

Example:
python3 client.py localhost 9095 letters 3 75 24 0.0001


8. View Final Validation Error
After training is complete, the final validation error will be displayed in the client console.

