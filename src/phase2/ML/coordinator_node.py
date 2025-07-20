import os
import sys
import glob
import threading
import queue
import numpy as np

from thrift.server import TServer
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift import Thrift

sys.path.append("gen-py")
from coordinator import Coordinator

import ML
from computenode import ComputeNode
from computenode.ttypes import Weights, TrainResult


def read_compute_nodes(filename="compute_nodes.txt"):
    nodes = []
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    host, port = line.split(',')
                    nodes.append((host.strip(), int(port.strip())))
    except Exception as e:
        print(f"Error reading compute nodes text file: {e}")
    return nodes


def connect_to_node(host, port):
    try:
        transport = TSocket.TSocket(host, port)
        transport.setTimeout(200000)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = ComputeNode.Client(protocol)
        transport.open()
        return client, transport
    except Exception as e:
        print(f"Coordinator: Failed to connect to {host}:{port} due to {e}")
        return None, None


def dispatch_task_with_model(host, port, training_file, init_weights, k, h, eta, epochs, scheduling_policy):

    client, transport = connect_to_node(host, port)
    if not client:
        return None

    try:
        result = client.train_with_weights(
            training_file,
            init_weights,
            0.0,
            k, 
            h, 
            eta, 
            epochs,
            scheduling_policy
        )
        # print(f"Coordinator: success calling train_with_weights on {host}:{port}, {training_file}")
        transport.close()
        return result
    except Exception as e:
        transport.close()
        return None


class CoordinatorHandler:
    def __init__(self, scheduling_policy):
        self.scheduling_policy = scheduling_policy
        self.model = ML.mlp()
        self.compute_nodes = read_compute_nodes()

    def train(self, dir, rounds, epochs, h, k, eta):

        training_files = glob.glob(os.path.join(dir, "train_letters*.txt"))
        if not training_files:
            print("Coordinator: No training files found in directory.")
            return -1.0

        self.model.init_training_random(training_files[0], k, h)

        for r in range(rounds):
            print(f"\n\nCoordinator: Starting round {r+1} / {rounds}...")

            work_queue = queue.Queue()
            for tfile in training_files:
                work_queue.put(tfile)

            results = []
            results_lock = threading.Lock()

            node_idx_lock = threading.Lock()
            self.next_node_idx = 0

            curr_V, curr_W = self.model.get_weights()
            init_weights = Weights(V=curr_V, W=curr_W)

            def worker():
                while True:
                    try:
                        tfile = work_queue.get_nowait()
                    except queue.Empty:
                        break

                    accepted = False
                    num_nodes = len(self.compute_nodes)
                    for attempt in range(num_nodes):
                        with node_idx_lock:
                            idx = self.next_node_idx
                            self.next_node_idx = (self.next_node_idx + 1) % num_nodes
                        host, port = self.compute_nodes[idx]

                        res = dispatch_task_with_model(
                            host, port,
                            tfile, init_weights, 
                            k, h, eta, epochs,
                            self.scheduling_policy
                        )
                        if res and res.accepted:
                            with results_lock:
                                print(f"Coordinator: Completed training {tfile} on {host}:{port}.")
                                results.append(res)
                            accepted = True
                            break

                    if not accepted:
                        work_queue.put(tfile)
                    work_queue.task_done()

            num_workers = len(self.compute_nodes) + 2
            threads = []
            for _ in range(num_workers):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if not results:
                print(f"Coordinator: results are empty in this round")
            else:
                sum_V = None
                sum_W = None
                for res in results:
                    gradV = np.array(res.weights.V, dtype=float)
                    gradW = np.array(res.weights.W, dtype=float)
                    if sum_V is None:
                        sum_V = gradV
                        sum_W = gradW
                    else:
                        sum_V += gradV
                        sum_W += gradW
                count = len(results)
                avg_V = (sum_V / count).tolist()
                avg_W = (sum_W / count).tolist()

                self.model.update_weights(avg_V, avg_W)

            val_file = os.path.join(dir, "validate_letters.txt")
            val_err = self.model.validate(val_file)
            print(f"Coordinator: Round {r+1} => Validation error={val_err}")

        final_err = self.model.validate(os.path.join(dir, "validate_letters.txt"))
        print(f"Coordinator: Final validation error => {final_err}")
        return final_err


def main():

    if len(sys.argv) < 3:
        print("Usage: python3 coordinatorNode.py <port> <scheduling_policy>")
        sys.exit(1)

    port = int(sys.argv[1])
    scheduling_policy = int(sys.argv[2])

    handler = CoordinatorHandler(scheduling_policy)
    processor = Coordinator.Processor(handler)

    transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()


if __name__ == '__main__':
    main()