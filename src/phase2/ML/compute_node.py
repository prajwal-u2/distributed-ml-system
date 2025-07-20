#!/usr/bin/env python
import sys
import time
import random
import numpy as np

sys.path.append("gen-py")

from thrift import Thrift
from thrift.server import TServer
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol

from computenode import ComputeNode
from computenode.ttypes import Weights, TrainResult

import ML

class ComputeNodeHandler:
    def __init__(self, load_probability):
        self.node_load_probability = load_probability

    def train_with_weights(self, training_file, initial_weights, load_probability, k, h, eta, epochs, scheduling_policy):

        if random.random() < self.node_load_probability:
            sleep_time = 3
            print(f"ComputeNode: Injecting load: sleeping {sleep_time} seconds.")
            time.sleep(sleep_time)

        if scheduling_policy == 2:
            if random.random() < self.node_load_probability:
                print("ComputeNode: Load Balancing - Task Rejected.")
                return TrainResult(accepted=False, training_error=-1.0, weights=Weights(V=[], W=[]))
            else:
                print("ComputeNode: Load Balancing - Task accepted.")
        else:
            print("ComputeNode: Random scheduling - Task accepted.")

        model = ML.mlp()
        success = model.init_training_model(training_file, initial_weights.V, initial_weights.W)
        if not success:
            print("Compute Node: Error initializing model with provided weights.")
            return TrainResult(accepted=False, training_error=-1.0, weights=Weights(V=[], W=[]))

        init_V, init_W = model.get_weights()
        init_V_np = np.array(init_V, dtype=float)
        init_W_np = np.array(init_W, dtype=float)

        training_error = model.train(eta, epochs)
        print(f"CopmuteNode: Training complete - {training_file} : Error={training_error}")

        final_V, final_W = model.get_weights()
        final_V_np = np.array(final_V, dtype=float)
        final_W_np = np.array(final_W, dtype=float)

        grad_V = (final_V_np - init_V_np).tolist()
        grad_W = (final_W_np - init_W_np).tolist()

        return TrainResult(
            accepted=True,
            training_error=training_error,
            weights=Weights(V=grad_V, W=grad_W)
        )

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 computenode.py <port> <load_probability>")
        sys.exit(1)

    port = int(sys.argv[1])
    node_load_probability = float(sys.argv[2])

    handler = ComputeNodeHandler(node_load_probability)
    processor = ComputeNode.Processor(handler)
    transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print(f"Compute Node: Starting server on port {port} with load_probability={node_load_probability}...")
    server.serve()

if __name__ == '__main__':
    main()
