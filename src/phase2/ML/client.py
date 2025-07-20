#!/usr/bin/env python
import sys
sys.path.append("gen-py")

from thrift import Thrift
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from coordinator import Coordinator

def main():
    
    if len(sys.argv) < 8:
        print("Usage: python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs> <h> <eta>")
        sys.exit(1)

    coordinator_ip = sys.argv[1]
    coordinator_port = int(sys.argv[2])
    dir_path = sys.argv[3]
    rounds = int(sys.argv[4])
    epochs = int(sys.argv[5])
    h = int(sys.argv[6])
    eta = float(sys.argv[7])

    transport = TSocket.TSocket(coordinator_ip, coordinator_port)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Coordinator.Client(protocol)

    transport.open()
    k = 26
    final_err = client.train(dir_path, rounds, epochs, h, k, eta)
    print(f"Client: Final validation error => {final_err}")

    transport.close()

if __name__ == '__main__':
    main()
