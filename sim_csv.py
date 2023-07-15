""" CSC 446
    Assignment 3
    Owen Thurston V00944754
    Cam Rohwer V00970474
    Andrei Mazilescu V00796396

    Attribution:
    The Original Code for Node and Splaytree Classes were adapted from:
    https://github.com/anoopj/pysplay/blob/master/splay.py
    - modifications done to allow event to be stored
    - Methods added to SplayTree Class
"""
import numpy as np
from numpy.random import default_rng
import itertools
from prettytable import PrettyTable
from pprint import pprint  # DEBUG
import csv


class Customer:
    """Customer object keeps track of customer's id"""

    generate_id = itertools.count()

    def __init__(self):
        self.id = next(Customer.generate_id)
        self.server = None
        self.arrival = None
        self.departure = None

    def __str__(self) -> str:
        return f"Customer {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def getServer(self):
        return self.server


class Event:
    generate_id = itertools.count()

    def __init__(self, etype: str, time, cust: Customer):
        self.id = next(Event.generate_id)
        self.cust = cust
        self.etype = etype
        self.time = time
        self.service_duration = None
        self.service_start_time = None

    def __eq__(self, other):
        return self.time == other.time and self.etype == other.etype

    def __lt__(self, other):
        # TODO: departures before arrivals
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time

    def __str__(self):  # for debug
        return f"Event ID: {self.id} Customer ID: {self.cust.id} Event Type: {self.etype} at Time: {self.time:.2f} in Server{self.cust.server}"

    def __repr__(self):
        return self.__str__()

    def getType(self):
        return self.etype

    def getTime(self):
        return self.time

    def getId(self):
        return self.id


class Node:
    def __init__(self, event):
        self.event = event
        self.left = self.right = None

    def equals(self, event):
        return self.event == Node.event


class SplayTree:
    def __init__(self):
        self.root = None
        self.header = Node(None)  # For splay()

    def enqueue(self, event):
        if self.root == None:
            self.root = Node(event)
            return

        self.splay(event)
        if self.root.event == event:
            # If the key is already there in the tree, don't do anything.
            return

        n = Node(event)
        if event < self.root.event:
            n.left = self.root.left
            n.right = self.root
            self.root.left = None
        else:
            n.right = self.root.right
            n.left = self.root
            self.root.right = None
        self.root = n

    def remove(self, event):
        self.splay(event)
        if event != self.root.event:
            raise "key not found in tree"

        # Now delete the root.
        if self.root.left == None:
            self.root = self.root.right
        else:
            x = self.root.right
            self.root = self.root.left
            self.splay(event)
            self.root.right = x

    def findMin(self):
        if self.root == None:
            return None
        x = self.root
        while x.left != None:
            x = x.left
        self.splay(x.event)
        return x.event

    def findMax(self):
        if self.root == None:
            return None
        x = self.root
        while x.right != None:
            x = x.right
        self.splay(x.event)
        return x.event

    def find(self, event):
        if self.root == None:
            return None
        self.splay(event)
        if self.root.event != event:
            return None
        return self.root.event

    def isEmpty(self):
        return self.root == None

    def splay(self, event):
        l = r = self.header
        t = self.root
        self.header.left = self.header.right = None
        while True:
            if event < t.event:
                if t.left == None:
                    break
                if event < t.left.event:
                    y = t.left
                    t.left = y.right
                    y.right = t
                    t = y
                    if t.left == None:
                        break
                r.left = t
                r = t
                t = t.left
            elif event > t.event:
                if t.right == None:
                    break
                if event > t.right.event:
                    y = t.right
                    t.right = y.left
                    y.left = t
                    t = y
                    if t.right == None:
                        break
                l.right = t
                l = t
                t = t.right
            else:
                break
        l.right = t.left
        r.left = t.right
        t.left = self.header.right
        t.right = self.header.left
        self.root = t

    def dequeue(self):
        if self.isEmpty() == True:
            return
        # Now delete the root.
        if self.root.left == None:
            self.root = self.root.right
        else:
            x = self.root.right
            self.root = self.root.left
            self.splay(x.event)
            self.root.right = x


def Sim(total_customers, num_servers, rng, MeanInterArrivalTime, mu, load_balancer, d, print_table=False):
    # Initialize parameters
    parameters = {
        "NumberOfDepartures": 0,
        "Clock": 0.0,
        "NumberInService": np.zeros(num_servers),
        "LastEventTime": 0.0,
        "TotalBusy": 0,
        "MaxQueueLength": np.zeros(num_servers),
        "SumResponseTime": 0,
        "PreviousServer": 0,  # Used for RR
        "rng": rng,
        "load_balancer": load_balancer,
        "rand_min_d": d,
        "num_servers": num_servers,
        'MeanInterArrivalTime': MeanInterArrivalTime,
        'mu': mu,
        "total_customers": total_customers
    }

    # Initailize data structures
    #print("D: " + str(parameters["rand_min_d"]))
    future_event_list = SplayTree()
    customer_queue = [[] for x in range(num_servers)]
    finishedEvents = []

    # Initialize FEL with first customer
    first_cust = Customer()
    evt = Event("arrival", 0.00, first_cust)
    first_cust.arrival = evt
    future_event_list.enqueue(evt)

    while not future_event_list.isEmpty():
        evt = future_event_list.findMin()
        future_event_list.dequeue()
        parameters["Clock"] = evt.getTime()
        if evt.getType() == "arrival":
            ProcessArrival(evt, future_event_list, customer_queue, parameters)
        else:
            ProcessDeparture(evt, future_event_list, customer_queue, parameters)
        finishedEvents.append(evt)

    # ReportGeneration(finishedEvents, parameters)
    if print_table == True:
        #print_simulation_table(finishedEvents, parameters)
        #print_event_list(finishedEvents, parameters)
        ReportGeneration(finishedEvents,parameters)


def ProcessArrival(evt, future_event_list, customer_queue, parameters):
    # Dispatch event to correct server
    if parameters["load_balancer"] is RandMin:
        # RandMin need customer_queue as an additional parameter to check
        #   queue length
        evt.cust.server = parameters["load_balancer"](parameters, customer_queue)
    else:
        evt.cust.server = parameters["load_balancer"](parameters)

    customer_queue[evt.cust.server].append(evt)

    if parameters["NumberInService"][evt.cust.server] == 0:
        ScheduleDeparture(evt, future_event_list, parameters)
    else:
        parameters["TotalBusy"] += parameters["Clock"] - parameters["LastEventTime"]

    if parameters["MaxQueueLength"][evt.cust.server] < len(customer_queue[evt.cust.server]):
        parameters["MaxQueueLength"][evt.cust.server] = len(customer_queue[evt.cust.server])

    next_customer = Customer()
    # No arrival event added to FEL once total number of customers already arrived
    if not next_customer.id > parameters['total_customers'] - 1:
        next_arrival = Event(
            "arrival",
            parameters["Clock"]
            + parameters["rng"].exponential(scale=parameters['MeanInterArrivalTime']),
            next_customer,
        )
        next_customer.arrival = next_arrival
        future_event_list.enqueue(next_arrival)
        parameters["LastEventTime"] = parameters["Clock"]


def ScheduleDeparture(evt, future_event_list, parameters):
    ServiceTime = parameters["rng"].exponential(scale=parameters['mu'])

    depart = Event("departure", parameters["Clock"] + ServiceTime, evt.cust)
    depart.service_start_time = parameters["Clock"]
    depart.service_duration = ServiceTime
    evt.cust.departure = depart
    future_event_list.enqueue(depart)
    parameters["NumberInService"][evt.cust.server] = 1


def ProcessDeparture(evt, future_event_list, customer_queue, parameters):
    # Processing departure by server
    finished = customer_queue[evt.cust.server].pop(0)

    if len(customer_queue[finished.cust.server]) > 0:
        next_cust = customer_queue[finished.cust.server][0]
        ScheduleDeparture(next_cust, future_event_list, parameters)
    else:
        # Server idle
        parameters["NumberInService"][evt.cust.server] = 0

    response = parameters["Clock"] - finished.getTime()
    parameters["SumResponseTime"] += response


    parameters["TotalBusy"] += parameters["Clock"] - parameters["LastEventTime"]
    parameters["NumberOfDepartures"] += 1
    parameters["LastEventTime"] = parameters["Clock"]


def ReportGeneration(finishedEvents, parameters):
    global data
    maxQlen = max(parameters["MaxQueueLength"])
    averageQlen = sum(parameters["MaxQueueLength"])/len(parameters["MaxQueueLength"])

    arrivals = []
    departures = []

    for e in finishedEvents:
        if e.etype == "arrival":
            arrivals.append(e.time)
        else:
            departures.append(e.time)

    TCSS = []
    for i in range(0,len(arrivals)):
        TCSS.append(departures[i]-arrivals[i])

    averageTCSS = sum(TCSS)/len(TCSS)
    
    print(f"Max Queue Length {maxQlen}")
    print(f"Average Queue Length {averageQlen}")
    print(f"Average TCSS {averageTCSS}")
    data.append(str(maxQlen))
    data.append(str(averageQlen))
    data.append(str(averageTCSS))


def print_simulation_table(all_events, parameters):
    evt_by_cust = {x: [] for x in range(parameters['total_customers'])}
    for e in all_events:
        evt_by_cust[e.cust.id].append(e)

    assert all(
        len(v) == 2 for v in evt_by_cust.values()
    ), "disparity between arrivals and departures"  # DEBUG


    pt = PrettyTable()
    pt.field_names = [
        "Customer",
        "Server",
        "Arrival Time",
        "Departure Time",
        "Service Time for the Customer",
        "The Customer's Waiting Time in the Queue",
    ]
    for c in list(evt_by_cust.values()):
        pt.add_row(
            [
                c[0].cust.id,
                c[0].cust.server,
                round(c[0].time, 2),
                round(c[1].time, 2),
                round(c[1].service_duration, 2),
                round(c[1].service_start_time - c[0].time, 2),
            ]
        )
    print(pt)


def print_event_list(all_events, parameters):
    pt = PrettyTable()
    pt.field_names = ["Event Type", "Server", "Customer Number", "Clock Time"]
    for e in all_events:
        pt.add_row([e.etype, e.cust.server, e.cust.id, round(e.time, 2)])
    print(pt)

def RandMin(params, customer_queue):
    # Choose rand_min_d server indices
    server_idxs = params["rng"].choice(
        list(range(params["num_servers"])), size=params["rand_min_d"], replace=False
    )
    
    # Create list of lengths of chosen server queues
    server_queue_lengths = [len(customer_queue[i]) for i in server_idxs]
    # Get index of the shortest queue length
    min_len_idx = np.argmin(server_queue_lengths)
    # return the true index of the server
    return server_idxs[min_len_idx]


def PureRand(params):
    return params["rng"].integers(0, params['num_servers'])


def RR(params):
    params["PreviousServer"] += 1
    return params.get("PreviousServer") % params['num_servers']




if __name__ == "__main__":
    num_servers = 6
    rand_min_d = [5,3,1]  # TODO: should be between 1 and num_servers
    MeanInterArrivalTime = 1/6  # Mean inter arrival time = 1/λ 
    total_customers = 10000 # Total customers
    mu = 1 # Service time

    seeds = np.arange(5)  # seed for random generation
    rngs = [default_rng(s) for s in seeds]
    load_balancers = [RandMin, PureRand, RR]
    ss = 0 #keeps track of seed
    print("Mu: " + str(mu))
    print("Lambda: " + str(MeanInterArrivalTime))
    print("Number of Servers: " + str(num_servers))
    print()
    setno = 0 #for keeping track of which set A, B or C


    header = ['Seed', 'Load Balancer',"Avg Q Len", 'Max Q Len', 'Avg TCSS']
    data = []
    outputfilename = 'c'

    for d in rand_min_d:
        print()
        letter = setno%3
        if letter == 0:
            print("SET C")
            print("------------------------------------")
        if letter == 1:
            print("SET B")
            print("------------------------------------")
            outputfilename = 'b'
        if letter == 2:
            print("SET A")
            print("------------------------------------")
            outputfilename = 'a'
        setno += 1
        with open(outputfilename + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for rng in rngs:
                print()
                print("Seed: " + str(ss % 5))
                
                ss += 1 
                for lb in load_balancers:
                    # Reset customer and event id counters
                    Customer.generate_id = itertools.count()
                    Event.generate_id = itertools.count()
                    data.append(str((ss -1) % 5))
                    print()
                    print(lb.__name__)
                    data.append(lb.__name__)
                    if lb is RandMin:
                        Sim(total_customers, num_servers, rng, MeanInterArrivalTime, mu, lb, d, True)
                        print("d: " + str(d))
                    else:
                        Sim(total_customers, num_servers, rng, MeanInterArrivalTime, mu, lb, None, True)
                    writer.writerow(data)
                    data = []
