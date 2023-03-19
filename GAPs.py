# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:46:30 2022

@author: Aaron Ogle

NOTE - The programs GAPsII, III, and IV are all identical to this program
except that they utilize a different number of nodes.  This program has more
comments in it than the others as the same logic will apply in all of the 
others.

NOTE - This is the only program that will calculate energy consumption; we
decided to focus just on this network model, instead of adding the same
code into each version as this program should give us an idea of how well
the energy is distributed.

NOTE - If the number of nodes are increased in the network then 
modifications need to be made throughout the program in order to
accommodate the change.  

Description:  This program will implement a geometric and probabilistic
based routing scheme for source location privacy in WSN.

The algorithm will do the following:
    
    1) Divide the sensor nodes into tiers by utilizing the convex hull
        a) The convex hull of the sensor nodes will be computed; the
           nodes that are contained in the convex hull will be removed
           from the set of nodes and the convex hull will be computed
           on the remaining nodes.  This process will be done until
           there is only one node left, the sink node.
        b) To compute this, we take advantage of the fact that our 
           nodes range in value from [0,0] to [12,12].  We note that
           the vertices which are contained in the convex hull  
           will first contain x and y values of 0 and 12; the next iteration
           will contain x and y values of 1, 11, then the next will contain
           x and y values of 2 and 10; and so on and so forth.
           Because of this, we can create a "points permutation" list
           which essentially sorts our sensor nodes based off of the
           order of which they lie in the computed convex hull.
        c) Once we have this list, we can break the list down into a
           a matrix of different tier groups.
    2) Once we have a tier representation matrix of the different layers
       of computed convex hulls of our points, our algorithm will proceed 
       as follows:
           1) Randomly select a tier group
           2) Randomly select a point that lies in this tier group
           3) Forward the packet to this path node utilizing the shortest path
           4) From this path node, once the packet is obtained, forward the
              packet to the sink node by the shortest path

Note that throughout the comments of this code the terms "sensors" 
and "points" will be used interchangeably since the points will
represent the location of our sensors in the 2-dimensional plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import networkx as nx
from networkx import grid_graph
import random

"""
Define a global variable to be our sensors.  The sensors are represented
by points in the 2D plane.  For this program, our total number of sensors
are 169, and their values are between ([0,0], [0,1],...,[12,12]).
"""
SENSORS = [[x,y] for x in range(0,13) for y in range(0, 13)]

"""
The generate_tier_matrix will accept our sensors (points) as input, and 
will partition the sensors into different tiers.

The function first contructs a list called points_permutation which
will keep track of the order of the points that lie in the different
convex hulls of our set of points.

The list num_of_sensors will keep track of how many sensors are being
removed at each stage.  We need this number in order to know how many
points lie in each of the different tiers.

We also keep track of the number of iterations through a variable called
count.  This count is utilized so that we can know how many tiers there 
will be.

While the number of sensors in our list is greater than one, we will 
compute the convex hull of the set of points.  If the values of the
points are the same as the values of the points in the vertices of the
convex hull, then we want to put these points in our points permutation
list, as these are the points that are contained within the convex hull.
We then want to remove these points from our initial list so that
we can compute the convex hull for the next level of points.  We do this
until there is only one point left in our set of points which will be
the point (6,6), our sink node.

We then make sure that the points in our permutation list are unique, and 
we construct a matrix of different tiers that is based off of the count
of iterations in our while loop.  We kept track of the number of sensors
being removed through each iteration, we then use this information
in order to know how many sensors need to go into each tier of the matrix

Once we have gone through this process we return our matrix that is
composed of different tier levels based off of computing the convex hull
of different sets of points described above.

"""

def generate_tier_matrix(sensors):
    # Initialize lists/count described above
    points_permutation = []
    num_of_sensors = []
    count = 0
    """
    While there are nodes other than the sink node in our sensors set of points:
        keep track of the number of nodes in our sensors list
        increment the count to keep track of number of tiers
        compute the convex hull of the set of points
        compute the vertices that form the convex hull
        check each of the sensors to see if one of their points
        equals the convex hulls vertices point, if it does
        then we know this point is in the convex hull and we add it
        to our list, then remove the point from our original sensor list
    """
    while len(sensors) > 1:
        num_of_sensors.append(len(sensors))
        count += 1
        sensor_array = np.array(sensors)
        hull = ConvexHull(sensors)
        ch_x_vertex = (sensor_array[hull.vertices][0][0])
        ch_x2_vertex = (sensor_array[hull.vertices][1][0])
        
        for sensor in sensors:
            if ch_x_vertex in sensor:
                points_permutation.append(sensor)
            if ch_x2_vertex in sensor:
                points_permutation.append(sensor)
        for sensor in points_permutation:
            if sensor in sensors:
                sensors.remove(sensor)  

    # Clean the list above in case a point was duplicated in our process
    distinct_permutated_pts = []
    for x in points_permutation:
        if x not in distinct_permutated_pts:
            distinct_permutated_pts.append(x)
            
    # Create a matrix based off of the number of tiers found above
    matrix = [[] for i in range(count)]
    
    # Use the number of sensors list defined above to compute the
    # number of nodes that belong to each tier
    t1 = num_of_sensors[0] - num_of_sensors[1]
    t2 = t1 + num_of_sensors[1] - num_of_sensors[2]
    t3 = t2 + num_of_sensors[2] - num_of_sensors[3]
    t4 = t3 + num_of_sensors[3] - num_of_sensors[4]
    t5 = t4 + num_of_sensors[4] - num_of_sensors[5]
    t6 = t5 + num_of_sensors[5] - 1
    
    # Add the correct nodes to the correct tier level in our matrix
    for i in range(t1):
        matrix[0].append(distinct_permutated_pts[i])
    for i in range(t1,t2):
        matrix[1].append(distinct_permutated_pts[i])
    for i in range(t2, t3):
        matrix[2].append(distinct_permutated_pts[i])
    for i in range(t3, t4):
        matrix[3].append(distinct_permutated_pts[i])
    for i in range(t4, t5):
        matrix[4].append(distinct_permutated_pts[i])
    for i in range(t5, t6):
        matrix[5].append(distinct_permutated_pts[i])
    

    return matrix

"""
neighbor_ nodes Function

The neighbor_nodes function will accept a node as input and it will then compute
all of the nodes that are its "neighbor nodes".  By neighbor nodes, we mean the
nodes that are within distance one of the given node.  For example, if the node provided
is (6,6) then the node that is directly above it is the point (6,7), for any
given node there will be exactly 8 neighbor nodes.  The function computes all of the
nodes neighbor nodes and then puts them into a list data structure and returns the
list.
"""

def neighbor_nodes(node):
    up = node[0], node[1] + 1
    down = node[0], node[1] - 1
    left = node[0] - 1, node[1]
    right = node[0] + 1, node[1]
    up_right = node[0] + 1, node[1] + 1
    up_left = node[0] - 1, node[1] + 1
    down_right = node[0] + 1, node[1] - 1
    down_left = node[0] - 1, node[0] - 1
    neighbor_nodes = [up, down, left, right, up_right, up_left, down_right, down_left]
    return neighbor_nodes

"""
adversary Function

The adversary function is a function that will represent our adversary in the
WSN.  The function takes as input the adversaries location, the adversaries
route list, and the path that is being taken by the packet in the network
when it is relayed from the source node to the sink node.

The function calls the neighbor nodes function defined above so that the
adversary "knows" which nodes are currently around it.  The adversary
then checks to see if any nodes in their neighbor nodes list is equal
to one of the nodes in the packets routed path and not in its route list,
if it is the adversary moves to that node.  The adversary keeps track of
all of the nodes that it visits in its "adv_route" list so that it does
not revisit nodes that it has already visited.
"""
    

def adversary(adversary, adv_route, path):

    nn = neighbor_nodes(adversary)

    if nn[0] in path and nn[0] not in adv_route:
        adversary = nn[0]
        adv_route.append(adversary)
    elif nn[1] in path and nn[1] not in adv_route:
        adversary = nn[1]
        adv_route.append(adversary)
    elif nn[2] in path and nn[2] not in adv_route:
        adversary = nn[2]
        adv_route.append(adversary)
    elif nn[3] in path and nn[3] not in adv_route:
        adversary = nn[3]
        adv_route.append(adversary)
    elif nn[4] in path and nn[4] not in adv_route:
        adversary = nn[4]
        adv_route.append(adversary)
    elif nn[5] in path and nn[5] not in adv_route:
        adversary = nn[5]
        adv_route.append(adversary)
    elif nn[6] in path and nn[6] not in adv_route:
        adversary = nn[6]
        adv_route.append(adversary)
    elif nn[7] in path and nn[7] not in adv_route:
        adversary = nn[7]
        adv_route.append(adversary)
    
    return adversary, adv_route

"""
The GAPs function will implement the proposed GAPs algorithm.

The algorithm first utilizes the networkx library to create a 
graph of the same shape as the one defined earlier in our program.

The algorithm then defines the sink node locatoin and the adversaries
initial start location along with an empty list for the adversary
to keep track of the nodes they visit

We create a count variable so that we can keep track of the number
of packets that are successfully sent before the adversary reaches
the location of the source node.

The algorithm then selects a random tier in the provided matrix,
and then selects a random point in that tier; we then define
this selected node as the path node and create a tuple data type of it.

With the path node with then utilize the networkx shortest path 
function so that the algorithm can find the shortest path from
the source node to that path node, then from the path node to the sink
node.  We then define a total path variable that contains this data.

Once the total path is determined, we call the adversary function and
send the path to that function.

We then create a while loop that repeats this process until the 
adversary is able to locate the location of the source node.  We create
a break function so that if the count reaches 1000 we do not keep
going through the while loop.  The loop also breaks when the
adversary reaches the source node.  The function then returns the
count which represents the total number of packets that are sent
before the adversary reaches the source node.

########################################################################

Update

The algorithm has been updated so that we can compute values for 
energy consumption
########################################################################
"""
                    
def GAPs(matrix, source_node):
    G = nx.generators.lattice.grid_2d_graph(13,13)
    # The below values up to tot_energy are needed to
    # compute energy consumption
    nodes = list(G.nodes)
    # Start each node with 0.5 joules of energy
    energy_dict = {nodes[i]: 0.5 for i in range(0, len(nodes))}
    # Based off of the works of others...and assuming that each
    # node is 20 meters away from a neighbor node
    # The energy in joules for receiving and transmitting a packet
    # are the below values
    e_rec = 0.0000512
    e_tran = 0.000055296
    tot_energy = e_rec + e_tran
    # Set sink node and adversary location
    sink_node = (0,0)
    adver_start = (0,0)
    adv_rte = []
    adv_rte.append(adver_start)
    count = 0
    # Randomly select tier and point in tier, then use that for the path node
    random_tier = random.randint(0,len(matrix)-1)
    random_point = random.randint(0,len(matrix[random_tier])-1)
    path_node = tuple(matrix[random_tier][random_point])
    p2pn = (nx.shortest_path(G, source = source_node , target = path_node ))
    p2sn = (nx.shortest_path(G, source = path_node, target = sink_node))
    # Add paths to get total path and pass this information to the adversary
    total_path = p2pn + p2sn
    adv_update, adv_r = adversary(adver_start, adv_rte, total_path)
    
    # Calculate energy consumed by nodes in initial path
    # Iterate through each node and see if it is in the path
    # that was generated by the algorithm, if it was then
    # subtract energy from it
    for key in energy_dict:
        for i in total_path:
            if key == i:
                energy_dict[key] -= tot_energy
    while (True):
        count += 1
        if count == 4000:
            break
        
        # Similar logic as what is in the above
        random_tier = random.randint(0,len(matrix)-1)
        random_point = random.randint(0,len(matrix[random_tier])-1)
        path_node = tuple(matrix[random_tier][random_point])
        p2pn = (nx.shortest_path(G, source = source_node , target = path_node ))
        p2sn = (nx.shortest_path(G, source = path_node, target = sink_node))
        total_path = p2pn + p2sn
        adv_update, adv_r = adversary(adv_update, adv_rte, total_path)
        
        # Calculate energy consumed while adversary is looking for source node
        for key in energy_dict:
            for i in total_path:
                if key == i:
                    energy_dict[key] -= tot_energy
        
        # Comment in and out for energy consumption measure
        """
        if source_node in adv_rte:
            break
        """
    return count, energy_dict

"""
shortest_path function

The shortest path function will utilize networkx's shortest path
function so that we can compare the GAPs algorithm to the shortest
path algorithm.

Similar to the GAPs function defined above the shortest path function
creates a networkx graph that is the same structure as the graph
defined previously in our function.  We then define the location of the
sink node and adversary in the network and create an empty adversary
route list.  

The function computes the shortest path between the given source node
and the sink node and then sends that information to the adversary
function.  This process is repeated until the adversary reaches
the source node or when the total number of packets sent equals
1000.  The function then returns the length of the adversaries
route as this will be the same value as the total number of packets
that are sent before the adversary reaches the source node.

########################################################################

Update

The algorithm has been updated so that we can compute values for 
energy consumption
########################################################################
"""
    
def shortest_path(source_node):
    G2 = nx.generators.lattice.grid_2d_graph(13,13)
    # The below values up to tot_energy are needed to
    # compute energy consumption
    nodes = list(G2.nodes)
    energy_dict = {nodes[i]: 0.5 for i in range(0, len(nodes))}
    e_rec = 0.0000512
    e_tran = 0.000055296
    tot_energy = e_rec + e_tran
    # Set sink node and adversary location
    sink_node = (0,0)
    adver_start = (0,0)
    adv_rte = []
    adv_rte.append(adver_start)
    count = 0
    path = (nx.shortest_path(G2, source = source_node, target = sink_node))
    adv_update, adv_rte = adversary(adver_start, adv_rte, path)
    
    # Calculate energy consumed by nodes in initial path
    for key in energy_dict:
        for i in path:
            if key == i:
                energy_dict[key] -= tot_energy

    while (True):
        count += 1
        if count == 4000:
            break
        path = (nx.shortest_path(G2, source = source_node, target = sink_node))
        adv_update, adv_rte = adversary(adv_update, adv_rte, path)
        
        # Calculate energy consumed by nodes 
        for key in energy_dict:
            for i in path:
                if key == i:
                    energy_dict[key] -= tot_energy
        
        # Comment in and out for energy consumption measure
        """
        if source_node in adv_rte:
            break
        """
    return count, energy_dict
        
    
def main():
    """
    Generate sensors (points) in the 2D plane
    
    These points will be between:
    ([0,0], [0,1], ...,[12,12])
    
    For a total of 169 sensors (points)
    
    Note that these points are the same as our sensor points
    defined above in our global var, these points will be
    changed through the matrix tier generator process.
    """
    points = [[x,y] for x in range(0,13) for y in range(0, 13)]
    
    
    # Generate a matrix of different tiers based off of the
    # computation of the convex hull
    # Algorithm defined in detail above
    matrix = (generate_tier_matrix(points))
    
    
    # The below is utilized for plotting
    
    """

    # T6 - Tier 6 - will be the first row of the matrix
    # T5 - Tier 5 - will be the second row of the matrix..etc

    t6 = matrix[0]
    t5 = matrix[1]
    t4 = matrix[2]
    t3 = matrix[3]
    t2 = matrix[4]
    t1 = matrix[5]
    
    # The sink node is the last remaining sensor/point in the points list
    # as it was not removed from the list during the generate tier matrix
    # process
    
    
    # Get the values for x and y in the above list of the global var SENSORS
    # Note that in the above, we changed the points list through
    # the generate tier matrix process
    x = [x for x, y in SENSORS]
    y = [y for x, y in SENSORS]
    # Plot our points (sensors)
    plt.scatter(x,y)
    plt.title(" 169 Network Sensors")
    
    # Uncomment the below to visualize the convex hulls
    
    # Compute and plot the convex hull for each of the tiers computed above
    tier_6_array = np.array(t6)
    hull_tier_6 = ConvexHull(tier_6_array)
    
    plt.plot(tier_6_array[:,0], tier_6_array[:,1], 'o')
    for s in hull_tier_6.simplices:
        plt.plot(tier_6_array[s, 0], tier_6_array[s, 1], 'k-')
    
    tier_5_array = np.array(t5)
    hull_tier_5 = ConvexHull(tier_5_array)
    
    plt.plot(tier_5_array[:,0], tier_5_array[:,1], 'o')
    for s in hull_tier_5.simplices:
        plt.plot(tier_5_array[s, 0], tier_5_array[s, 1], 'k-')
      
    tier_4_array = np.array(t4)
    hull_tier_4 = ConvexHull(tier_4_array)
    
    plt.plot(tier_4_array[:,0], tier_4_array[:,1], 'o')
    for s in hull_tier_4.simplices:
        plt.plot(tier_4_array[s, 0], tier_4_array[s, 1], 'k-')
        
    tier_3_array = np.array(t3)
    hull_tier_3 = ConvexHull(tier_3_array)
    
    plt.plot(tier_3_array[:,0], tier_3_array[:,1], 'o')
    for s in hull_tier_3.simplices:
        plt.plot(tier_3_array[s, 0], tier_3_array[s, 1], 'k-')
        
    tier_2_array = np.array(t2)
    hull_tier_2 = ConvexHull(tier_2_array)
    
    plt.plot(tier_2_array[:,0], tier_2_array[:,1], 'o')
    for s in hull_tier_2.simplices:
        plt.plot(tier_2_array[s, 0], tier_2_array[s, 1], 'k-')
        
    tier_1_array = np.array(t1)
    hull_tier_1 = ConvexHull(tier_1_array)
    
    plt.plot(tier_1_array[:,0], tier_1_array[:,1], 'o')
    for s in hull_tier_1.simplices:
        plt.plot(tier_1_array[s, 0], tier_1_array[s, 1], 'k-')


    """
    
    #############################################################################################
    #
    #
    # MAIN TEST SECTION
    #
    #
    # TEST 1
    #
    #
    #############################################################################################
    """
    Find average number of packets sent for 100 simulations
    
    NOTE - if the below needs to be ran then modifications need to be made
    in the GAPs and shortest_path functions so that the adversaries location
    is being checked...this section is commented out in order to compute
    energy consumption
    """
    """
    gaps_count = 0
    short_count = 0
    for i in range(100):
        #s_node = (random.randint(0, 12), random.randint(0,12))
        s_node = (12,12)
        gaps_count += GAPs(matrix, s_node)[0]
        short_count += shortest_path(s_node)[0]
    print(gaps_count / 100, short_count / 100, "\n")
    """
    
    """
    Calculate Energy Consumption
    """
    #s_node = (random.randint(0, 12), random.randint(0,12))
    s_node = (12,12)
    gaps_energy = GAPs(matrix, s_node)[1]
    sp_energy = shortest_path(s_node)[1]
    
    """
    Plot energy consumption as a bar chart to show distribution
    """
    s = [i+1 for i in range(169)]
    #plt.bar(s, list(gaps_energy.values()))
    plt.bar(s, list(sp_energy.values()))
    plt.xlabel("Nodes")
    plt.ylabel("Energy Consumed (Joules)")
    plt.title("SP - Energy Consumption Distribution")
    plt.show()
    
    
    
    """
    # Plot algorithms by avg num of hops
    
    # Average number of hops
    GAPs_Avg_Hop = [22, 25, 29, 31]
    SP_Avg_Hop = [11, 13, 15, 17]
    nodes = [169, 225, 289, 361]
    
    plt.plot(nodes, GAPs_Avg_Hop, 'red', label="GAPs")
    plt.plot(nodes, SP_Avg_Hop, 'green', label="SP")
    plt.xlabel("Nodes")
    plt.ylabel("Avg Number of Hops")
    plt.title("Avg Number of Hops by Number of Nodes")
    plt.legend(loc='upper left')
    plt.show()
    """
    
    #############################################################################################
    #
    #
    # Experimental Results Test 1 - source node is randomly generated with sink node in
    # the middle of the network
    #
    #
    #############################################################################################
    """
        # Average number of pkts sent
    GAPs_Avg_Pkt = [236, 327, 370, 385]
    SP_Avg_Pkt = [7, 8, 9, 11]
    nodes = [169, 225, 289, 361]
    
    plt.plot(nodes, GAPs_Avg_Pkt, 'red', label="GAPs")
    plt.plot(nodes, SP_Avg_Pkt, 'green', label="SP")
    plt.xlabel("Nodes")
    plt.ylabel("Avg Number of Packets Sent")
    plt.title("Avg Number of Packets by Number of Nodes")
    plt.legend(loc='upper left')
    plt.show()
    """
    #############################################################################################
    #
    #
    # MAIN TEST SECTION
    #
    #
    # TEST 2
    #
    #
    #############################################################################################
    """
    gaps_count = 0
    short_count = 0
    for i in range(100):
        s_node = (random.randint(0, 12), random.randint(0,12))
        gaps_count += GAPs(matrix, s_node)
        short_count += shortest_path(s_node)
    print(gaps_count / 100, short_count / 100)
    
    #############################################################################################
    #
    #
    # Experimental Results Test 2 - source node is randomly generated with sink node in
    # the bottem left corner of the network
    #
    #
    #############################################################################################
    """
    """
    # Average number of pkts sent
    GAPs_Avg_Pkt = [238,322,325,468]
    SP_Avg_Pkt = [13,15,17,29]
    nodes = [169, 225, 289, 361]
    
    plt.plot(nodes, GAPs_Avg_Pkt, 'red', label="GAPs")
    plt.plot(nodes, SP_Avg_Pkt, 'green', label="SP")
    plt.xlabel("Nodes")
    plt.ylabel("Avg Number of Packets Sent")
    plt.title("Avg Number of Packets by Number of Nodes")
    plt.legend(loc='upper left')
    plt.show()
    """
    
    #############################################################################################
    #
    #
    # MAIN TEST SECTION
    #
    #
    # TEST 3
    #
    #
    #############################################################################################
    """
    gaps_count = 0
    short_count = 0
    for i in range(100):
        #s_node = (random.randint(0, 12), random.randint(0,12))
        s_node = (12,12)
        gaps_count += GAPs(matrix, s_node)
        short_count += shortest_path(s_node)
    print(gaps_count / 100, short_count / 100)
    
    #############################################################################################
    #
    #
    # Experimental Results Test 3 - source node is no longer random and is
    # instead placed at the top right corner of the network with sink node in
    # the bottem left corner of the network
    #
    #
    #############################################################################################
    
    
    # Average number of pkts sent
    GAPs_Avg_Pkt = [610,759,822,861]
    SP_Avg_Pkt = [25,29,33,37]
    nodes = [169, 225, 289, 361]
    
    plt.plot(nodes, GAPs_Avg_Pkt, 'red', label="GAPs")
    plt.plot(nodes, SP_Avg_Pkt, 'green', label="SP")
    plt.xlabel("Nodes")
    plt.ylabel("Avg Number of Packets Sent")
    plt.title("Avg Number of Packets by Number of Nodes")
    plt.legend(loc='upper left')
    plt.show()
    
    """
    
main()