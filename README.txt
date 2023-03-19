Project: GAPs - A Geometric and Probabilistic based routing scheme for SLP 
(source location privacy) in WSN

Author: Aaron Ogle

Description:

The first section of the program imports the following libraries:

numpy
matplotlib.pyplot
sci.spatial
networkx
random

Some of the source code that utilizes these libraries has been commented out due to only needing to be used
for certain portions of the program.

The first function that is defined in the program is the generate_tier_matrix function that accepts
the sensors as input.  This function will construct our partitioned network that is implemented as a matrix
or list of lists in python.

The next function that is implemented is the neighbor_nodes function; this function computes the 8 neighbor nodes
of a given input node.

The next function is the adversary function.  This function first calculates its neighbor nodes and then checks
to see if any of its neighbor nodes is in the input path and has not already been visited; if the node
has not been visited yet then the adversary moves to that node.

The next function is the actual implemenation of the GAPs algorithm.

Modifications have been made to it so that we can calculate the energy that is consumed in the network.  The 
function is currently set to calculate energy consumption, and modificiations have to be made to the function 
in order to have it used for the source location privacy portion of the code.

The next function is the shortest path function.  This function is currently set to compute energy
consumption as well, and will need to be updated if we want to investigate the source location
privacy portion of the code as well.

The final portion of the source code is our driver/main function that is utilized in order to actually
implement all of the other defined functions.  There are several commented out sections of the main
function due to it being used for different reasons; i.e. if we want to plot the network partition, vs,
if we want to run different tests for the safety period of the algorithms.

The program is currently set to compute energy consumption, but again can be modified in order to compute
the safety period/investigate the SLP protection of the network.

INSTALLING and RUNNING the program

As long as the libraries that were mentioned above:

numpy
matplotlib.pyplot
scipy.spatial
networkx
random

are all imported then I believe that these .py files should run on any python IDE.  


