# bistable-spring-net
 
 This repository contains code associated with the paper ["Drive-specific adaptation in disordered mechanical networks of bistable springs"](https://arxiv.org/abs/1908.09332). 
 
 Essentially the question being explored is summarized in this slide:
 ![](https://github.com/hrideshkedia/bistable-spring-net/blob/main/figs/intro.jpeg)
 
 
#### What we find, surprisingly, is:


![](https://github.com/hrideshkedia/bistable-spring-net/blob/main/figs/punch.jpeg)
 
 For the details of the story, please see the paper ["Drive-specific adaptation in disordered mechanical networks of bistable springs"](https://arxiv.org/abs/1908.09332).
 
 The simulation engine and calculation of the linearized dynamical matrix and its eigenvalues is written in Cython in the bistable_spring_net_fns.pyx file.
 To analyze the results of a large batch of simulations and generate aggregated results, appropriately modify and run the bistable_spring_net_data_analyse.py file. 
 To initialize and set up a batch of simulations appropriately modify and run the bistable_spring_net.py file.
 
The code constructs a random spring network, assumes identical bistable springs between all pairs of nodes connected by an edge, and simulates the dynamics of the network when the most connected node is driven by a periodic force. It also allows for a time-varying forcing direction, switching between different forcing frequencies, and for subjecting the most connected node to a periodic displacement, or switching between periodic forcing and periodic displacement. 

To ensure that the network remains confined and to remove the two global translations and a global rotation that don't cost energy, and to make the network dynamics more constrained, we anchor 3 nodes that are atleast two edges away from the most connected node.

Each simulation creates a new random network, and an associated folder, in which the simulation data including a summary plot of the relevant physical quantities including Kinetic Energy, Dissipation rate, Work absorption rate, etc is stored. 

There are also functions to create a video of the simulation from the stored simulation data, which require PIL, io and cv2 packages. This repo requires python 2.7 and cython. 

There are comments in each file intended to clarify the code. The code documentation will be continually improved with time. 
