# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
import numpy as np
cimport numpy as np
import cmath
import os
#import math
import cython
cimport cython
from cython.view cimport array as cvarray
from libcpp cimport bool
from scipy import signal
from scipy import linalg
from libc.math cimport sin, cos, sqrt, fabs, round, exp
from libc.stdlib cimport rand, RAND_MAX

DTYPE = np.float64
ITYPE = np.int
CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t
ctypedef np.int_t ITYPE_t

cpdef initialize_velocityvec(dict params_dict):
    #Bolztmann distribution
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef ITYPE_t NODES = params_dict['NODES']
    cdef DTYPE_t beta = params_dict['BETA']
    return np.random.multivariate_normal( np.zeros(NODES), (1./(beta * MASS))*np.identity(NODES))

cpdef init_positions(dict params_dict):
    #random postion on a circle of radius L1
    cdef DTYPE_t beta = params_dict['BETA']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef ITYPE_t[:,:] Amat = params_dict['AMAT']
    cdef ITYPE_t EDGES = params_dict['EDGES']
    cdef ITYPE_t NODES = params_dict['NODES']
    cdef np.ndarray[DTYPE_t, ndim=1] init_xlist = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] init_ylist = np.empty(NODES, dtype=DTYPE)
    cdef DTYPE_t r_init_circle = 1.5*params_dict['L1']
    cdef ITYPE_t node
    #random angle along the circle
    cdef np.ndarray[DTYPE_t, ndim=1] phi_angles = 2*np.pi*np.random.random_sample((NODES,))
    for node in range(NODES):
        init_xlist[node] = r_init_circle*cos(phi_angles[node])
        init_ylist[node] = r_init_circle*sin(phi_angles[node])
    init_xlist[max_deg_vertex_i] = 0.
    init_ylist[max_deg_vertex_i] = 0.

    return [init_xlist,init_ylist]

cdef K_eff(dict params_dict, DTYPE_t dij):
    cdef DTYPE_t SPRING_K = params_dict['SPRING_K']
    if dij < 4./5:
        return SPRING_K*25.
    elif dij < 6./5:
        return SPRING_K*0.5
    elif dij < 14./5:
        return SPRING_K*(1.77002*(12*dij*dij -2*24.6703*dij + 47.733))
    elif dij < 16./5:
        return SPRING_K*5.
    else:
        return SPRING_K*20.

cdef get_scaled_data(DTYPE_t[:] data_array, DTYPE_t scale_factor):
# assumes length of avg_array is ceiling of length of data_array divided by block_size
    cdef ITYPE_t data_size = data_array.shape[0]
    cdef ITYPE_t i
    for i in range(data_size):
        data_array[i] *= scale_factor

cdef add_arrays(DTYPE_t[:] array1, DTYPE_t[:] array2, DTYPE_t[:] sum_array):
    cdef ITYPE_t arr_size = array1.shape[0]
    cdef ITYPE_t i
    for i in range(arr_size):
        sum_array[i] = array1[i] + array2[i]

cdef get_scaled_block_sum(DTYPE_t[:] data_array, DTYPE_t[:] avg_array, ITYPE_t block_size, DTYPE_t scale_factor):
# assumes length of avg_array is ceiling of length of data_array divided by block_size
    cdef ITYPE_t data_size = data_array.shape[0]
    cdef ITYPE_t avg_size = avg_array.shape[0]
    cdef ITYPE_t i
    cdef ITYPE_t n
    cdef DTYPE_t running_sum
    for n in range(avg_size-1):
        running_sum = 0
        for i in range(block_size):
            running_sum += data_array[block_size*n + i]
        avg_array[n] = (running_sum*scale_factor)
    n +=1
    i=0
    running_sum = 0
    while (n*block_size + i) < data_size:
        running_sum += data_array[(n*block_size + i)]
        i+=1
    avg_array[n] = (running_sum*scale_factor)

cpdef get_spring_freezing_times(ITYPE_t[:,:] spring_bit_flips_tsteps, DTYPE_t[:] tlist,
                                        DTYPE_t[:] spring_freezing_times):
    cdef ITYPE_t spring_index, tindex, freezing_index, edges, tlength
    cdef DTYPE_t freezing_time

    tlength = spring_bit_flips_tsteps.shape[0]
    edges = spring_bit_flips_tsteps.shape[1]

    for spring_index in range(edges):
        if spring_bit_flips_tsteps[tlength-1][spring_index] !=0:
            spring_freezing_times[spring_index] = tlist[tlength-1]
        else:
            tindex = tlength - 2
            while spring_bit_flips_tsteps[tindex][spring_index]==0 and tindex>0:
                tindex -= 1
            spring_freezing_times[spring_index] = tlist[tindex+1]

cpdef get_num_frozen_springs(DTYPE_t[:] spring_freezing_times, DTYPE_t[:] tlist,
                                ITYPE_t[:] num_frozen_springs_tsteps):
    cdef ITYPE_t spring_index, tindex, edges, tlength, num_frozen_springs
    cdef DTYPE_t freezing_time

    tlength = tlist.shape[0]
    edges = spring_freezing_times.shape[0]

    for tindex in range(tlength):
        num_frozen_springs = 0
        for spring_index in range(edges):
            if tlist[tindex] >= spring_freezing_times[spring_index]:
                num_frozen_springs += 1
        num_frozen_springs_tsteps[tindex] = num_frozen_springs


cpdef get_scaled_block_avg(DTYPE_t[:] data_array, DTYPE_t[:] avg_array, ITYPE_t block_size, DTYPE_t scale_factor):
# assumes length of avg_array is ceiling of length of data_array divided by block_size
    cdef ITYPE_t data_size = data_array.shape[0]
    cdef ITYPE_t avg_size = avg_array.shape[0]
    cdef ITYPE_t i
    cdef ITYPE_t n
    cdef DTYPE_t running_sum
    for n in range(avg_size-1):
        running_sum = 0
        for i in range(block_size):
            running_sum += data_array[block_size*n + i]
        avg_array[n] = (running_sum*scale_factor)/(block_size)
    n +=1
    i=0
    running_sum = 0
    while (n*block_size + i) < data_size:
        running_sum += data_array[(n*block_size + i)]
        i+=1
    avg_array[n] = (running_sum*scale_factor)/i


cpdef get_plot_data_dict(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                        DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] tlist, DTYPE_t[:] f_ext_tlist,
                        DTYPE_t[:] work_rate_tsteps, DTYPE_t[:] diss_rate_tsteps, dict params_dict, DTYPE_t tol=1e-8):
    # get column 1: U_total, (KE_total, E_total), netEminusWrate; column2: state_bit_number(graycode), (num_ones, num_flips), KE_com / KE_total; column3: power, diss_rate, f_ext(t)
    # , DTYPE_t[:] phi_tlist
    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef ITYPE_t EDGES = params_dict['EDGES']
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t NODES = params_dict['NODES']
    cdef ITYPE_t num_dwell_times, num_flip_times
    cdef DTYPE_t DT = params_dict['DT']
    cdef ITYPE_t RELAX_STEPS = params_dict['RELAX_STEPS']
    cdef DTYPE_t FORCE_PERIOD = params_dict['FORCE_PERIOD']
    cdef DTYPE_t L1 = params_dict['L1']
    cdef DTYPE_t Lbarrier = params_dict['Lbarrier']
    cdef DTYPE_t barrier_height = U_spring(Lbarrier, params_dict)# should be calculated from zero, to edit the potential offset
    #dimensionless factors
    cdef DTYPE_t energy_scale_factor = 1./(barrier_height)
    cdef DTYPE_t power_scale_factor = FORCE_PERIOD*energy_scale_factor
    cdef DTYPE_t time_scale_factor = 1./FORCE_PERIOD
    cdef DTYPE_t force_scale_factor = (Lbarrier-L1)*energy_scale_factor
    cdef np.ndarray[DTYPE_t, ndim=2]forcing_direction = np.zeros([2*NODES,xilists_tsteps.shape[0]])
    cdef ITYPE_t i

    for i in range(2*params_dict['NODES']):
        forcing_direction[i,:] = params_dict['FORCING_VEC'][i]

    #num of steps in one force period
    cdef ITYPE_t block_size = np.int(round(FORCE_PERIOD/DT))
    #plotting points time steps
    cdef np.ndarray[ITYPE_t, ndim=1] block_indices_tlist = np.arange(0,tlength,block_size)
    #plotting time
    cdef np.ndarray[DTYPE_t, ndim=1] tlist_block_wise = np.take(tlist, block_indices_tlist, axis=0)
    #num of plotting points
    cdef ITYPE_t tavg_length = block_indices_tlist.shape[0]

    #only used for plotting force
    cdef dict plot_data_dict = dict()

    print 'initialization of variables'
    #storing the scale factor
    params_dict['barrier_height'] = barrier_height
    params_dict['energy_scale_factor'] = energy_scale_factor
    params_dict['power_scale_factor'] = power_scale_factor
    params_dict['force_scale_factor'] = force_scale_factor

    #dividing time list by force period
    get_scaled_data(tlist_block_wise, time_scale_factor)
    plot_data_dict['t/T'] = tlist_block_wise

    #buffer to store the variables
    cdef np.ndarray[DTYPE_t,ndim=1] E_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] U_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] K_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] K_com_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t,ndim=1] work_rate_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t,ndim=1] diss_rate_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] spring_lengths_tsteps = np.empty([tlength, EDGES],dtype=DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] spring_bits_tsteps = np.empty([tlength, EDGES],dtype=ITYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] num_spring_transitions_tsteps= np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=1] num_unique_states_tsteps= np.empty(tlength,dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] plot_num_unique_states_tsteps = np.empty(tavg_length, dtype=ITYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] diss_rate_lr_tsteps= np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] plot_diss_rate_lr_tsteps = np.empty(tavg_length, dtype=DTYPE)

#    cdef np.ndarray[DTYPE_t,ndim=1] system_state_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] d_relaxed_state_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] avg_tlist = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] num_spring_transitions_tavg= np.empty(tavg_length,dtype=DTYPE)

    get_lr_diss_rate_c(xilists_tsteps, yilists_tsteps, tlist,
            forcing_direction, params_dict, Amat, diss_rate_lr_tsteps)
    get_scaled_block_avg(diss_rate_lr_tsteps, plot_diss_rate_lr_tsteps, block_size, power_scale_factor)
    plot_data_dict['<dissipation_rate_lr(t)/(Barrier/T)>_T'] = plot_diss_rate_lr_tsteps

    get_spring_lengths_tsteps(xilists_tsteps, yilists_tsteps, Amat, spring_lengths_tsteps)
################## GET SPRING EXTENSIONS, EXTENSION RATES, EXTENSION SQUARED, AND POTENTIAL ENERGY ###############
    get_spring_bits_tsteps(spring_lengths_tsteps, spring_bits_tsteps, params_dict)

    get_num_unique_spring_states(spring_bits_tsteps, num_unique_states_tsteps, plot_data_dict)
    plot_num_unique_states_tsteps = np.take(num_unique_states_tsteps, block_indices_tlist, axis=0)
    plot_data_dict['num_unique_states_tsteps'] = plot_num_unique_states_tsteps

    #calculate distance from relaxed state, store in E_buffer_data_arr
    get_distance_from_relaxed_state_tsteps(spring_bits_tsteps, tlist, params_dict, E_buffer_data_arr)
    #take avg over drive period, and store in system_state_tavg
    get_scaled_block_avg(E_buffer_data_arr, d_relaxed_state_tavg, block_size, 1.)
    #plot_data_dict['<springs state(t) (gray code to decimal)>_T'] = system_state_tavg
    plot_data_dict['<distance from relaxed state(t)>_T'] = d_relaxed_state_tavg

    #calculate number of ones from spring_bits_tsteps, and store in E_buffer_data_arr
    get_num_long_springs_tsteps(spring_bits_tsteps, E_buffer_data_arr)
    cdef np.ndarray[DTYPE_t,ndim=1] num_long_springs_tavg= np.empty(tavg_length,dtype=DTYPE)
    #calculate avg over drive period, and store in num_long_springs_tavg
    get_scaled_block_avg(E_buffer_data_arr, num_long_springs_tavg, block_size, 1.)
    plot_data_dict['<# long springs(t)>_T'] = num_long_springs_tavg
    #calculated the number of the bit flips, and store in num_spring_transitions
    get_num_spring_transitions_tsteps(spring_bits_tsteps, num_spring_transitions_tsteps)

    get_Utotal_tsteps(spring_lengths_tsteps, U_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] Utotal_tavg = np.empty(tavg_length,dtype=DTYPE)
    #calculate avg of potential energy over drive periods, and store in Utotal_tavg
    get_scaled_block_avg(U_buffer_data_arr, Utotal_tavg, block_size, energy_scale_factor)
    plot_data_dict['<U(t)/(Barrier)>_T'] = Utotal_tavg

    get_KE_tsteps(vxilists_tsteps, vyilists_tsteps, K_buffer_data_arr, K_com_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] KE_tavg = np.empty(tavg_length,dtype=DTYPE)
    #calculate avg over drive period for KE_total and KE_com/KE_total, and store
    get_scaled_block_avg(K_buffer_data_arr, KE_tavg, block_size, energy_scale_factor)
    plot_data_dict['<KE(t)/(Barrier)>_T'] = KE_tavg
    cdef np.ndarray[DTYPE_t, ndim=1] KEfraction_com_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(K_com_buffer_data_arr, KEfraction_com_tavg, block_size, 1.)
    plot_data_dict['<KE(t)_{c.o.m.}/KE>_T'] = KEfraction_com_tavg

    add_arrays(U_buffer_data_arr, K_buffer_data_arr, E_buffer_data_arr)
    cdef np.ndarray[DTYPE_t, ndim=1] Etotal_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(E_buffer_data_arr, Etotal_tavg, block_size, energy_scale_factor)
    plot_data_dict['<E(t)/(Barrier)>_T'] = Etotal_tavg

    #get_dissipation_rate(vxilists_tsteps, vyilists_tsteps, diss_rate_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] diss_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(diss_rate_tsteps, diss_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'] = diss_rate_tavg

    #get_work_rate(vxilists_tsteps, vyilists_tsteps, f_ext_tlist, work_rate_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] work_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(work_rate_tsteps, work_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'] = work_rate_tavg


    #calculate sum of bit flips over each drive period, and store in num_spring_transitions_tavg
    get_scaled_block_sum(num_spring_transitions_tsteps, num_spring_transitions_tavg, block_size, 1.)
    plot_data_dict['<# spring transitions(t)>_T'] = num_spring_transitions_tavg

    #check the violation
    cdef np.ndarray[DTYPE_t, ndim=1] net_EminusW_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] net_EminusW_rate = np.empty(tlength, dtype=DTYPE)

    get_net_EminusW_rate(E_buffer_data_arr, work_rate_tsteps, diss_rate_tsteps, net_EminusW_rate, DT)
    get_scaled_block_avg(net_EminusW_rate, net_EminusW_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<(Edot + dissipation_rate - work_rate)/(Barrier/T)>_T'] = net_EminusW_rate_tavg

    return plot_data_dict

cpdef get_perturbed_plot_data_dict(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                            DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] tlist, DTYPE_t[:] work_rate_tsteps,
                            DTYPE_t[:] diss_rate_tsteps, dict params_dict, DTYPE_t tol=1e-8):
    # get column 1: U_total, (KE_total, E_total), netEminusWrate; column2: state_bit_number(graycode), (num_ones, num_flips), KE_com / KE_total; column3: power, diss_rate, f_ext(t)
    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef ITYPE_t EDGES = params_dict['EDGES']
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t NODES = params_dict['NODES']
    cdef DTYPE_t DT = params_dict['DT']
    cdef ITYPE_t RELAX_STEPS = params_dict['RELAX_STEPS']
    cdef DTYPE_t FORCE_PERIOD = params_dict['FORCE_PERIOD']
    cdef DTYPE_t L1 = params_dict['L1']
    cdef DTYPE_t Lbarrier = params_dict['Lbarrier']
    cdef DTYPE_t barrier_height = U_spring(Lbarrier, params_dict)# should be calculated from zero, to edit the potential offset
    #dimensionless factors
    cdef DTYPE_t energy_scale_factor = 1./(barrier_height)
    cdef DTYPE_t power_scale_factor = FORCE_PERIOD*energy_scale_factor
    cdef DTYPE_t time_scale_factor = 1./FORCE_PERIOD
    cdef DTYPE_t force_scale_factor = (Lbarrier-L1)*energy_scale_factor

    #num of steps in one force period
    cdef ITYPE_t block_size = np.int(round(FORCE_PERIOD/DT))
    #plotting points time steps
    cdef np.ndarray[ITYPE_t, ndim=1] block_indices_tlist = np.arange(0,tlength,block_size)
    #plotting time
    cdef np.ndarray[DTYPE_t, ndim=1] tlist_block_wise = np.take(tlist, block_indices_tlist, axis=0)
    #num of plotting points
    cdef ITYPE_t tavg_length = block_indices_tlist.shape[0]
    cdef ITYPE_t dummyNode
    cdef dict plot_data_dict = dict()

    #storing the scale factor
    params_dict['barrier_height'] = barrier_height
    params_dict['energy_scale_factor'] = energy_scale_factor
    params_dict['power_scale_factor'] = power_scale_factor
    params_dict['force_scale_factor'] = force_scale_factor
#    params_dict['num_samples_per_period'] = num_samples_per_period

    #dividing time list by force period
    get_scaled_data(tlist_block_wise, time_scale_factor)
    plot_data_dict['t/T'] = tlist_block_wise

    print 'allocated memory'
    #buffer to store the variables
    cdef np.ndarray[DTYPE_t,ndim=1] E_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] U_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] K_buffer_data_arr = np.empty(tlength,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] K_com_buffer_data_arr = np.empty(tlength,dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] spring_lengths_tsteps = np.empty([tlength, EDGES],dtype=DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] spring_bits_tsteps = np.empty([tlength, EDGES],dtype=ITYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] num_spring_transitions_tsteps= np.empty(tlength,dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] avg_tlist = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] num_spring_transitions_tavg= np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] xilists_tavg = np.empty([tavg_length, NODES],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] yilists_tavg = np.empty([tavg_length, NODES],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] radius_gyration_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] x_com_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] y_com_tavg = np.empty(tavg_length,dtype=DTYPE)

    get_com_tsteps(xilists_tsteps, yilists_tsteps, x_com_tavg, y_com_tavg, block_size)
    get_radius_gyration(xilists_tsteps, yilists_tsteps, x_com_tavg, y_com_tavg, radius_gyration_tavg, block_size)
    plot_data_dict['<x_com(t)>_T'] = x_com_tavg
    plot_data_dict['<y_com(t)>_T'] = y_com_tavg
    plot_data_dict['<r_g(t)>_T'] = radius_gyration_tavg

    get_spring_lengths_tsteps(xilists_tsteps, yilists_tsteps, Amat, spring_lengths_tsteps)
    get_spring_bits_tsteps(spring_lengths_tsteps, spring_bits_tsteps, params_dict)

    #calculated the number of the bit flips, and store in num_spring_transitions
    get_num_spring_transitions_tsteps(spring_bits_tsteps, num_spring_transitions_tsteps)

    get_Utotal_tsteps(spring_lengths_tsteps, U_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] Utotal_tavg = np.empty(tavg_length,dtype=DTYPE)
    #calculate avg of potential energy over drive periods, and store in Utotal_tavg
    get_scaled_block_avg(U_buffer_data_arr, Utotal_tavg, block_size, energy_scale_factor)
    plot_data_dict['<U(t)/(Barrier)>_T'] = Utotal_tavg

    get_KE_tsteps(vxilists_tsteps, vyilists_tsteps, K_buffer_data_arr, K_com_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] KE_tavg = np.empty(tavg_length,dtype=DTYPE)
    #calculate avg over drive period for KE_total and KE_com/KE_total, and store
    get_scaled_block_avg(K_buffer_data_arr, KE_tavg, block_size, energy_scale_factor)
    plot_data_dict['<KE(t)/(Barrier)>_T'] = KE_tavg
    cdef np.ndarray[DTYPE_t, ndim=1] KEfraction_com_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(K_com_buffer_data_arr, KEfraction_com_tavg, block_size, 1.)
    plot_data_dict['<KE(t)_{c.o.m.}/KE>_T'] = KEfraction_com_tavg

    add_arrays(U_buffer_data_arr, K_buffer_data_arr, E_buffer_data_arr)
    cdef np.ndarray[DTYPE_t, ndim=1] Etotal_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(E_buffer_data_arr, Etotal_tavg, block_size, energy_scale_factor)
    plot_data_dict['<E(t)/(Barrier)>_T'] = Etotal_tavg

#    get_dissipation_rate(vxilists_tsteps, vyilists_tsteps, diss_rate_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] diss_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(diss_rate_tsteps, diss_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'] = diss_rate_tavg

#    get_work_rate(vxilists_tsteps, vyilists_tsteps, f_ext_tlist, work_rate_buffer_data_arr, params_dict)
    cdef np.ndarray[DTYPE_t, ndim=1] work_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    get_scaled_block_avg(work_rate_tsteps, work_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'] = work_rate_tavg

    #calculate sum of bit flips over each drive period, and store in num_spring_transitions_tavg
    get_scaled_block_sum(num_spring_transitions_tsteps, num_spring_transitions_tavg, block_size, 1.)
    plot_data_dict['<# spring transitions(t)>_T'] = num_spring_transitions_tavg

    #check the violation
    cdef np.ndarray[DTYPE_t, ndim=1] net_EminusW_rate_tavg = np.empty(tavg_length,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] net_EminusW_rate = np.empty(tlength, dtype=DTYPE)
    #cdef DTYPE_t dt = (avg_tlist[1] - avg_tlist[0])/time_scale_factor
    get_net_EminusW_rate(E_buffer_data_arr, work_rate_tsteps, diss_rate_tsteps, net_EminusW_rate, DT)
    get_scaled_block_avg(net_EminusW_rate, net_EminusW_rate_tavg, block_size, power_scale_factor)
    plot_data_dict['<(Edot + dissipation_rate - work_rate)/(Barrier/T)>_T'] = net_EminusW_rate_tavg

    return plot_data_dict

cdef get_com_tsteps(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:] x_com_tsteps,
                    DTYPE_t[:] y_com_tsteps, ITYPE_t block_size):
    cdef ITYPE_t data_size = xilists_tsteps.shape[0]
    cdef ITYPE_t avg_size = x_com_tsteps.size
    cdef ITYPE_t nodes = xilists_tsteps.shape[1]
    cdef ITYPE_t i, n, dummyNode
    cdef DTYPE_t x_com_tavg, y_com_tavg, x_com, y_com
    for n in range(avg_size-1):
        x_com_tavg = 0.
        y_com_tavg = 0.
        for i in range(block_size):
            x_com = 0.
            y_com = 0.
            for dummyNode in range(nodes):
                x_com += xilists_tsteps[block_size*n+i][dummyNode]/nodes
                y_com += yilists_tsteps[block_size*n+i][dummyNode]/nodes
            x_com_tavg += x_com
            y_com_tavg += y_com
        x_com_tsteps[n] = x_com_tavg/block_size
        y_com_tsteps[n] = y_com_tavg/block_size

    n +=1
    i=0
    x_com_tavg = 0.
    y_com_tavg = 0.
    while (n*block_size + i) < data_size:
        x_com = 0.
        y_com = 0.
        for dummyNode in range(nodes):
            x_com += xilists_tsteps[block_size*n+i][dummyNode]/nodes
            y_com += yilists_tsteps[block_size*n+i][dummyNode]/nodes
        x_com_tavg += x_com
        y_com_tavg += y_com
        i+=1
    x_com_tsteps[n] = x_com_tavg/i
    y_com_tsteps[n] = y_com_tavg/i


cdef get_radius_gyration(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:] x_com_tavg,
                        DTYPE_t[:] y_com_tavg, DTYPE_t[:] radius_gyration_tsteps, ITYPE_t block_size):
    cdef ITYPE_t data_size = xilists_tsteps.shape[0]
    cdef ITYPE_t avg_size = x_com_tavg.shape[0]
    cdef ITYPE_t nodes = xilists_tsteps.shape[1]
    cdef ITYPE_t i, n, dummyNode
    cdef DTYPE_t radius_gyration_squared, radius_gyration
    for n in range(avg_size-1):
        radius_gyration = 0.
        for i in range(block_size):
            radius_gyration_squared = 0.
            for dummyNode in range(nodes):
                radius_gyration_squared += (xilists_tsteps[block_size*n+i][dummyNode]-
                    x_com_tavg[n])*(xilists_tsteps[block_size*n+i][dummyNode]-x_com_tavg[n])/nodes
                radius_gyration_squared += (yilists_tsteps[block_size*n+i][dummyNode]-
                    y_com_tavg[n])*(yilists_tsteps[block_size*n+i][dummyNode]-y_com_tavg[n])/nodes
            radius_gyration += sqrt(radius_gyration_squared)
        radius_gyration_tsteps[n] = radius_gyration/block_size
    n +=1
    i=0
    radius_gyration = 0.
    while (n*block_size + i) < data_size:
        radius_gyration_squared = 0.
        for dummyNode in range(nodes):
            radius_gyration_squared += (xilists_tsteps[block_size*n+i][dummyNode]-
                x_com_tavg[n])*(xilists_tsteps[block_size*n+i][dummyNode]-x_com_tavg[n])/nodes
            radius_gyration_squared += (yilists_tsteps[block_size*n+i][dummyNode]-
                y_com_tavg[n])*(yilists_tsteps[block_size*n+i][dummyNode]-y_com_tavg[n])/nodes
        radius_gyration += sqrt(radius_gyration_squared)
        i+=1
    radius_gyration_tsteps[n] = (radius_gyration)/i


cpdef get_driven_angular_momentum(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                            DTYPE_t[:,:] vyilists_tsteps, dict params_dict, DTYPE_t[:] L_tsteps):
    cdef DTYPE_t mass = np.float64(params_dict['MASS'])
    cdef DTYPE_t x_ref, y_ref, x_i_ref, y_i_ref, L_i, vxi, vyi
    cdef ITYPE_t anchor_0 = np.int64(params_dict['ANCHORED_NODES'][0])
    cdef ITYPE_t nodes = np.int64(params_dict['NODES'])
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef ITYPE_t tindex, nodeindex
    x_ref = xilists_tsteps[tlength-1][anchor_0]
    y_ref = yilists_tsteps[tlength-1][anchor_0]

    for tindex in range(tlength):
        L_i = 0.
        for nodeindex in range(nodes):
            x_i_ref = xilists_tsteps[tindex][nodeindex] - x_ref
            y_i_ref = yilists_tsteps[tindex][nodeindex] - y_ref
            vxi = vxilists_tsteps[tindex][nodeindex]
            vyi = vyilists_tsteps[tindex][nodeindex]
            L_i += mass*(x_i_ref*vyi - y_i_ref*vxi)
        L_tsteps[tindex] = L_i


cdef get_num_unique_spring_states(ITYPE_t[:,:] spring_bits_tsteps, ITYPE_t[:] num_unique_states_tsteps,
                                    dict plot_data_dict):
    cdef ITYPE_t tlength = spring_bits_tsteps.shape[0]
    cdef ITYPE_t tindex, edgeIndex
    cdef ITYPE_t edges = spring_bits_tsteps.shape[1]
    cdef np.ndarray[ITYPE_t, ndim=1] state_id_tsteps = np.empty(tlength, dtype=ITYPE)

    for tindex in range(tlength):
        state_id_tsteps[tindex] = 0
        for edgeIndex in range(edges):
            state_id_tsteps[tindex] = state_id_tsteps[tindex] + spring_bits_tsteps[tindex,edgeIndex]*pow(2,edgeIndex)

    unique_states = [state_id_tsteps[0]]
    num_unique_states_tsteps[0] = 1

    for tindex in range(1,tlength):
        if state_id_tsteps[tindex] not in unique_states:
            unique_states.append(state_id_tsteps[tindex])
            num_unique_states_tsteps[tindex] = num_unique_states_tsteps[tindex-1] +1
        else:
            num_unique_states_tsteps[tindex] = num_unique_states_tsteps[tindex-1]
    plot_data_dict['unique_states_visited'] = np.array(unique_states)


cpdef get_sum_between_sample_indices(ITYPE_t[:] data_array, ITYPE_t[:] sample_indices, DTYPE_t deltaT, DTYPE_t[:] sum_array):
# assumes length of avg_array is ceiling of length of data_array divided by block_size
    cdef ITYPE_t data_size = data_array.size
    cdef ITYPE_t tot_sample_indices = sample_indices.size
    cdef ITYPE_t i = 0
    cdef ITYPE_t n
    cdef DTYPE_t running_sum

    for n in range(tot_sample_indices):
        running_sum = 0.
        while i <= sample_indices[n]:
            running_sum += data_array[i]
            i += 1
        sum_array[n] = running_sum/deltaT


cpdef get_normal_mode_participation_ratios(DTYPE_t[:] xilist, DTYPE_t[:] yilist, DTYPE_t[:] participation_ratios,
                            dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t tol):
    get_normal_mode_participation_ratios_c(xilist, yilist, participation_ratios, params_dict, Amat, tol)


cdef get_normal_mode_participation_ratios_c(DTYPE_t[:] xilist, DTYPE_t[:] yilist, DTYPE_t[:] participation_ratios,
                            dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t tol):
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, dummyNode
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] eigvecs = np.empty([num_dof,num_dof], dtype=DTYPE)
    cdef ITYPE_t i_driven = params_dict['MAX_DEG_VERTEX_I']
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef DTYPE_t damping_rate = params_dict['DAMPING_RATE']
    cdef DTYPE_t mass = params_dict['MASS']

    for i in range(nodes):
        sigmaj_Dij = 0.
        sigmaj_Dijplus = 0.
        sigmaj_Diplusjplus = 0.
        for j in range(nodes):
            if Amat[i][j] == 0:
                Dmat[2*i][2*j] = 0.
                Dmat[2*i][2*j+1] = 0.
                Dmat[2*i+1][2*j] = 0.
                Dmat[2*i+1][2*j+1] = 0.
            else:
                xij = xilist[i]-xilist[j]
                yij = yilist[i]-yilist[j]
                dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                Aij = Amat[i][j]
                kij = K_eff(params_dict, dij)/mass
                Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                sigmaj_Dij += Dmat[2*i][2*j]
                sigmaj_Dijplus += Dmat[2*i][2*j+1]
                sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
        Dmat[2*i][2*i] = -sigmaj_Dij
        Dmat[2*i][2*i+1] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
    for dummyNode in anchored_nodes:
        Dmat[2*dummyNode][2*dummyNode] += 1e2
        Dmat[2*dummyNode+1][2*dummyNode+1] += 1e2

    eigvals, eigvecs = linalg.eigh(Dmat)

    for i in range(num_dof):
        participation_ratios[i] = get_participation_ratio(eigvecs[:,i])


cpdef get_normal_mode_greens_data(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction,
                            dict params_dict, ITYPE_t[:,:] Amat,
                            DTYPE_t[:,:] eigvals_tsteps, DTYPE_t[:,:] forcing_overlap_tsteps
                            ):
    get_normal_mode_greens_data_c(xilists_tsteps, yilists_tsteps, tlist, forcing_direction,
                            params_dict, Amat, eigvals_tsteps, forcing_overlap_tsteps
                            )

cdef get_normal_mode_greens_data_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction,
                            dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t[:,:] eigvals_tsteps, DTYPE_t[:,:] forcing_overlap_tsteps
                            ):
    cdef ITYPE_t tlength = tlist.size
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, tindex, omega_i
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] eigvecs = np.empty([num_dof,num_dof], dtype=DTYPE)
    cdef ITYPE_t i_driven = params_dict['MAX_DEG_VERTEX_I']
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef np.ndarray[DTYPE_t, ndim=1] omega_plot = np.array(params_dict['omega_plot'])
    cdef ITYPE_t num_omegas = omega_plot.shape[0]
    cdef np.ndarray[DTYPE_t,ndim=1] T_list = np.array(params_dict['T_list'])
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef DTYPE_t omega_val #product_div_rates, div_overlap, div_perp_overlap
    cdef DTYPE_t damping_rate = params_dict['DAMPING_RATE']
    cdef DTYPE_t mass = params_dict['MASS']
    cdef ITYPE_t dummyNode, switch_time_index


    for tindex in range(tlength):
        for i in range(nodes):
            sigmaj_Dij = 0.
            sigmaj_Dijplus = 0.
            sigmaj_Diplusjplus = 0.
            for j in range(nodes):
                if Amat[i][j] == 0:
                    Dmat[2*i][2*j] = 0.
                    Dmat[2*i][2*j+1] = 0.
                    Dmat[2*i+1][2*j] = 0.
                    Dmat[2*i+1][2*j+1] = 0.
                else:
                    xij = xilists_tsteps[tindex][i]-xilists_tsteps[tindex][j]
                    yij = yilists_tsteps[tindex][i]-yilists_tsteps[tindex][j]
                    dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                    Aij = Amat[i][j]
                    kij = K_eff(params_dict, dij)/mass
                    Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                    Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                    sigmaj_Dij += Dmat[2*i][2*j]
                    sigmaj_Dijplus += Dmat[2*i][2*j+1]
                    sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
            Dmat[2*i][2*i] = -sigmaj_Dij
            Dmat[2*i][2*i+1] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
        for dummyNode in anchored_nodes:
            Dmat[2*dummyNode][2*dummyNode] += 1e2
            Dmat[2*dummyNode+1][2*dummyNode+1] += 1e2

        eigvals, eigvecs = linalg.eigh(Dmat)

        for i in range(num_dof):
            eigvals_tsteps[tindex][i] = sqrt(round(1e8*fabs(eigvals[i]))/1e8)
            forcing_overlap_tsteps[i][tindex] = 0.

            for j in range(num_dof):
                forcing_overlap_tsteps[i][tindex] += eigvecs[j,i]*forcing_direction[j][tindex]

            forcing_overlap_tsteps[i][tindex] = forcing_overlap_tsteps[i][tindex]*forcing_overlap_tsteps[i][tindex]


cdef DTYPE_t get_participation_ratio( DTYPE_t[:] eigvec):
    cdef ITYPE_t num_dof = eigvec.size
    cdef DTYPE_t sum_elements_squared, sum_elements_fourth_power, participation_ratio
    cdef ITYPE_t dof_index
    sum_elements_squared = 0.
    sum_elements_fourth_power = 0.
    for dof_index in range(num_dof):
        sum_elements_squared += eigvec[dof_index]*eigvec[dof_index]
        sum_elements_fourth_power += eigvec[dof_index]*eigvec[dof_index]*eigvec[dof_index]*eigvec[dof_index]
    participation_ratio = sum_elements_squared/(num_dof*sum_elements_fourth_power)
    return participation_ratio


cpdef get_num_phase_space_contracting_dims(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps,
            dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t tol, DTYPE_t[:] num_contracting_dims, DTYPE_t[:] num_stationary_dims,
             DTYPE_t[:] num_diverging_dims):
    get_num_phase_space_contracting_dims_c(xilists_tsteps, yilists_tsteps, params_dict, Amat, tol, num_contracting_dims,
            num_stationary_dims, num_diverging_dims)

cdef get_num_phase_space_contracting_dims_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps,
            dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t tol, DTYPE_t[:] num_contracting_dims, DTYPE_t[:] num_stationary_dims,
             DTYPE_t[:] num_diverging_dims):
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, tindex, diss_tindex
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef DTYPE_t mass = params_dict['MASS']
    cdef ITYPE_t dummyNode

    for tindex in range(tlength):
        for i in range(nodes):
            sigmaj_Dij = 0.
            sigmaj_Dijplus = 0.
            sigmaj_Diplusjplus = 0.
            for j in range(nodes):
                if Amat[i][j] == 0:
                    Dmat[2*i][2*j] = 0.
                    Dmat[2*i][2*j+1] = 0.
                    Dmat[2*i+1][2*j] = 0.
                    Dmat[2*i+1][2*j+1] = 0.
                else:
                    xij = xilists_tsteps[tindex][i]-xilists_tsteps[tindex][j]
                    yij = yilists_tsteps[tindex][i]-yilists_tsteps[tindex][j]
                    dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                    Aij = Amat[i][j]
                    kij = K_eff(params_dict, dij)/mass
                    Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                    Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                    sigmaj_Dij += Dmat[2*i][2*j]
                    sigmaj_Dijplus += Dmat[2*i][2*j+1]
                    sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
            Dmat[2*i][2*i] = -sigmaj_Dij
            Dmat[2*i][2*i+1] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
        for dummyNode in anchored_nodes:
            Dmat[2*dummyNode][2*dummyNode] += 1e2
            Dmat[2*dummyNode+1][2*dummyNode+1] += 1e2

        eigvals = linalg.eigvalsh(Dmat)

        num_contracting_dims[tindex] = 0.
        num_stationary_dims[tindex] = 0.
        num_diverging_dims[tindex] = 0.
        for i in range(num_dof):
            if eigvals[i] > tol:
                num_contracting_dims[tindex] += 1
            elif eigvals[i] > -tol:
                num_stationary_dims[tindex] += 1
            else:
                num_diverging_dims[tindex] += 1

cpdef get_tstop(DTYPE_t[:] num_spring_transitions_tsteps, DTYPE_t[:] t_by_T, dict params_dict):
    cdef ITYPE_t last_nz_index, tsteps, i, nz_length
    cdef DTYPE_t t_stop
    tsteps = num_spring_transitions_tsteps.size
    cdef np.ndarray[ITYPE_t,ndim=1] non_zero_transitions_tsteps = np.empty(tsteps, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t,ndim=1] nz_indices
    for i in range(tsteps):
        non_zero_transitions_tsteps[i] = np.int(num_spring_transitions_tsteps[i] > 0.5)
    nz_indices = np.nonzero(non_zero_transitions_tsteps)[0]
    nz_length = nz_indices.size
    last_nz_index = nz_indices[nz_length-1]
    t_stop = t_by_T[last_nz_index]
    params_dict['t_stop'] = t_stop

cpdef get_sampled_lr_diss_rate_phase_flow_div_dirns_KE_unstable_dirns(DTYPE_t[:,:] xilists_tsteps,
                        DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps, DTYPE_t[:,:] vyilists_tsteps,
                        DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction, dict params_dict, ITYPE_t[:,:] Amat,
                        DTYPE_t[:] diss_rate_linear_response, DTYPE_t[:] fraction_div_phase_space_flow_per_dirn,
                        DTYPE_t[:] fraction_unstable_KE_per_dirn, DTYPE_t[:] max_unstable_KE, DTYPE_t[:] max_KE_all_dirns,
                        DTYPE_t tol):
    get_sampled_lr_diss_rate_phase_flow_div_dirns_KE_unstable_dirns_c(xilists_tsteps, yilists_tsteps,
                        vxilists_tsteps, vyilists_tsteps, tlist, forcing_direction, params_dict, Amat,
                        diss_rate_linear_response, fraction_div_phase_space_flow_per_dirn, fraction_unstable_KE_per_dirn,
                        max_unstable_KE, max_KE_all_dirns, tol)

cdef get_sampled_lr_diss_rate_phase_flow_div_dirns_KE_unstable_dirns_c(DTYPE_t[:,:] xilists_tsteps,
                        DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps, DTYPE_t[:,:] vyilists_tsteps,
                        DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction, dict params_dict, ITYPE_t[:,:] Amat,
                        DTYPE_t[:] diss_rate_linear_response, DTYPE_t[:] fraction_div_phase_space_flow_per_dirn,
                        DTYPE_t[:] fraction_unstable_KE_per_dirn, DTYPE_t[:] max_unstable_KE, DTYPE_t[:] max_KE_all_dirns,
                        DTYPE_t tol):
    cdef ITYPE_t tlength = tlist.size
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus, v_squared, phase_flow_tangent_mag
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, tindex, diss_tindex, num_div_dims
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] phase_space_flow_tangent = np.empty(2*num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] velocity_vec = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] velocity_overlap = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] unstable_velocity_overlap = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] forcing_overlap = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] eigvecs = np.empty([num_dof,num_dof], dtype=DTYPE)
    cdef ITYPE_t i_driven = params_dict['MAX_DEG_VERTEX_I']
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef np.ndarray[DTYPE_t,ndim=1] T_list = np.array(params_dict['T_list'])
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef DTYPE_t omega_val, diss_rate, div_rate, phase_flow_div_overlap
    cdef DTYPE_t damping_rate = params_dict['DAMPING_RATE']
    cdef DTYPE_t force_amp = params_dict['FORCE_AMP']
    cdef DTYPE_t mass = params_dict['MASS']
    cdef ITYPE_t dummyNode, switch_time_index

    for tindex in range(tlength):
        for i in range(nodes):
            velocity_vec[2*i] = vxilists_tsteps[tindex,i]
            velocity_vec[2*i+1] = vyilists_tsteps[tindex,i]
            if tindex != 0:
                phase_space_flow_tangent[2*i] = (vxilists_tsteps[tindex,i]-vxilists_tsteps[tindex-1,i])/(tlist[tindex]-
                                                    tlist[tindex-1])
                phase_space_flow_tangent[2*i+1] = (vyilists_tsteps[tindex,i]-vyilists_tsteps[tindex-1,i])/(tlist[tindex]-
                                                    tlist[tindex-1])
            else:
                phase_space_flow_tangent[2*i] = 0.
                phase_space_flow_tangent[2*i+1] = 0.

        phase_space_flow_tangent[num_dof:] = velocity_vec
        phase_flow_tangent_mag = np.sqrt(np.dot(phase_space_flow_tangent, phase_space_flow_tangent))
        phase_space_flow_tangent = phase_space_flow_tangent/phase_flow_tangent_mag
        v_squared = np.dot(velocity_vec, velocity_vec)

        for i in range(nodes):
            sigmaj_Dij = 0.
            sigmaj_Dijplus = 0.
            sigmaj_Diplusjplus = 0.
            for j in range(nodes):
                if Amat[i][j] == 0:
                    Dmat[2*i][2*j] = 0.
                    Dmat[2*i][2*j+1] = 0.
                    Dmat[2*i+1][2*j] = 0.
                    Dmat[2*i+1][2*j+1] = 0.
                else:
                    xij = xilists_tsteps[tindex][i]-xilists_tsteps[tindex][j]
                    yij = yilists_tsteps[tindex][i]-yilists_tsteps[tindex][j]
                    dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                    Aij = Amat[i][j]
                    kij = K_eff(params_dict, dij)/mass
                    Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                    Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                    sigmaj_Dij += Dmat[2*i][2*j]
                    sigmaj_Dijplus += Dmat[2*i][2*j+1]
                    sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
            Dmat[2*i][2*i] = -sigmaj_Dij
            Dmat[2*i][2*i+1] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
        for dummyNode in anchored_nodes:
            Dmat[2*dummyNode][2*dummyNode] += 1e2
            Dmat[2*dummyNode+1][2*dummyNode+1] += 1e2

        eigvals, eigvecs = linalg.eigh(Dmat)
        fraction_div_phase_space_flow_per_dirn[tindex] = 0.
        fraction_unstable_KE_per_dirn[tindex] = 0.
        unstable_velocity_overlap = np.zeros(num_dof, dtype=DTYPE)
        num_div_dims = 0
        for i in range(num_dof):
            forcing_overlap[i] = np.dot(eigvecs[:,i],forcing_direction[:,tindex])
            velocity_overlap[i] = np.dot(eigvecs[:,i],velocity_vec)
            velocity_overlap[i] = velocity_overlap[i]*velocity_overlap[i]
            if eigvals[i] < -tol:
                div_rate = 0.5*(np.sqrt(damping_rate*damping_rate-4*eigvals[i]) - damping_rate)
                phase_flow_div_overlap = (np.dot(phase_space_flow_tangent[num_dof:], eigvecs[:,i]) +
                                            div_rate*np.dot(phase_space_flow_tangent[:num_dof], eigvecs[:,i]))
                phase_flow_div_overlap = phase_flow_div_overlap*phase_flow_div_overlap/(div_rate*div_rate+1)
                fraction_div_phase_space_flow_per_dirn[tindex] += phase_flow_div_overlap
                fraction_unstable_KE_per_dirn[tindex] += velocity_overlap[i]/v_squared
                unstable_velocity_overlap[i] = velocity_overlap[i]
                num_div_dims += 1

        if num_div_dims > 0:
            fraction_div_phase_space_flow_per_dirn[tindex] = fraction_div_phase_space_flow_per_dirn[tindex]/num_div_dims
            fraction_unstable_KE_per_dirn[tindex] = fraction_unstable_KE_per_dirn[tindex]/num_div_dims

        max_KE_all_dirns[tindex] = 0.5*mass*np.amax(velocity_overlap)
        max_unstable_KE[tindex] = 0.5*mass*np.amax(unstable_velocity_overlap)

        if tlist[tindex] <= switch_times_list[0]:
            omega_val = 2*np.pi/T_list[0]
        else:
            for switch_time_index in range(1,switch_times_list.size):
                if tlist[tindex] <= switch_times_list[switch_time_index]:
                    omega_val = 2*np.pi/T_list[switch_time_index]
                    break

        diss_rate = 0.
        for i in range(num_dof):
            if eigvals[i] > 0.:
                diss_rate += (0.5*damping_rate*force_amp*force_amp*omega_val*omega_val*
                                    (forcing_overlap[i]*forcing_overlap[i])/
                    ((eigvals[i]-omega_val*omega_val)*(eigvals[i]-omega_val*omega_val) +
                    damping_rate*damping_rate*omega_val*omega_val/(mass*mass)) )
        diss_rate_linear_response[tindex] = diss_rate


cpdef get_lr_diss_rate(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps,
                        DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction, dict params_dict, ITYPE_t[:,:] Amat,
                        DTYPE_t[:] diss_rate_linear_response):
    get_lr_diss_rate_c(xilists_tsteps, yilists_tsteps, tlist,
            forcing_direction, params_dict, Amat, diss_rate_linear_response)

cdef get_lr_diss_rate_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps,
                        DTYPE_t[:] tlist, DTYPE_t[:,:] forcing_direction, dict params_dict, ITYPE_t[:,:] Amat,
                        DTYPE_t[:] diss_rate_linear_response):
    cdef ITYPE_t tlength = tlist.size
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, tindex, diss_tindex
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] forcing_overlap = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] eigvecs = np.empty([num_dof,num_dof], dtype=DTYPE)
    cdef ITYPE_t i_driven = params_dict['MAX_DEG_VERTEX_I']
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef np.ndarray[DTYPE_t,ndim=1] T_list = np.array(params_dict['T_list'])
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef DTYPE_t omega_val, diss_rate
    cdef DTYPE_t damping_rate = params_dict['DAMPING_RATE']
    cdef DTYPE_t force_amp = params_dict['FORCE_AMP']
    cdef DTYPE_t mass = params_dict['MASS']
    cdef ITYPE_t dummyNode, switch_time_index

    for tindex in range(tlength):

        for i in range(nodes):
            sigmaj_Dij = 0.
            sigmaj_Dijplus = 0.
            sigmaj_Diplusjplus = 0.
            for j in range(nodes):
                if Amat[i][j] == 0:
                    Dmat[2*i][2*j] = 0.
                    Dmat[2*i][2*j+1] = 0.
                    Dmat[2*i+1][2*j] = 0.
                    Dmat[2*i+1][2*j+1] = 0.
                else:
                    xij = xilists_tsteps[tindex][i]-xilists_tsteps[tindex][j]
                    yij = yilists_tsteps[tindex][i]-yilists_tsteps[tindex][j]
                    dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                    Aij = Amat[i][j]
                    kij = K_eff(params_dict, dij)/mass
                    Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                    Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                    Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                    sigmaj_Dij += Dmat[2*i][2*j]
                    sigmaj_Dijplus += Dmat[2*i][2*j+1]
                    sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
            Dmat[2*i][2*i] = -sigmaj_Dij
            Dmat[2*i][2*i+1] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i] = -sigmaj_Dijplus
            Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
        for dummyNode in anchored_nodes:
            Dmat[2*dummyNode][2*dummyNode] += 1e2
            Dmat[2*dummyNode+1][2*dummyNode+1] += 1e2

        eigvals, eigvecs = linalg.eigh(Dmat)

        for i in range(num_dof):
            forcing_overlap[i] = np.dot(eigvecs[:,i],forcing_direction[:,tindex])

        if tlist[tindex] <= switch_times_list[0]:
            omega_val = 2*np.pi/T_list[0]
        else:
            for switch_time_index in range(1,switch_times_list.size):
                if tlist[tindex] <= switch_times_list[switch_time_index]:
                    omega_val = 2*np.pi/T_list[switch_time_index]
                    break

        diss_rate = 0.
        for i in range(num_dof):
            if eigvals[i] > 0.:
                diss_rate += (0.5*damping_rate*force_amp*force_amp*omega_val*omega_val*
                                    (forcing_overlap[i]*forcing_overlap[i])/
                    ((eigvals[i]-omega_val*omega_val)*(eigvals[i]-omega_val*omega_val) +
                    damping_rate*damping_rate*omega_val*omega_val/(mass*mass)) )
        diss_rate_linear_response[tindex] = diss_rate


cpdef get_perturbed_forcing_direction(DTYPE_t[:] xilist, DTYPE_t[:] yilist, DTYPE_t[:] forcing_direction,
                        dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t rotation_angle,
                        DTYPE_t[:] perturbed_forcing_direction, DTYPE_t[:] degenerate_lambda_info,
                        DTYPE_t anchored_participation_threshold=0.05, DTYPE_t eig_threshold=0.7):
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef ITYPE_t num_dof = 2*nodes
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef np.ndarray[DTYPE_t,ndim=2] Dmat = np.empty([num_dof, num_dof], dtype=DTYPE)
    cdef ITYPE_t i, j, dummyNode, degenerate_eig_i, degenerate_eig_j
    cdef np.ndarray[DTYPE_t,ndim=1] eigvals = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] delta_eigvals = np.empty(num_dof*num_dof, dtype=DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=1] sorted_delta_eigval_indices = np.empty(num_dof-1, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] forcing_overlap = np.empty(num_dof, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] eigvecs = np.empty([num_dof,num_dof], dtype=DTYPE)
    cdef ITYPE_t i_driven = params_dict['MAX_DEG_VERTEX_I']
    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef DTYPE_t anchored_participation, old_projection_i, old_projection_j, new_projection_i, new_projection_j
    cdef DTYPE_t damping_rate = params_dict['DAMPING_RATE']
    cdef DTYPE_t force_amp = params_dict['FORCE_AMP']
    cdef DTYPE_t mass = params_dict['MASS']

    for i in range(nodes):
        sigmaj_Dij = 0.
        sigmaj_Dijplus = 0.
        sigmaj_Diplusjplus = 0.
        for j in range(nodes):
            if Amat[i][j] == 0:
                Dmat[2*i][2*j] = 0.
                Dmat[2*i][2*j+1] = 0.
                Dmat[2*i+1][2*j] = 0.
                Dmat[2*i+1][2*j+1] = 0.
            else:
                xij = xilist[i]-xilist[j]
                yij = yilist[i]-yilist[j]
                dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                Aij = Amat[i][j]
                kij = K_eff(params_dict, dij)/mass
                Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                sigmaj_Dij += Dmat[2*i][2*j]
                sigmaj_Dijplus += Dmat[2*i][2*j+1]
                sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
        Dmat[2*i][2*i] = -sigmaj_Dij
        Dmat[2*i][2*i+1] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus
    for dummyNode in anchored_nodes:
        Dmat[2*dummyNode][2*dummyNode] += 1e3
        Dmat[2*dummyNode+1][2*dummyNode+1] += 1e3

    eigvals, eigvecs = linalg.eigh(Dmat)

    for i in range(num_dof):
        for j in range(num_dof):
            if j <= i or eigvals[i] < eig_threshold:
                delta_eigvals[num_dof*i+j] = 1e4
            else:
                delta_eigvals[num_dof*i+j] = eigvals[j] - eigvals[i]

    sorted_delta_eigval_indices = np.argsort(delta_eigvals)

    for i in range(num_dof*num_dof-1):
        degenerate_eig_i = sorted_delta_eigval_indices[i] // num_dof
        degenerate_eig_j = sorted_delta_eigval_indices[i] % num_dof
        anchored_participation = 0.
        for dummyNode in anchored_nodes:
            anchored_participation += (eigvecs[2*dummyNode,degenerate_eig_i]*eigvecs[2*dummyNode,degenerate_eig_i]+
                eigvecs[2*dummyNode+1,degenerate_eig_i]*eigvecs[2*dummyNode+1,degenerate_eig_i])
            anchored_participation += (eigvecs[2*dummyNode,degenerate_eig_j]*eigvecs[2*dummyNode,degenerate_eig_j]+
                eigvecs[2*dummyNode+1,degenerate_eig_j]*eigvecs[2*dummyNode+1,degenerate_eig_j])
        if sqrt(anchored_participation) <= anchored_participation_threshold:
            break

    print 'degenerate eigenvalue is %.2f' %(eigvals[degenerate_eig_i])
    print 'difference between degenerate eigenvalues is %.2f'%(eigvals[degenerate_eig_j] - eigvals[degenerate_eig_i])
    degenerate_lambda_info[0] = eigvals[degenerate_eig_i]
    degenerate_lambda_info[1] = eigvals[degenerate_eig_j] - eigvals[degenerate_eig_i]
    degenerate_lambda_info[2] = sqrt(anchored_participation)

    for i in range(num_dof):
        perturbed_forcing_direction[i] = forcing_direction[i]
    old_projection_i = np.dot(forcing_direction, eigvecs[:,degenerate_eig_i])
    old_projection_j = np.dot(forcing_direction, eigvecs[:,degenerate_eig_j])
    new_projection_i = old_projection_i*cos(rotation_angle) - old_projection_j*sin(rotation_angle)
    new_projection_j = old_projection_i*sin(rotation_angle) + old_projection_j*cos(rotation_angle)
    for i in range(num_dof):
        perturbed_forcing_direction[i] -= old_projection_i*eigvecs[i,degenerate_eig_i]
        perturbed_forcing_direction[i] -= old_projection_j*eigvecs[i,degenerate_eig_j]
        perturbed_forcing_direction[i] += new_projection_i*eigvecs[i,degenerate_eig_i]
        perturbed_forcing_direction[i] += new_projection_j*eigvecs[i,degenerate_eig_j]

    if abs(np.dot(perturbed_forcing_direction,perturbed_forcing_direction) - np.dot(forcing_direction,forcing_direction)) > 1e-12:
        print 'norm not preserved by rotation'



cpdef get_hist_tsteps(DTYPE_t[:,:] vals_tsteps, DTYPE_t[:] bins, ITYPE_t[:,:] vals_hist_tsteps):
    cdef ITYPE_t tindex, bindex
    cdef ITYPE_t numbins = bins.shape[0]-1
    cdef np.ndarray[ITYPE_t,ndim=1] vals_hist = np.empty(numbins, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] dummy_bins = np.empty(numbins, dtype=DTYPE)
    cdef ITYPE_t tlength = vals_tsteps.shape[0]

    for tindex in range(tlength):
        vals_hist, dummy_bins = np.histogram(vals_tsteps[tindex], bins=bins)
        for bindex in range(numbins):
            vals_hist_tsteps[bindex][tindex] = vals_hist[bindex]

cpdef get_binned_mode_overlap_tsteps(DTYPE_t[:,:] mode_overlap_tsteps, ITYPE_t[:,:] eig_hist_tsteps, DTYPE_t[:,:] binned_mode_overlap_tsteps):
    cdef ITYPE_t tindex, bindex, num_modes_in_bin, mode_in_bin_index, mode_bin_start
    cdef ITYPE_t numbins = eig_hist_tsteps.shape[0]
    cdef ITYPE_t tlength = eig_hist_tsteps.shape[1]

    for tindex in range(tlength):
        mode_bin_start=0
        for bindex in range(numbins):
            binned_mode_overlap_tsteps[bindex][tindex] = 0.
            num_modes_in_bin = eig_hist_tsteps[bindex][tindex]
            for mode_in_bin_index in range(num_modes_in_bin):
                binned_mode_overlap_tsteps[bindex][tindex] += mode_overlap_tsteps[mode_bin_start+mode_in_bin_index][tindex]
            mode_bin_start += num_modes_in_bin

cpdef get_avged_diss_spectrum_tsteps(DTYPE_t[:,:] diss_spectrum_tsteps, ITYPE_t omegas_per_bin, ITYPE_t num_bins,
                                    DTYPE_t[:,:] avged_diss_spectrum_tsteps):
    """
    params:
        greens_overlap_tsteps:2d numpy, overlapped greens function, indexing omega(in discrete order) and time
        num_omegas: int , num of discrete omega in greens overlap tsteps.
        num_bins_green: int, num of bins

    calculate:
        binned_greens_overlap_tsteps: 2d numpy, indexing omega and time, first index range (0, greens_bins.shape[0]-1)
    """
    cdef ITYPE_t tindex, bindex, omega_i
    cdef ITYPE_t tlength = diss_spectrum_tsteps.shape[1]
    for tindex in range(tlength):
        for bindex in range(num_bins):
            avged_diss_spectrum_tsteps[bindex][tindex] = 0.
            for omega_i in range(omegas_per_bin):
                avged_diss_spectrum_tsteps[bindex][tindex] += diss_spectrum_tsteps[bindex*omegas_per_bin+omega_i][tindex]/omegas_per_bin


cpdef get_dynamical_matrix(DTYPE_t[:] xilist, DTYPE_t[:] yilist, dict params_dict, ITYPE_t[:,:] Amat, DTYPE_t[:,:] Dmat):
    cdef DTYPE_t xij, yij, dij, kij, Aij, sigmaj_Dij, sigmaj_Dijplus, sigmaj_Diplusjplus
    cdef ITYPE_t i,j
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef DTYPE_t mass = params_dict['MASS']
    for i in range(nodes):
        sigmaj_Dij = 0.
        sigmaj_Dijplus = 0.
        sigmaj_Diplusjplus = 0.
        for j in range(nodes):
            if Amat[i][j] == 0:
                Dmat[2*i][2*j] = 0.
                Dmat[2*i][2*j+1] = 0.
                Dmat[2*i+1][2*j] = 0.
                Dmat[2*i+1][2*j+1] = 0.
            else:
                xij = xilist[i]-xilist[j]
                yij = yilist[i]-yilist[j]
                dij = max(sqrt(xij*xij+yij*yij), 1e-12)
                Aij = Amat[i][j]
                kij = K_eff(params_dict, dij)/mass
                Dmat[2*i][2*j] = -Aij*kij*(xij*xij)/(dij*dij)
                Dmat[2*i][2*j+1] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j] = -Aij*kij*(xij*yij)/(dij*dij)
                Dmat[2*i+1][2*j+1] = -Aij*kij*(yij*yij)/(dij*dij)
                sigmaj_Dij += Dmat[2*i][2*j]
                sigmaj_Dijplus += Dmat[2*i][2*j+1]
                sigmaj_Diplusjplus += Dmat[2*i+1][2*j+1]
        Dmat[2*i][2*i] = -sigmaj_Dij
        Dmat[2*i][2*i+1] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i] = -sigmaj_Dijplus
        Dmat[2*i+1][2*i+1] = -sigmaj_Diplusjplus

cpdef U_spring(DTYPE_t dij, dict params_dict):
    cdef DTYPE_t SPRING_K = params_dict['SPRING_K']
    cdef DTYPE_t u_spring
    if dij < 4./5:
        u_spring = 12.5*dij*dij-20.1*dij+8.09
    elif dij < 6./5:
        u_spring =  0.25*(dij-1.)*(dij-1.)
    elif dij < 14./5:
        u_spring = (1.77002*(dij-2.)*(dij-2.)*(dij-2.)*(dij-2.) -0.395508*(dij-2.)*(dij-2.)*(dij-2.)
        - 2.60937*(dij-2.)*(dij-2.) + 0.309375*(dij-2.)+1.)
    elif dij < 16./5:
        u_spring = 2.5*(dij-3.)*(dij-3.)
    else:
        u_spring = 10.*dij*dij-63.*dij+99.3
    u_spring = SPRING_K*u_spring
    return u_spring

cdef get_Utotal_tsteps(DTYPE_t[:,:] spring_lengths_tsteps, DTYPE_t[:] Utotal_tsteps, dict params_dict):
    cdef ITYPE_t tlength = spring_lengths_tsteps.shape[0]
#    cdef ITYPE_t EDGES = spring_lengths_tsteps.shape[1]
    cdef ITYPE_t tindex
#    cdef ITYPE_t edgeIndex
    for tindex in range(tlength):
        Utotal_tsteps[tindex] = get_Utotal(spring_lengths_tsteps[tindex], params_dict)

cdef DTYPE_t get_Utotal(DTYPE_t[:] spring_lengths, dict params_dict):
    cdef ITYPE_t EDGES = params_dict['EDGES']
    cdef ITYPE_t edgeIndex
    cdef DTYPE_t Utotal = 0.
    for edgeIndex in range(EDGES):
        Utotal += U_spring(spring_lengths[edgeIndex], params_dict)
    return Utotal


cdef get_KE_tsteps(DTYPE_t[:,:] vxilists_tsteps, DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] KE_tsteps,
                        DTYPE_t[:] KEfraction_com_tsteps, dict params_dict):
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef ITYPE_t tlength = vxilists_tsteps.shape[0]
    cdef ITYPE_t NODES = vxilists_tsteps.shape[1]
    cdef ITYPE_t tindex
    cdef ITYPE_t nodeindex
    cdef DTYPE_t vx_com
    cdef DTYPE_t vy_com
    cdef DTYPE_t m_total = NODES * MASS
    cdef DTYPE_t KE_com

    for tindex in range(tlength):
        KE_tsteps[tindex] = 0.
        vx_com = 0.
        vy_com = 0.
        for nodeindex in range(NODES):
            KE_tsteps[tindex] += 0.5*MASS*(vxilists_tsteps[tindex][nodeindex]*vxilists_tsteps[tindex][nodeindex] +
                                        vyilists_tsteps[tindex][nodeindex]*vyilists_tsteps[tindex][nodeindex])
            vx_com += (MASS*vxilists_tsteps[tindex][nodeindex])/m_total
            vy_com += (MASS*vyilists_tsteps[tindex][nodeindex])/m_total
        KE_com = 0.5*m_total*(vx_com*vx_com + vy_com*vy_com)
        KEfraction_com_tsteps[tindex] = KE_com / KE_tsteps[tindex]


cpdef get_spring_bit_flips_tsteps(DTYPE_t[:,:] spring_lengths_tsteps, dict params_dict,
                                ITYPE_t[:,:] spring_bit_flips_tsteps):
    get_spring_bits_tsteps(spring_lengths_tsteps, spring_bit_flips_tsteps, params_dict)
    cdef ITYPE_t tindex = 0
    cdef ITYPE_t tlength = spring_bit_flips_tsteps.shape[0]
    cdef ITYPE_t edgeIndex = 0
    cdef ITYPE_t EDGES = spring_bit_flips_tsteps.shape[1]
    for tindex in range(tlength-1):
        for edgeIndex in range(EDGES):
            spring_bit_flips_tsteps[tindex][edgeIndex] = int(spring_bit_flips_tsteps[tindex][edgeIndex]!=spring_bit_flips_tsteps[tindex+1][edgeIndex])

    for edgeIndex in range(EDGES):
        spring_bit_flips_tsteps[tlength-1][edgeIndex] = 0

cpdef get_spring_U_avg_T(DTYPE_t[:,:] spring_lengths_tsteps, DTYPE_t[:] spring_U_avg_T, ITYPE_t Tstart, ITYPE_t Tend, DTYPE_t deltaT, dict params_dict):
    cdef ITYPE_t tindex
    cdef ITYPE_t edgeIndex
    cdef ITYPE_t EDGES = spring_lengths_tsteps.shape[1]
    for edgeIndex in range(EDGES):
        spring_U_avg_T[edgeIndex] = U_spring(spring_lengths_tsteps[Tstart][edgeIndex], params_dict)
    tindex = Tstart+1
    while tindex < Tend:
        for edgeIndex in range(EDGES):
            spring_U_avg_T[edgeIndex] += U_spring(spring_lengths_tsteps[tindex][edgeIndex], params_dict)
        tindex += 1

    for edgeIndex in range(EDGES):
        spring_U_avg_T[edgeIndex] /= (Tend - Tstart) #deltaT

cpdef get_KE_node_avg_T(DTYPE_t[:,:] vxilists_tsteps, DTYPE_t[:,:] vyilists_tsteps, dict params_dict, DTYPE_t[:] KE_node_avg_T, ITYPE_t Tstart, ITYPE_t Tend, DTYPE_t deltaT):
    cdef ITYPE_t tindex
    cdef ITYPE_t nodeIndex
    cdef ITYPE_t NODES = vxilists_tsteps.shape[1]
    for nodeIndex in range(NODES):
        KE_node_avg_T[nodeIndex] = 0.5*params_dict['MASS']*(vxilists_tsteps[Tstart][nodeIndex]*vxilists_tsteps[Tstart][nodeIndex]+
                                        vyilists_tsteps[Tstart][nodeIndex]*vyilists_tsteps[Tstart][nodeIndex])
    tindex = Tstart+1
    while tindex < Tend:
        for nodeIndex in range(NODES):
            KE_node_avg_T[nodeIndex] += 0.5*params_dict['MASS']*(vxilists_tsteps[tindex][nodeIndex]*vxilists_tsteps[tindex][nodeIndex]+
                                        vyilists_tsteps[tindex][nodeIndex]*vyilists_tsteps[tindex][nodeIndex])
        tindex += 1
    for nodeIndex in range(NODES):
        KE_node_avg_T[nodeIndex] /= deltaT

cpdef get_spring_lengths_tsteps(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, ITYPE_t[:,:] Amat,
                                DTYPE_t[:,:] spring_lengths_tsteps):
    cdef ITYPE_t i,dummyI,dummyJ
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef ITYPE_t tindex = 0
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef ITYPE_t edgeIndex
# assuming tlength == yilists_tsteps.shape[0] and tlength == spring_lengths_tsteps.shape[0]:
    cdef np.ndarray[ITYPE_t,ndim=1] i_nonzero, j_nonzero
    (i_nonzero, j_nonzero) = np.nonzero(Amat)
    cdef ITYPE_t num_nonzero = i_nonzero.shape[0]

    for tindex in range(tlength):
        edgeIndex = 0
        for i in range(num_nonzero):
            if i_nonzero[i]>j_nonzero[i]:
                dummyI=i_nonzero[i]
                dummyJ=j_nonzero[i]
                spring_lengths_tsteps[tindex][edgeIndex] = sqrt(
                    (xilists_tsteps[tindex][dummyI] - xilists_tsteps[tindex][dummyJ])*(xilists_tsteps[tindex][dummyI] - xilists_tsteps[tindex][dummyJ]) +
                    (yilists_tsteps[tindex][dummyI] - yilists_tsteps[tindex][dummyJ])*(yilists_tsteps[tindex][dummyI] - yilists_tsteps[tindex][dummyJ]))
                edgeIndex +=1


cpdef get_spring_lengths(DTYPE_t[:] xilist, DTYPE_t[:] yilist, ITYPE_t[:,:] Amat,
                                DTYPE_t[:] spring_lengths):
    cdef ITYPE_t i,dummyI,dummyJ, edgeIndex
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef np.ndarray[ITYPE_t,ndim=1] i_nonzero, j_nonzero
    (i_nonzero, j_nonzero) = np.nonzero(Amat)
    cdef ITYPE_t num_nonzero = i_nonzero.shape[0]

    edgeIndex = 0
    for i in range(num_nonzero):
        if i_nonzero[i]>j_nonzero[i]:
            dummyI=i_nonzero[i]
            dummyJ=j_nonzero[i]
            spring_lengths[edgeIndex] = sqrt(
                    (xilist[dummyI] - xilist[dummyJ])*(xilist[dummyI] - xilist[dummyJ]) +
                    (yilist[dummyI] - yilist[dummyJ])*(yilist[dummyI] - yilist[dummyJ]) )
            edgeIndex +=1


cpdef get_spring_bits_tsteps(DTYPE_t[:,:] spring_lengths_tsteps, ITYPE_t[:,:] spring_bits_tsteps, dict params_dict):
    cdef DTYPE_t Lbarrier = params_dict['Lbarrier']
    cdef ITYPE_t tlength = spring_lengths_tsteps.shape[0]
    cdef ITYPE_t EDGES = spring_lengths_tsteps.shape[1]
    cdef ITYPE_t tindex = 0
    cdef ITYPE_t edgeIndex = 0
    for tindex in range(tlength):
        for edgeIndex in range(EDGES):
            spring_bits_tsteps[tindex][edgeIndex] = int(spring_lengths_tsteps[tindex][edgeIndex] > Lbarrier)


#get hamming distance from state at end of relaxation process
cdef get_distance_from_relaxed_state_tsteps(ITYPE_t[:,:] spring_bits_tsteps, DTYPE_t[:] tlist, dict params_dict,
                                                DTYPE_t[:] distance_from_relaxed_state_tsteps):
#    cdef str binary_state_string
    cdef ITYPE_t tlength = spring_bits_tsteps.shape[0]
    cdef ITYPE_t EDGES = params_dict['EDGES']
    cdef ITYPE_t ref_index = params_dict['RELAX_STEPS']
    cdef DTYPE_t relax_time = params_dict['T_RELAX']
    cdef np.ndarray[ITYPE_t,ndim=1] ref_state = np.empty(EDGES, dtype=ITYPE)
    cdef ITYPE_t distance_from_relaxed_state
    cdef ITYPE_t edgeIndex, i, k
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef ITYPE_t num_switches = switch_times_list.size
    i = 0

    for k in range(num_switches):
        for edgeIndex in np.arange(EDGES):
            ref_state[edgeIndex] = spring_bits_tsteps[ref_index][edgeIndex]
        while (tlist[i] < switch_times_list[k]+relax_time) and (i < tlength):
            distance_from_relaxed_state = 0
            for edgeIndex in np.arange(EDGES):
                distance_from_relaxed_state += np.int(ref_state[edgeIndex] != spring_bits_tsteps[i][edgeIndex])
            distance_from_relaxed_state_tsteps[i] = 1.*distance_from_relaxed_state
            i += 1
        ref_index = i-1

#get hamming distance from state at end of relaxation process
cpdef get_distance_between_trajectories_tsteps(ITYPE_t[:,:] spring_i_bits_tsteps, ITYPE_t[:,:] spring_j_bits_tsteps,
                                               DTYPE_t[:] distance_i_j_tsteps):
#    cdef str binary_state_string
    cdef ITYPE_t tlength = spring_i_bits_tsteps.shape[0]
    cdef ITYPE_t EDGES = spring_i_bits_tsteps.shape[1]
    cdef ITYPE_t distance_from_relaxed_state
    cdef ITYPE_t edgeIndex, tindex

    for tindex in range(tlength):
        distance_from_relaxed_state = 0
        for edgeIndex in np.arange(EDGES):
            distance_from_relaxed_state += np.int(spring_i_bits_tsteps[tindex][edgeIndex] != spring_j_bits_tsteps[tindex][edgeIndex])
        distance_i_j_tsteps[tindex] = 1.*distance_from_relaxed_state


cdef get_num_long_springs_tsteps(ITYPE_t[:,:] spring_bits_tsteps, DTYPE_t[:] num_long_springs_tsteps):
    cdef ITYPE_t tlength = spring_bits_tsteps.shape[0]
    cdef ITYPE_t EDGES = spring_bits_tsteps.shape[1]
    cdef ITYPE_t tindex
    cdef ITYPE_t edgeIndex
    cdef ITYPE_t num_long_springs
    for tindex in range(tlength):
        num_long_springs = 0
        for edgeIndex in range(EDGES):
            num_long_springs += spring_bits_tsteps[tindex][edgeIndex]
        num_long_springs_tsteps[tindex] = 1.*num_long_springs

cdef get_num_spring_transitions_tsteps(ITYPE_t[:,:] spring_bits_tsteps, DTYPE_t[:] num_spring_transitions_tsteps):
    cdef ITYPE_t tlength = spring_bits_tsteps.shape[0]
    cdef ITYPE_t EDGES = spring_bits_tsteps.shape[1]
    cdef ITYPE_t tindex
    cdef ITYPE_t edgeIndex
    cdef ITYPE_t num_spring_transitions
    num_spring_transitions_tsteps[0] = 0.
    for tindex in np.arange(1,tlength):
        num_spring_transitions = 0
        for edgeIndex in range(EDGES):
            num_spring_transitions += int(spring_bits_tsteps[tindex][edgeIndex]!=spring_bits_tsteps[tindex-1][edgeIndex])
        num_spring_transitions_tsteps[tindex] = 1.*num_spring_transitions

# assumes force direction is at 45 deg wrt x-axis and only applied to max_deg_vertex_i
cpdef get_work_rate(DTYPE_t[:,:] vxilists_tsteps, DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] f_ext_tlist,
                    DTYPE_t[:] work_rate_tlist, dict params_dict):
    cdef ITYPE_t tlength = f_ext_tlist.shape[0]
    cdef ITYPE_t nodes = params_dict['NODES']
    cdef np.ndarray[DTYPE_t, ndim=1] forcing_vec = params_dict['FORCING_VEC']
    cdef ITYPE_t tindex, nodeIndex
    for tindex in range(tlength):
        work_rate_tlist[tindex] = 0.
        for nodeIndex in range(nodes):
            work_rate_tlist[tindex] += f_ext_tlist[tindex]*(forcing_vec[2*nodeIndex]*vxilists_tsteps[tindex][nodeIndex] +
                                        forcing_vec[2*nodeIndex+1]*vyilists_tsteps[tindex][nodeIndex])

cdef get_net_EminusW_rate(DTYPE_t[:] Etotal_tsteps, DTYPE_t[:] work_rate_tlist,
                            DTYPE_t[:] diss_rate_tlist, DTYPE_t[:] net_EminusW_rate, DTYPE_t dt):
    cdef ITYPE_t tlength = work_rate_tlist.shape[0]
    cdef ITYPE_t tindex
    net_EminusW_rate[0]=0.
    for tindex in np.arange(1,tlength):
        net_EminusW_rate[tindex] = ((Etotal_tsteps[tindex] - Etotal_tsteps[tindex-1])/dt +
                                    0.5*(diss_rate_tlist[tindex]+diss_rate_tlist[tindex-1])-0.5*(work_rate_tlist[tindex]+
                                    work_rate_tlist[tindex-1]))

cdef DTYPE_t sine(DTYPE_t time, DTYPE_t amp, DTYPE_t period, DTYPE_t phase):
    cdef DTYPE_t sine_val = amp*sin((2*np.pi*time)/period + phase)
    return sine_val

cdef DTYPE_t square(DTYPE_t time, DTYPE_t amp, DTYPE_t period, DTYPE_t phase):
    cdef DTYPE_t square_val = amp*signal.square((2*np.pi*time)/period + phase)
    return square_val

cdef DTYPE_t heaviside(DTYPE_t x):
    cdef DTYPE_t heaviside_val = 0.5*(np.sign(x)+1)
    return heaviside_val

cdef DTYPE_t step(DTYPE_t t, DTYPE_t amp, DTYPE_t period, DTYPE_t phase):
    cdef DTYPE_t step_val = amp*heaviside(t-phase)
    return step_val

cdef DTYPE_t pulse(DTYPE_t t, DTYPE_t amp, DTYPE_t period, DTYPE_t phase):
    cdef DTYPE_t pulse_val = amp*( heaviside(t - phase) - heaviside(t-phase-period))
    return pulse_val

ctypedef DTYPE_t (*f_ext)(DTYPE_t t, DTYPE_t amp, DTYPE_t period, DTYPE_t phase)

cdef f_ext get_f_ext(str f_ext_string):
    if f_ext_string == 'square':
        return &square
    elif f_ext_string == 'sine':
        return &sine
    elif f_ext_string == 'step':
        return &step
    elif f_ext_string == 'pulse':
        return &pulse

cpdef get_ramp_f_ext_tlist(DTYPE_t[:] tlist, DTYPE_t[:] f_ext_tlist, dict params_dict):
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t i, k
    cdef DTYPE_t amp_slope = params_dict['FORCE_AMP_SLOPE']
    cdef DTYPE_t FORCE_PERIOD = params_dict['FORCE_PERIOD']
    cdef DTYPE_t FORCE_PHASE = params_dict['FORCE_PHASE']
    cdef DTYPE_t dt = params_dict['sim_DT']
    cdef:
        f_ext f
    f = get_f_ext('sine')

    for i in range(tlength):
        f_ext_tlist[i] = f(tlist[i], amp_slope*(i*dt), FORCE_PERIOD, FORCE_PHASE)


#gives scalar forcing not vector, vector chosen to be at 45 deg to x and y.
cpdef get_f_ext_tlist(DTYPE_t[:] tlist, DTYPE_t[:] f_ext_tlist, dict params_dict):
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t i, k
    cdef DTYPE_t FORCE_AMP = params_dict['FORCE_AMP']
    cdef DTYPE_t FORCE_PHASE = params_dict['FORCE_PHASE']
    cdef np.ndarray[DTYPE_t,ndim=1] T_list = np.array(params_dict['T_list'])
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef ITYPE_t num_periods = T_list.size
    cdef DTYPE_t relax_time = params_dict['T_RELAX']

    cdef:
        f_ext f
    f = get_f_ext(params_dict['FORCE_STRING'])

    i = 0
    for k in range(num_periods):
        while (tlist[i] < (switch_times_list[k] + relax_time)) and (i < tlength):
            f_ext_tlist[i] = f(tlist[i], FORCE_AMP, T_list[k], FORCE_PHASE)
            i += 1

cpdef get_pos_tlist(DTYPE_t[:] tlist, DTYPE_t[:] f_ext_tlist, dict params_dict):
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t i, k
    cdef DTYPE_t POS_AMP = params_dict['POS_AMP']
    cdef DTYPE_t FORCE_PHASE = params_dict['FORCE_PHASE']
    cdef np.ndarray[DTYPE_t,ndim=1] T_list = np.array(params_dict['T_list'])
    cdef np.ndarray[DTYPE_t,ndim=1] switch_times_list = np.array(params_dict['freq_switch_times'])
    cdef ITYPE_t num_periods = T_list.size
    cdef DTYPE_t relax_time = params_dict['T_RELAX']

    cdef:
        f_ext f
    f = get_f_ext(params_dict['FORCE_STRING'])

    i = 0
    for k in range(num_periods):
        while (tlist[i] < (switch_times_list[k] + relax_time)) and (i < tlength):
            f_ext_tlist[i] = f(tlist[i], POS_AMP, T_list[k], FORCE_PHASE)
            i += 1


cpdef get_phi_tlist(DTYPE_t[:] tlist, DTYPE_t[:] phi_tlist, dict params_dict):
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t i, j
    cdef DTYPE_t FORCE_PERIOD = params_dict['FORCE_PERIOD']
    cdef DTYPE_t num_cycles_per_phi = params_dict['num_cycles_per_phi']
    cdef np.ndarray[DTYPE_t,ndim=1] phi_values = np.array(params_dict['phi_values_list'])
    cdef DTYPE_t DT = params_dict['sim_DT']
    cdef ITYPE_t phi_period_steps = int(FORCE_PERIOD*num_cycles_per_phi/DT)
    cdef ITYPE_t num_phi_periods = int((tlength*1.)/(phi_period_steps*1.))
    cdef ITYPE_t num_phi_values = phi_values.shape[0]
    cdef DTYPE_t phi

    for i in range(num_phi_periods):
#        phi = 2*np.pi*(rand()/(RAND_MAX + 1.0))
        for j in range(phi_period_steps):
            phi_tlist[i*phi_period_steps + j] = phi_values[i%num_phi_values]

    i = num_phi_periods*phi_period_steps
    phi = phi_values[num_phi_periods%num_phi_values]
#    phi = 2*np.pi*(rand()/(RAND_MAX + 1.0))
    while i < tlength:
        phi_tlist[i] = phi
        i+=1

cpdef get_phi_tlist_for_vid(DTYPE_t[:] tlist, DTYPE_t[:] phi_tlist, dict params_dict):
    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t i, j
    cdef DTYPE_t FORCE_PERIOD = params_dict['FORCE_PERIOD']
    cdef DTYPE_t num_cycles_per_phi = params_dict['num_cycles_per_phi']
    cdef np.ndarray[DTYPE_t,ndim=1] phi_values = np.array(params_dict['phi_values_list'])
    cdef DTYPE_t DT = params_dict['sim_DT']*params_dict['PLOT_DELTA_STEPS']*params_dict['SIM_STEPS_PER_DT']
    cdef ITYPE_t phi_period_steps = int(FORCE_PERIOD*num_cycles_per_phi/DT)
    cdef ITYPE_t num_phi_periods = int((tlength*1.)/(phi_period_steps*1.))
    cdef ITYPE_t num_phi_values = phi_values.shape[0]
    cdef DTYPE_t phi

    for i in range(num_phi_periods):
        for j in range(phi_period_steps):
            phi_tlist[i*phi_period_steps + j] = phi_values[i%num_phi_values]

    i = num_phi_periods*phi_period_steps
    phi = phi_values[num_phi_periods%num_phi_values]
    while i < tlength:
        phi_tlist[i] = phi
        i+=1


cdef force_defn_bistable_spring(DTYPE_t dij, dict params_dict):
    cdef DTYPE_t SPRING_K = params_dict['SPRING_K']
    if dij < 4./5:
        return SPRING_K*( -(25.*dij-20.))
    elif dij < 6./5:
        return SPRING_K *(-(0.5*dij-0.5))
    elif dij < 14./5:
        return SPRING_K*(-(1.77002*(4*dij*dij*dij -24.6703*dij*dij + 47.733*dij -28.6098)))
    elif dij < 16./5:
        return SPRING_K*(-(5.*dij-15.))
    else:
        return SPRING_K*(-(20.*dij-63.))


cdef aix_network(DTYPE_t[:] xilist, DTYPE_t[:] yilist, ITYPE_t[:,:] Amat, DTYPE_t[:] aix, dict params_dict):
    #give the ax due to springs (not include the constrain)
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef ITYPE_t node
    cdef ITYPE_t dummyNode
    cdef DTYPE_t dij
    cdef DTYPE_t ERR = params_dict['ERR']
    for node in range(NODES):
        if Amat[node,0] != 0:
            dij = max(sqrt((xilist[node]-xilist[0])*(xilist[node]-xilist[0]) + (yilist[node]-yilist[0])*(yilist[node]-yilist[0])),
                        ERR)
            aix[node]= Amat[node,0]*force_defn_bistable_spring(dij, params_dict)*((xilist[node]-xilist[0])/dij)/MASS
        else:
            aix[node] = 0.
        for dummyNode in xrange(1,NODES):
            if Amat[node,dummyNode] != 0:
                dij = max(sqrt((xilist[node]-xilist[dummyNode])*(xilist[node]-xilist[dummyNode]) +
                            (yilist[node]-yilist[dummyNode])*(yilist[node]-yilist[dummyNode])), ERR)
                aix[node] += force_defn_bistable_spring(dij, params_dict)*((xilist[node]-xilist[dummyNode])/dij)/MASS

cdef aiy_network(DTYPE_t[:] xilist, DTYPE_t[:] yilist, ITYPE_t[:,:] Amat, DTYPE_t[:] aiy, dict params_dict):
    #give the ay due to springs (not include the constrain)
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef int node
    cdef int dummyNode
    cdef DTYPE_t dij
    cdef DTYPE_t ERR = params_dict['ERR']
    for node in range(NODES):
        if Amat[node,0] != 0:
            dij = max(sqrt((xilist[node]-xilist[0])*(xilist[node]-xilist[0]) + (yilist[node]-yilist[0])*(yilist[node]-yilist[0])),
                        ERR)
            aiy[node]= Amat[node,0]*force_defn_bistable_spring(dij, params_dict)*((yilist[node]-yilist[0])/dij)/MASS
        else:
            aiy[node] = 0.
        for dummyNode in xrange(1,NODES):
            if Amat[node,dummyNode] != 0:
                dij = max(sqrt((xilist[node]-xilist[dummyNode])*(xilist[node]-xilist[dummyNode]) +
                            (yilist[node]-yilist[dummyNode])*(yilist[node]-yilist[dummyNode])), ERR)
                aiy[node] += Amat[node,dummyNode]*force_defn_bistable_spring(dij, params_dict)*((yilist[node]-yilist[dummyNode])/dij)/MASS



#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
#beta_drive_tlist are of size nsteps*sim_steps_per_dt
cpdef temperature_drive_network_with_noise(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] betaInv_drive_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    temperature_drive_network_with_noise_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, betaInv_drive_tlist,
                    wdot_tsteps, qdot_tsteps, params_dict)

cdef temperature_drive_network_with_noise_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] betaInv_drive_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef DTYPE_t drive_variance, wdot, qdot
    cdef ITYPE_t i, k
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t dummyNode

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = params_dict['ANCHORED_NODES']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] drive_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] drive_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    wdot_tsteps[0] = 0.
    qdot_tsteps[0] = 0.
    for i in range(tlength-1):
        wdot = 0.
        qdot = 0.
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi, yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)
            drive_variance = sqrt(2*DAMPING_RATE*(betaInv_drive_tlist[i*sim_steps_per_dt+k])*DT)
            drive_kick_x = np.random.normal(0.,drive_variance, NODES)
            drive_kick_y = np.random.normal(0.,drive_variance, NODES)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS +
                                    0.5*b_damping_factor*DT*(drive_kick_x[dummyNode]/MASS))
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS +
                                    0.5*b_damping_factor*DT*drive_kick_y[dummyNode]/MASS)

            for dummyNode in range(num_anchors):
                xiplus[anchored_nodes[dummyNode]] = xi0[dummyNode]
                yiplus[anchored_nodes[dummyNode]] = yi0[dummyNode]

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + (thermal_kick_x[dummyNode] +
                                                    drive_kick_x[dummyNode])/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + (thermal_kick_y[dummyNode] +
                                                    drive_kick_y[dummyNode])/MASS

            for dummyNode in range(num_anchors):
                vxiplus[anchored_nodes[dummyNode]] = 0.
                vyiplus[anchored_nodes[dummyNode]] = 0.

            for dummyNode in range(NODES):
                wdot += 0.5*(drive_kick_x[dummyNode]*(vxi[dummyNode]+vxiplus[dummyNode])+
                            drive_kick_y[dummyNode]*(vyi[dummyNode]+vyiplus[dummyNode]))/DT
                qdot += 0.5*(vxiplus[dummyNode]+vxi[dummyNode])*(DAMPING_RATE*(xiplus[dummyNode]-xi[dummyNode])-
                                        thermal_kick_x[dummyNode])/DT
                qdot += 0.5*(vyiplus[dummyNode]+vyi[dummyNode])*(DAMPING_RATE*(yiplus[dummyNode]-yi[dummyNode])-
                                        thermal_kick_y[dummyNode])/DT
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]
            wdot_tsteps[i+1] = wdot/sim_steps_per_dt
            qdot_tsteps[i+1] = qdot/sim_steps_per_dt


#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
#beta_drive_tlist are of size nsteps*sim_steps_per_dt
cpdef temperature_drive_node_with_noise(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] betaInv_drive_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    temperature_drive_node_with_noise_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, betaInv_drive_tlist,
                    wdot_tsteps, qdot_tsteps, params_dict)

cdef temperature_drive_node_with_noise_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] betaInv_drive_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef DTYPE_t drive_variance, drive_kick_x, drive_kick_y, wdot, qdot
    cdef ITYPE_t i, k
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t dummyNode

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = params_dict['ANCHORED_NODES']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    wdot_tsteps[0] = 0.
    qdot_tsteps[0] = 0.
    for i in range(tlength-1):
        wdot = 0.
        qdot = 0.
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi, yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)
            drive_variance = sqrt(2*DAMPING_RATE*(betaInv_drive_tlist[i*sim_steps_per_dt+k])*DT)
            drive_kick_x = np.random.normal(0.,drive_variance)
            drive_kick_y = np.random.normal(0.,drive_variance)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS)
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS)

            #external force for max_deg node
            xiplus[max_deg_vertex_i] += 0.5*b_damping_factor*DT*(drive_kick_x/MASS)
            yiplus[max_deg_vertex_i] += 0.5*b_damping_factor*DT*(drive_kick_y/MASS)

            for dummyNode in range(num_anchors):
                xiplus[anchored_nodes[dummyNode]] = xi0[dummyNode]
                yiplus[anchored_nodes[dummyNode]] = yi0[dummyNode]

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + thermal_kick_x[dummyNode]/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + thermal_kick_y[dummyNode]/MASS
            #velocities for t+1
            vxiplus[max_deg_vertex_i] += (drive_kick_x/MASS)
            vyiplus[max_deg_vertex_i] += (drive_kick_y/MASS)

            for dummyNode in range(num_anchors):
                vxiplus[anchored_nodes[dummyNode]] = 0.
                vyiplus[anchored_nodes[dummyNode]] = 0.

            wdot += 0.5*(drive_kick_x*(vxi[max_deg_vertex_i]+vxiplus[max_deg_vertex_i])+
                        drive_kick_y*(vyi[max_deg_vertex_i]+vyiplus[max_deg_vertex_i]))/DT

            for dummyNode in range(NODES):
                qdot += 0.5*(vxiplus[dummyNode]+vxi[dummyNode])*(DAMPING_RATE*(xiplus[dummyNode]-xi[dummyNode])-
                                        thermal_kick_x[dummyNode])/DT
                qdot += 0.5*(vyiplus[dummyNode]+vyi[dummyNode])*(DAMPING_RATE*(yiplus[dummyNode]-yi[dummyNode])-
                                        thermal_kick_y[dummyNode])/DT
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]
            wdot_tsteps[i] = wdot/sim_steps_per_dt
            qdot_tsteps[i] = qdot/sim_steps_per_dt


#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
cpdef undriven_relaxation(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, dict params_dict):
    undriven_relaxation_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, params_dict)

cdef undriven_relaxation_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
#    cdef DTYPE_t phi = params_dict['FORCE_DIRN']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef ITYPE_t i, k
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t dummyNode
    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for i in range(tlength-1):
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi, yi,Amat,ayi, params_dict)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode])
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode])

            #external force for max_deg node
            xiplus[max_deg_vertex_i] -= 0.5*b_damping_factor*(DT*DT)*SPRING_K0*xi[max_deg_vertex_i]/MASS
            yiplus[max_deg_vertex_i] -= 0.5*b_damping_factor*(DT*DT)*SPRING_K0*yi[max_deg_vertex_i]/MASS

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS
            #velocities for t+1
            vxiplus[max_deg_vertex_i] -= 0.5*DT*(SPRING_K0*xiplus[max_deg_vertex_i]/MASS + SPRING_K0*xi[max_deg_vertex_i]/MASS)
            vyiplus[max_deg_vertex_i] -= 0.5*DT*(SPRING_K0*yiplus[max_deg_vertex_i]/MASS + SPRING_K0*yi[max_deg_vertex_i]/MASS)

            for dummyNode in range(NODES):
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]

#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
cpdef undriven_relaxation_with_noise(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, dict params_dict):
    undriven_relaxation_with_noise_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, params_dict)

cdef undriven_relaxation_with_noise_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = params_dict['NODES']
    cdef DTYPE_t BETA = params_dict['BETA']
#    cdef DTYPE_t phi = params_dict['FORCE_DIRN']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef ITYPE_t i, k
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef ITYPE_t dummyNode
    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for i in range(tlength-1):
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi,yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)

            # axi, ayi divide by mass already
            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS)
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS)

            #external force for max_deg node
            xiplus[max_deg_vertex_i] -= 0.5*b_damping_factor*(DT*DT)*(SPRING_K0*xi[max_deg_vertex_i]/MASS)
            yiplus[max_deg_vertex_i] -= 0.5*b_damping_factor*(DT*DT)*(SPRING_K0*yi[max_deg_vertex_i]/MASS)

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + thermal_kick_x[dummyNode]/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + thermal_kick_y[dummyNode]/MASS
            #velocities for t+1
            vxiplus[max_deg_vertex_i] -= 0.5*DT*(SPRING_K0*xiplus[max_deg_vertex_i]/MASS + SPRING_K0*xi[max_deg_vertex_i]/MASS)
            vyiplus[max_deg_vertex_i] -= 0.5*DT*(SPRING_K0*yiplus[max_deg_vertex_i]/MASS + SPRING_K0*yi[max_deg_vertex_i]/MASS)

            for dummyNode in range(NODES):
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]


#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
#f_ext_tlist, phi_tlist are of size nsteps*sim_steps_per_dt
cpdef verlet_iterate(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] f_ext_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    verlet_iterate_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, f_ext_tlist, wdot_tsteps,
                        qdot_tsteps, params_dict)

cdef verlet_iterate_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] f_ext_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):

    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef ITYPE_t i, k, dummyNode
    cdef DTYPE_t a_ext, wdot, qdot

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef np.ndarray[DTYPE_t, ndim=1] forcing_vec = params_dict['FORCING_VEC']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    wdot_tsteps[0] = 0.
    qdot_tsteps[0] = 0.
    for i in range(tlength-1):
        wdot = 0.
        qdot = 0.
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi, yi,Amat,ayi, params_dict)

            a_ext = (f_ext_tlist[i*sim_steps_per_dt+k]/MASS)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*a_ext*forcing_vec[2*dummyNode])
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*a_ext*forcing_vec[2*dummyNode+1])

            for dummyNode in range(num_anchors):
                xiplus[anchored_nodes[dummyNode]] = xi0[dummyNode]
                yiplus[anchored_nodes[dummyNode]] = yi0[dummyNode]

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + DT*a_ext*forcing_vec[2*dummyNode]
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + DT*a_ext*forcing_vec[2*dummyNode+1]

            for dummyNode in range(num_anchors):
                vxiplus[anchored_nodes[dummyNode]] = 0.
                vyiplus[anchored_nodes[dummyNode]] = 0.

            for dummyNode in range(NODES):
                wdot += 0.5*MASS*a_ext*(forcing_vec[2*dummyNode]*(vxi[dummyNode]+vxiplus[dummyNode])+
                                    forcing_vec[2*dummyNode+1]*(vyi[dummyNode]+vyiplus[dummyNode]))
                qdot += 0.5*(vxiplus[dummyNode]+vxi[dummyNode])*DAMPING_RATE*(xiplus[dummyNode]-xi[dummyNode])/DT
                qdot += 0.5*(vyiplus[dummyNode]+vyi[dummyNode])*DAMPING_RATE*(yiplus[dummyNode]-yi[dummyNode])/DT
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]
            wdot_tsteps[i+1] = wdot/sim_steps_per_dt
            qdot_tsteps[i+1] = qdot/sim_steps_per_dt


#yi, vxi, vyi, xilists_tsteps, are of size nsteps=(t_end - t_start)/(sim_dt*sim_steps_per_dt)
#f_ext_tlist, phi_tlist are of size nsteps*sim_steps_per_dt
cpdef verlet_iterate_with_noise(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] f_ext_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    verlet_iterate_with_noise_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, f_ext_tlist,
                                wdot_tsteps, qdot_tsteps, params_dict)

cdef verlet_iterate_with_noise_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] f_ext_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
#    cdef DTYPE_t phi = params_dict['FORCE_DIRN']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef ITYPE_t i, k, dummyNode
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef DTYPE_t a_ext, wdot, qdot

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = np.array(params_dict['ANCHORED_NODES'])
    cdef np.ndarray[DTYPE_t, ndim=1] forcing_vec = params_dict['FORCING_VEC']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    wdot_tsteps[0] = 0.
    qdot_tsteps[0] = 0.
    for i in range(tlength-1):
        wdot = 0.
        qdot = 0.
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi,yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)

            a_ext = (f_ext_tlist[i*sim_steps_per_dt+k]/MASS)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*a_ext*forcing_vec[2*dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS)
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*a_ext*forcing_vec[2*dummyNode+1] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS)

            #external force for max_deg node
#            xiplus[max_deg_vertex_i] += 0.5*b_damping_factor*(DT*DT)*(f_ext_tlist[i*sim_steps_per_dt+k]/MASS)*cos(phi_tlist[i*sim_steps_per_dt+k])
#            yiplus[max_deg_vertex_i] += 0.5*b_damping_factor*(DT*DT)*(f_ext_tlist[i*sim_steps_per_dt+k]/MASS)*sin(phi_tlist[i*sim_steps_per_dt+k])

            for dummyNode in range(num_anchors):
                xiplus[anchored_nodes[dummyNode]] = xi0[dummyNode]
                yiplus[anchored_nodes[dummyNode]] = yi0[dummyNode]

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + (thermal_kick_x[dummyNode]/MASS +
                                                    DT*a_ext*forcing_vec[2*dummyNode])
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + (thermal_kick_y[dummyNode]/MASS +
                                                    DT*a_ext*forcing_vec[2*dummyNode+1])

            #velocities for t+1
#            vxiplus[max_deg_vertex_i] += 0.5*DT*2*(f_ext_tlist[i*sim_steps_per_dt+k]/MASS)*cos(phi_tlist[i*sim_steps_per_dt+k])
#            vyiplus[max_deg_vertex_i] += 0.5*DT*2*(f_ext_tlist[i*sim_steps_per_dt+k]/MASS)*sin(phi_tlist[i*sim_steps_per_dt+k])

            for dummyNode in range(num_anchors):
                vxiplus[anchored_nodes[dummyNode]] = 0.
                vyiplus[anchored_nodes[dummyNode]] = 0.

            for dummyNode in range(NODES):
                wdot += 0.5*MASS*a_ext*(forcing_vec[2*dummyNode]*(vxi[dummyNode]+vxiplus[dummyNode])+
                                    forcing_vec[2*dummyNode+1]*(vyi[dummyNode]+vyiplus[dummyNode]))
                qdot += 0.5*(vxiplus[dummyNode]+vxi[dummyNode])*(DAMPING_RATE*(xiplus[dummyNode]-xi[dummyNode])-
                                        thermal_kick_x[dummyNode])/DT
                qdot += 0.5*(vyiplus[dummyNode]+vyi[dummyNode])*(DAMPING_RATE*(yiplus[dummyNode]-yi[dummyNode])-
                                        thermal_kick_y[dummyNode])/DT
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]
            wdot_tsteps[i+1] = wdot/sim_steps_per_dt
            qdot_tsteps[i+1] = qdot/sim_steps_per_dt


cpdef verlet_pos_drive_with_noise_unanchored(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] pos_drive_tlist, DTYPE_t[:] phi_tlist, dict params_dict):
    verlet_pos_drive_with_noise_unanchored_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps, pos_drive_tlist, phi_tlist, params_dict)

cdef verlet_pos_drive_with_noise_unanchored_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] pos_drive_tlist, DTYPE_t[:] phi_tlist, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
#    cdef DTYPE_t phi = params_dict['FORCE_DIRN']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef ITYPE_t i, k, dummyNode
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef DTYPE_t x_driven_0, y_driven_0

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = params_dict['ANCHORED_NODES']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    x_driven_0 = xilists_tsteps[0][max_deg_vertex_i]
    y_driven_0 = yilists_tsteps[0][max_deg_vertex_i]

    for i in range(tlength-1):
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi,yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS)
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS)

            #overwrite posn for max_deg node
            xiplus[max_deg_vertex_i] = x_driven_0 + pos_drive_tlist[i*sim_steps_per_dt+k]*cos(phi_tlist[i*sim_steps_per_dt+k])
            yiplus[max_deg_vertex_i] = y_driven_0 + pos_drive_tlist[i*sim_steps_per_dt+k]*sin(phi_tlist[i*sim_steps_per_dt+k])

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + thermal_kick_x[dummyNode]/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + thermal_kick_y[dummyNode]/MASS
            #overwrite velocities for t+1
            vxiplus[max_deg_vertex_i] = (xiplus[max_deg_vertex_i] - xi[max_deg_vertex_i])/DT
            vyiplus[max_deg_vertex_i] = (yiplus[max_deg_vertex_i] - yi[max_deg_vertex_i])/DT

            for dummyNode in range(NODES):
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]


cpdef verlet_pos_drive_with_noise_anchored(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] pos_drive_tlist, DTYPE_t[:] phi_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    verlet_pos_drive_with_noise_anchored_c(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                    pos_drive_tlist, phi_tlist, wdot_tsteps, qdot_tsteps, params_dict)

cdef verlet_pos_drive_with_noise_anchored_c(DTYPE_t[:,:] xilists_tsteps, DTYPE_t[:,:] yilists_tsteps, DTYPE_t[:,:] vxilists_tsteps,
                    DTYPE_t[:,:] vyilists_tsteps, DTYPE_t[:] pos_drive_tlist, DTYPE_t[:] phi_tlist, DTYPE_t[:] wdot_tsteps,
                    DTYPE_t[:] qdot_tsteps, dict params_dict):
    ##### changes from larger memory sims ######
    cdef ITYPE_t tlength = xilists_tsteps.shape[0]
    cdef DTYPE_t DT = params_dict['sim_DT']#tlist[1] - tlist[0]
    cdef ITYPE_t sim_steps_per_dt = params_dict['SIM_STEPS_PER_DT']
    ##########################################

    cdef np.ndarray[ITYPE_t,ndim=2] Amat = params_dict['AMAT']
    cdef ITYPE_t max_deg_vertex_i = params_dict['MAX_DEG_VERTEX_I']
    cdef DTYPE_t MASS = params_dict['MASS']
    cdef DTYPE_t DAMPING_RATE = params_dict['DAMPING_RATE']
    cdef DTYPE_t SPRING_K0 = params_dict['SPRING_K0']
#    cdef DTYPE_t DT = tlist[1] - tlist[0]
    cdef ITYPE_t NODES = Amat.shape[0]
    cdef DTYPE_t BETA = params_dict['BETA']
#    cdef DTYPE_t phi = params_dict['FORCE_DIRN']
    cdef DTYPE_t b_damping_factor = 1./(1.+((DAMPING_RATE*DT)/(2.*MASS)))
    cdef DTYPE_t thermal_variance = sqrt(2*DAMPING_RATE*(1./BETA)*DT)
    cdef ITYPE_t i, k, dummyNode
#    cdef ITYPE_t tlength = tlist.shape[0]
    cdef DTYPE_t x_driven_0, y_driven_0, wdot, qdot, ax_driven, ay_driven, ax_network, ay_network

    cdef np.ndarray[ITYPE_t, ndim=1] anchored_nodes = params_dict['ANCHORED_NODES']
    cdef ITYPE_t num_anchors = anchored_nodes.size
    cdef np.ndarray[DTYPE_t, ndim=1] xi0 = np.empty(num_anchors, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yi0 = np.empty(num_anchors, dtype=DTYPE)

    #acceleration here is just due to the network (postions)
    cdef np.ndarray[DTYPE_t,ndim=1] ayi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axi = np.empty(NODES, dtype=DTYPE)
    #a at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] ayiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] axiplus = np.empty(NODES, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_x = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thermal_kick_y = np.empty(NODES, dtype=DTYPE)

    #x,y,vx,vy at time t
    cdef np.ndarray[DTYPE_t,ndim=1] yi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyi = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxi = np.empty(NODES, dtype=DTYPE)
    #x,y,vx,vy at time t+1
    cdef np.ndarray[DTYPE_t,ndim=1] yiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vyiplus = np.empty(NODES, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] vxiplus = np.empty(NODES, dtype=DTYPE)

    for dummyNode in range(NODES):
        xi[dummyNode] = xilists_tsteps[0][dummyNode]
        vxi[dummyNode] = vxilists_tsteps[0][dummyNode]
        yi[dummyNode] = yilists_tsteps[0][dummyNode]
        vyi[dummyNode] = vyilists_tsteps[0][dummyNode]

    for dummyNode in range(num_anchors):
        xi0[dummyNode] = xilists_tsteps[0][anchored_nodes[dummyNode]]
        yi0[dummyNode] = yilists_tsteps[0][anchored_nodes[dummyNode]]

    x_driven_0 = xilists_tsteps[0][max_deg_vertex_i]
    y_driven_0 = yilists_tsteps[0][max_deg_vertex_i]

    wdot_tsteps[0] = 0.
    qdot_tsteps[0] = 0.
    for i in range(tlength-1):
        wdot = 0.
        qdot = 0.
        for k in range(sim_steps_per_dt):
            aix_network(xi,yi,Amat,axi, params_dict)
            aiy_network(xi,yi,Amat,ayi, params_dict)
            thermal_kick_x = np.random.normal(0.,thermal_variance,NODES)
            thermal_kick_y = np.random.normal(0.,thermal_variance,NODES)

            for dummyNode in range(NODES):
                xiplus[dummyNode] = (xi[dummyNode] + b_damping_factor*DT*vxi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*axi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_x[dummyNode]/MASS)
                yiplus[dummyNode] = (yi[dummyNode] + b_damping_factor*DT*vyi[dummyNode] +
                                    0.5*b_damping_factor*(DT*DT)*ayi[dummyNode] +
                                    0.5*b_damping_factor*DT*thermal_kick_y[dummyNode]/MASS)

            #overwrite posn for max_deg node
            xiplus[max_deg_vertex_i] = x_driven_0 + pos_drive_tlist[i*sim_steps_per_dt+k]*cos(phi_tlist[i*sim_steps_per_dt+k])
            yiplus[max_deg_vertex_i] = y_driven_0 + pos_drive_tlist[i*sim_steps_per_dt+k]*sin(phi_tlist[i*sim_steps_per_dt+k])

            for dummyNode in range(num_anchors):
                xiplus[anchored_nodes[dummyNode]] = xi0[dummyNode]
                yiplus[anchored_nodes[dummyNode]] = yi0[dummyNode]

            aix_network(xiplus,yiplus,Amat,axiplus,params_dict)
            aiy_network(xiplus,yiplus,Amat,ayiplus,params_dict)

            for dummyNode in range(NODES):
                vxiplus[dummyNode] = vxi[dummyNode] + 0.5*DT*(axiplus[dummyNode] +
                                                    axi[dummyNode]) - DAMPING_RATE*(xiplus[dummyNode] -
                                                    xi[dummyNode])/MASS + thermal_kick_x[dummyNode]/MASS
                vyiplus[dummyNode] = vyi[dummyNode] + 0.5*DT*(ayiplus[dummyNode] +
                                                    ayi[dummyNode]) - DAMPING_RATE*(yiplus[dummyNode] -
                                                    yi[dummyNode])/MASS + thermal_kick_y[dummyNode]/MASS
            #overwrite velocities for t+1
            vxiplus[max_deg_vertex_i] = (xiplus[max_deg_vertex_i] - xi[max_deg_vertex_i])/DT
            vyiplus[max_deg_vertex_i] = (yiplus[max_deg_vertex_i] - yi[max_deg_vertex_i])/DT

            for dummyNode in range(num_anchors):
                vxiplus[anchored_nodes[dummyNode]] = 0.
                vyiplus[anchored_nodes[dummyNode]] = 0.

            ax_driven = (vxiplus[max_deg_vertex_i] - vxi[max_deg_vertex_i])/DT
            ax_network = 0.5*(axi[max_deg_vertex_i] + axiplus[max_deg_vertex_i])
            fx_drive = MASS*(ax_driven - ax_network)
            ay_driven = (vyiplus[max_deg_vertex_i] - vyi[max_deg_vertex_i])/DT
            ay_network = 0.5*(ayi[max_deg_vertex_i] + ayiplus[max_deg_vertex_i])
            fy_drive = MASS*(ay_driven - ay_network)

            wdot += 0.5*(fx_drive*(vxi[max_deg_vertex_i]+vxiplus[max_deg_vertex_i])+
                        fy_drive*(vyi[max_deg_vertex_i]+vyiplus[max_deg_vertex_i]))

            for dummyNode in range(NODES):

                qdot += 0.5*DAMPING_RATE*((vxiplus[dummyNode]+vxi[dummyNode])*(xiplus[dummyNode]-xi[dummyNode]) +
                             (vyiplus[dummyNode]+vyi[dummyNode])*(yiplus[dummyNode]-yi[dummyNode]))/DT
                qdot -= 0.5*DAMPING_RATE*((vxiplus[dummyNode]+vxi[dummyNode])*thermal_kick_x[dummyNode] +
                                            (vyiplus[dummyNode]+vyi[dummyNode])*thermal_kick_y[dummyNode])/DT
                xi[dummyNode] = xiplus[dummyNode]
                vxi[dummyNode] = vxiplus[dummyNode]
                yi[dummyNode] = yiplus[dummyNode]
                vyi[dummyNode] = vyiplus[dummyNode]

        for dummyNode in range(NODES):
            xilists_tsteps[i+1][dummyNode] = xiplus[dummyNode]
            vxilists_tsteps[i+1][dummyNode] = vxiplus[dummyNode]
            yilists_tsteps[i+1][dummyNode] = yiplus[dummyNode]
            vyilists_tsteps[i+1][dummyNode] = vyiplus[dummyNode]
            wdot_tsteps[i+1] = wdot/sim_steps_per_dt
            qdot_tsteps[i+1] = qdot/sim_steps_per_dt