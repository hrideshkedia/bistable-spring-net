# coding=utf-8
import numpy as np
import networkx as nx
import os, random, json, datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import stats
from bistable_spring_net_fns import *
import io, cv2
from PIL import Image
from networkx.readwrite import json_graph
import multiprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_params = {
    'axes.linewidth': 0.5,
    # 'axes.labelsize': 14,
    'legend.frameon':False,
    'legend.markerscale':1,
    'legend.columnspacing':0.5,
    # 'legend.labelspacing':0.5,
    'text.usetex': True,
    'text.latex.preamble': [r'\usepackage{amsmath}'],
    'font.family': 'sans-serif',
    'font.sans-serif': ['Tahoma'],
    'mathtext.fontset': 'stixsans',
    'lines.markeredgewidth': 2.,
     # 'xtick.major.size': 2,
     # 'xtick.minor.size': 1,
     # 'ytick.major.size': 2,
     # 'ytick.minor.size': 1,
     'lines.linewidth' : 0.5
}
plt.rc(plot_params)
plt.rcParams['agg.path.chunksize'] = 10000
#  plt.rcParams.update({'font.size': 16})
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)

### convert to avi ###
# FFMPEG_BIN = "ffmpeg"
# FRAME_RATE = 15
# QUALITY = 95
# FFMPEG_BIN = "ffmpeg"
## cmd i/p: ffmpeg -r 15 -pattern_type glob -f image2 -s 480x480 -i 'n7_m15_equal_E_diff_k*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
# fig = Figure()
# canvas = FigureCanvas(fig)
# ax = fig.gca()
# ax.text(0.0,0.0,"Test", fontsize=45)
# ax.axis('off')
# canvas.draw()       # draw the canvas, cache the renderer
# image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

def get_data_dirs(pathname, dirstrings):
    data_dirs = map(lambda data_dir: os.path.join(pathname, data_dir),
                        sorted(os.listdir(pathname)))
    data_dirs = filter(lambda data_dir: os.path.isdir(data_dir), data_dirs)
    for dirstring in dirstrings:
        data_dirs = filter(lambda data_dir: dirstring in os.path.basename(data_dir), data_dirs)

    return data_dirs


#assume knowledge of keys in plot_data_dict which is defined in *.pyx:
def save_sim_data(plot_data_dict, sampled_xilist, sampled_yilist, sampled_vxilist, sampled_vyilist, sampled_tlist,
                sampled_f_extlist, params_dict, ergraph, tol=1e-8, delta_phi=np.pi/4): #sampled_philist,
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_xilist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_xilist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_yilist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_yilist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_vxilist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_vxilist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_vyilist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_vyilist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_tlist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_tlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_f_extlist.json')
    with open(dfn, 'w') as f:
        json.dump(sampled_f_extlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    # dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_philist.json')
    # with open(dfn, 'w') as f:
    #     json.dump(sampled_philist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    #for reconstructing the graph
    adj_data = json_graph.adjacency_data(ergraph)
    dfn = os.path.join(params_dict['PLOT_DIR'], 'adjacency_data.json')
    with open(dfn, 'w') as f:
        json.dump(adj_data, f)

    #spectrum info
    eigvals_tsteps = np.empty([sampled_xilist.shape[0], 2*params_dict['NODES']])
    forcing_overlap_tsteps = np.empty([2*params_dict['NODES'], sampled_xilist.shape[0]])
    forcing_direction = np.zeros([2*params_dict['NODES'],sampled_xilist.shape[0]])
    forcing_vec = params_dict['FORCING_VEC']

    for i in range(2*params_dict['NODES']):
        forcing_direction[i,:] = params_dict['FORCING_VEC'][i]

    params_dict['omega'] = params_dict['SPRING_K']*7.
    params_dict['omega_spacing'] = 0.05
    params_dict['omega_plot'] = np.linspace(params_dict['omega_spacing'], params_dict['omega'],
                                            int(params_dict['omega']/params_dict['omega_spacing']))
    #
    drive_start_index = int(params_dict['T_RELAX']/(2*params_dict['DT']*params_dict['PLOT_DELTA_STEPS']))
    #
    get_normal_mode_greens_data(sampled_xilist, sampled_yilist, params_dict['FORCE_PERIOD']*sampled_tlist,
        forcing_direction,
        params_dict, np.array(params_dict['AMAT']), eigvals_tsteps,
        forcing_overlap_tsteps)
    #
    #save spectrum data at the start and the end for comparison
    plot_data_dict['eigvals_drive_start'] = eigvals_tsteps[drive_start_index,:]
    plot_data_dict['forcing_overlap_drive_start'] = forcing_overlap_tsteps[:,drive_start_index]
    plot_data_dict['eigvals_drive_end'] = eigvals_tsteps[-1,:]
    plot_data_dict['forcing_overlap_drive_end'] = forcing_overlap_tsteps[:,-1]

    max_eigval = params_dict['omega']#np.amax(eigvals_tsteps)
    min_eigval = 1e-6 #np.amin(eigvals_tsteps)
    numbins = 20 #int(np.sqrt(2*params_dict['NODES']))
    eig_bins = np.linspace(min_eigval, max_eigval+1e-6,num=numbins)
    #
    #eigenvectors in position space/ dot product with Force in position space
    eig_hist_tsteps = np.empty([eig_bins.shape[0]-1, sampled_xilist.shape[0]],dtype=int)
    binned_forcing_overlap_tsteps = np.empty([eig_hist_tsteps.shape[0], eig_hist_tsteps.shape[1]])
    get_hist_tsteps(eigvals_tsteps, eig_bins, eig_hist_tsteps)
    get_binned_mode_overlap_tsteps(forcing_overlap_tsteps, eig_hist_tsteps, binned_forcing_overlap_tsteps)

    num_omegas = params_dict['omega_plot'].size
    omegas_per_bin = 8
    num_bins = int(num_omegas/omegas_per_bin)
    omega_bins = np.linspace(params_dict['omega_spacing'], params_dict['omega'], num_bins+1)
    #
    plot_data_dict['omega_bins'] = omega_bins
    #
    plot_data_dict['eig_hist_tsteps'] = eig_hist_tsteps
    plot_data_dict['eig_bins'] = eig_bins
    #
    plot_data_dict['forcing_overlap_tsteps'] = binned_forcing_overlap_tsteps
    params_dict['cmap'] = 'YlOrBr'

    dfn = os.path.join(params_dict['PLOT_DIR'], 'params_dict.json')
    with open(dfn, 'w') as f:
        json.dump(params_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    for k,v in plot_data_dict.iteritems():
        dfn = os.path.join(params_dict['PLOT_DIR'], str(k).replace('/','_by_')+'.json')
        with open(dfn, 'w') as f:
            json.dump(v, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))


def read_sim_data_plot_make_vid(PLOT_DIR, t_start, t_end, fps):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn,'r') as f:
        params_dict = json.load(f)
    params_dict['PLOT_DIR'] = PLOT_DIR
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn,'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn,'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    with open(dfn,'r') as f:
        plot_tlist = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_f_extlist.json')
    with open(dfn,'r') as f:
        plot_f_ext_tlist = np.array(json.load(f))

    plot_phi_tlist = np.zeros(plot_tlist.size)
    drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']
    drive_start_idx = getLeftIndex(plot_tlist, drive_start_t)
    get_phi_tlist_for_vid(plot_tlist[drive_start_idx+1:], plot_phi_tlist[drive_start_idx+1:], params_dict)

    dfn = os.path.join(PLOT_DIR, 'adjacency_data.json')
    with open(dfn,'r') as f:
        adj_data = json.load(f)
    ERGRAPH = json_graph.adjacency_graph(adj_data)
    print 'tlist size is %d, phi_tlist_size is %d, phi end is %.1f'%(plot_tlist.size, plot_phi_tlist.size, plot_phi_tlist[-1])
    plot_save_video(plot_tlist, plot_xilist_tsteps, plot_yilist_tsteps, plot_f_ext_tlist, plot_phi_tlist,
                    t_start, t_end, params_dict, ERGRAPH, fps)
    return PLOT_DIR

def get_spring_distance_driven(RGRAPH, node_i, node_j, driven_node):
    return min(nx.shortest_path_length(RGRAPH, source=driven_node, target=node_i),
               nx.shortest_path_length(RGRAPH, source=driven_node, target=node_j))

def save_spring_transitions_data(xilist_tsteps, yilist_tsteps, tlist, params_dict, RGRAPH, plot_data_dict):
    tsteps = xilist_tsteps.shape[0]
    edges = params_dict['EDGES']
    Amat = np.array(params_dict['AMAT'])
    spring_lengths_tsteps = np.empty([tsteps, edges])
    get_spring_lengths_tsteps(xilist_tsteps, yilist_tsteps, Amat, spring_lengths_tsteps)
    spring_bit_flips_tsteps = np.empty([tsteps, edges], dtype=int)
    get_spring_bit_flips_tsteps(spring_lengths_tsteps, params_dict, spring_bit_flips_tsteps)
    spring_freezing_times = np.empty(edges)
    get_spring_freezing_times(spring_bit_flips_tsteps, tlist, spring_freezing_times)
    num_frozen_springs_tsteps = np.empty(tsteps, dtype=int)
    get_num_frozen_springs(spring_freezing_times, tlist, num_frozen_springs_tsteps)
    block_size = np.int(round(params_dict['FORCE_PERIOD']/params_dict['DT']))
    drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
    drive_start_index = getLeftIndex(plot_data_dict['t/T'], drive_start_t)
    drive_start_tlist = plot_data_dict['t/T'][drive_start_index:]
    num_frozen_springs_tavg = np.zeros(drive_start_tlist.size)
    get_scaled_block_avg(1.*num_frozen_springs_tsteps, num_frozen_springs_tavg, block_size, 1.)
    if abs(num_frozen_springs_tavg[-1]) < 1e-3:
        num_frozen_springs_tavg[-1] = num_frozen_springs_tavg[-2]
    plot_data_dict['<num_frozen_springs(t)>_T'] = num_frozen_springs_tavg
    plot_data_dict['spring_freezing_times'] = spring_freezing_times/params_dict['FORCE_PERIOD']
    (i_nonzero, j_nonzero) = np.nonzero(Amat)
    num_nonzero = i_nonzero.shape[0]
    ten_index_length = int(10.*params_dict['FORCE_PERIOD']/(params_dict['DT'])) ################## MAKE THIS 100. INSTEAD OF 1.
    running_avg_indices = np.arange(0, tsteps, ten_index_length) + ten_index_length
    running_avg_indices[-1] = tsteps -1
    summed_spring_bit_flips_tsteps = np.empty([running_avg_indices.size, edges])
    for edgeindex in range(edges):
        get_sum_between_sample_indices(spring_bit_flips_tsteps[:,edgeindex], running_avg_indices, 10.,
                                       summed_spring_bit_flips_tsteps[:, edgeindex])
    running_avg_tlist = np.take(tlist, running_avg_indices)/params_dict['FORCE_PERIOD']

    spring_ij_degree_dist_dict = dict()
    Aij_spring_dict = [[dict() for i in xrange(params_dict['NODES'])] for j in xrange(params_dict['NODES'])]
    edgeindex = 0
    for i in range(num_nonzero):
        if i_nonzero[i] > j_nonzero[i]:
            Aij_spring_dict[i_nonzero[i]][j_nonzero[i]]['flip_rate'] = summed_spring_bit_flips_tsteps[:,edgeindex]
            Aij_spring_dict[i_nonzero[i]][j_nonzero[i]]['distance_driven_node'] = get_spring_distance_driven(RGRAPH,
                                  i_nonzero[i], j_nonzero[i], params_dict['MAX_DEG_VERTEX_I'])
            spring_ij_degree_dist_dict[(i_nonzero[i], j_nonzero[i])] = get_spring_distance_driven(RGRAPH, i_nonzero[i],
                                                      j_nonzero[i], params_dict['MAX_DEG_VERTEX_I'])
            edgeindex += 1

    edges = RGRAPH.edges()
    list_ij_sorted_by_dist = [ item_tuple[0] for item_tuple in sorted(spring_ij_degree_dist_dict.items(), key=lambda kv: kv[1]) ]

    spring_flips_tsteps_by_distance = np.empty([params_dict['EDGES'], running_avg_indices.size])

    for edgeindex, ij_pair in enumerate(list_ij_sorted_by_dist):
        spring_flips_tsteps_by_distance[edgeindex,:] = Aij_spring_dict[ij_pair[0]][ij_pair[1]]['flip_rate']

    plot_data_dict['spring_flips_tsteps_by_distance'] = spring_flips_tsteps_by_distance
    plot_data_dict['spring_flips_tlist'] = running_avg_tlist


def getLeftIndex(myArray, myNumber):
    """
    Assumes myList is sorted. Returns closest index to myNumber.
    If two numbers are equally close, return the smallest index.
    """
    compareArrary = myArray > myNumber
    compareList = compareArrary.tolist()
    if True in compareList:
        pos = compareList.index(True)
        # pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0
        else:
            return pos-1
    else:
        return len(compareList)-1

def read_perturb_sim_data_save_hop_Qdot_avg(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()
    keys_list = ['t/T', '<# spring transitions(t)>_T', '<U(t)/(Barrier)>_T', '<KE(t)/(Barrier)>_T',
                 '<E(t)/(Barrier)>_T', '<F(t).v(t)/(Barrier/T)>_T',
                 '<dissipation_rate(t)/(Barrier/T)>_T']
    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            plot_data_dict[k] = np.array(json.load(f))

    dfn = os.path.join(PLOT_DIR, 'first_hop_dict.json')
    with open(dfn, 'r') as f:
        first_hop_dict = json.load(f)

    first_hop_time = first_hop_dict['first_hop_time']
    first_hop_index = getLeftIndex(plot_data_dict['t/T'], first_hop_time)
    first_hop_qdot_avg = np.mean(
                            plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][first_hop_index-10:first_hop_index+1])
    first_hop_dict['first_hop_Qdot_avg'] = first_hop_qdot_avg

    dfn = os.path.join(params_dict['PLOT_DIR'], 'first_hop_dict.json')
    with open(dfn, 'w') as f:
        json.dump(first_hop_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

def read_perturb_sim_data_save_hop_Wdot_avg(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()
    keys_list = ['t/T', '<# spring transitions(t)>_T', '<F(t).v(t)/(Barrier/T)>_T']
    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            plot_data_dict[k] = np.array(json.load(f))

    dfn = os.path.join(PLOT_DIR, 'first_hop_dict.json')
    with open(dfn, 'r') as f:
        first_hop_dict = json.load(f)

    first_hop_time = first_hop_dict['first_hop_time']
    first_hop_index = getLeftIndex(plot_data_dict['t/T'], first_hop_time)
    first_hop_wdot_avg = np.mean(
                            plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'][first_hop_index-10:first_hop_index+1])
    first_hop_dict['first_hop_Wdot_avg'] = first_hop_wdot_avg

    dfn = os.path.join(params_dict['PLOT_DIR'], 'first_hop_dict.json')
    with open(dfn, 'w') as f:
        json.dump(first_hop_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

def read_perturb_sim_data_return_qdot_wdot_orig(PLOT_DIR):
    original_plot_dir = PLOT_DIR[:PLOT_DIR.rfind('_beta_1e4_rep_')+15]
    keys_list = ['<F(t).v(t)/(Barrier/T)>_T', '<dissipation_rate(t)/(Barrier/T)>_T']
    save_keys = ['orig_wdot_end', 'orig_qdot_end']
    dfn = os.path.join(PLOT_DIR, 'first_hop_dict.json')
    with open(dfn, 'r') as f:
        first_hop_dict = json.load(f)

    for i, k in enumerate(keys_list):
        dfn = os.path.join(original_plot_dir, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            data_arr = np.array(json.load(f))
            first_hop_dict[save_keys[i]] = np.mean(data_arr[-10:])

    dfn = os.path.join(PLOT_DIR, 'first_hop_dict.json')
    with open(dfn, 'w') as f:
        json.dump(first_hop_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    return [first_hop_dict['orig_wdot_end'], first_hop_dict['orig_qdot_end']]

def save_perturbed_plot(plot_data_dict, params_dict):
    fig = plt.figure(figsize=(15, 10), dpi=100)
    gs = gridspec.GridSpec(2, 3)
    axarr = [ [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])],
             [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])] ]

    axarr[0][0].plot(plot_data_dict['t/T'], plot_data_dict['<U(t)/(Barrier)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][0].set(ylabel=r"$\langle U(t)/E_b\rangle_T$")
    axarr[0][0].margins(0.05)

    axarr[1][0].plot(plot_data_dict['t/T'], plot_data_dict['<KE(t)/(Barrier)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][0].set(ylabel=r"$\langle KE(t)/E_b\rangle_T$")
    axarr[1][0].margins(0.05)

    axarr[0][1].plot(plot_data_dict['t/T'], plot_data_dict['<# spring transitions(t)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][1].set(ylabel=r"$\langle$ # spring transitions $\rangle_T$")
    axarr[0][1].margins(0.05)

    axarr[1][1].plot(plot_data_dict['t/T'], plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][1].set(ylabel=r"$\langle\dot{Q}(t)/(E_b/T)\rangle_T$")
    axarr[1][1].margins(0.05)

    axarr[0][2].plot(plot_data_dict['t/T'], plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][2].set(ylabel=r"$\langle \dot{W}(t)/(E_b/T)\rangle_T$")
    axarr[0][2].margins(0.05)

    axarr[1][2].plot(plot_data_dict['t/T'], plot_data_dict['<E(t)/(Barrier)>_T'],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][2].set(ylabel=r"$\langle E(t)/E_b\rangle_T$")
    axarr[1][2].margins(0.05)

    gs.update(left=0.02, right=0.98, bottom=0.05, top=0.98) # ,wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    plot_fn = os.path.join(params_dict['PLOT_DIR'], 'summary_plot.png')
    fig.savefig(plot_fn)
    plt.close()


def read_perturb_sim_data_save_plot(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()
    keys_list = ['t/T', '<# spring transitions(t)>_T', '<U(t)/(Barrier)>_T', '<KE(t)/(Barrier)>_T',
                '<E(t)/(Barrier)>_T', '<F(t).v(t)/(Barrier/T)>_T',
                '<dissipation_rate(t)/(Barrier/T)>_T',
                 ]
    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
    params_dict['PLOT_DIR'] = PLOT_DIR

    save_perturbed_plot(plot_data_dict, params_dict)

#assume knowledge of keys in plot_data_dict which is defined in *.pyx:
def save_plots(plot_data_dict, params_dict):
    #create the summary plot
    fig = plt.figure(figsize=(26, 15), dpi=100)
    gs = gridspec.GridSpec(3, 5)
    axarr = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]),
              plt.subplot(gs[0, 4])],
             [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3]),
              plt.subplot(gs[1, 4])],
             [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(gs[2, 3]),
              plt.subplot(gs[2, 4])]]
    start_index = params_dict['START_PLOT_INDEX']
    drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
    drive_start_index = getLeftIndex(plot_data_dict['t/T'], drive_start_t)

    params_dict['freq_switch_times'] = [ freq_switch_t + params_dict['T_RELAX'] for freq_switch_t in params_dict['freq_switch_times'] ]
    axarr[0][0].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<U(t)/(Barrier)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][0].set(ylabel=r"$\langle U(t)/\mathrm{Barrier}\rangle_T$")
    axarr[0][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][0].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][0].margins(0.05)

    axarr[1][0].plot(plot_data_dict['t/T'][start_index:],
                     plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][start_index:], linestyle=':',
                     c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][0].set(ylabel=r"$\langle\dot{Q}(t)/(\mathrm{Barrier}/T)\rangle_T$")
    axarr[1][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][0].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][0].margins(0.05)

    axarr[2][0].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<KE(t)_{c.o.m.}/KE>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[2][0].set(xlabel=r"time, $t/T_{\mathrm{drive}}$",
                    ylabel=r"$\langle\mathrm{KE(t)}_\mathrm{c.o.m.}/\mathrm{KE(t)}\rangle_T$")
    axarr[2][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    axarr[2][0].margins(0.05)

    axarr[0][1].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<E(t)/(Barrier)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][1].set(ylabel=r" $\langle\mathrm{E}(t)/\mathrm{B}\rangle_T$")
    axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][1].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][1].margins(0.05)

    axarr[1][1].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][1].set(ylabel=r"$\langle F(t)\cdot \mathbf{v}(t)/(\mathrm{Barrier}/T)\rangle_T$")
    axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][1].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][1].margins(0.05)

    axarr[2][1].plot(plot_data_dict['t/T'][start_index:],
                     plot_data_dict['<distance from relaxed state(t)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0], marker='o', markerfacecolor=params_dict['COLORS'][0],
                     markeredgecolor='None', markersize=5)
    axarr[2][1].set(ylabel=r"$\langle d_\mathrm{relaxed}(t)\rangle_T$",
                    xlabel=r"time, $t/T_{\mathrm{drive}}$")
    axarr[2][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[2][1].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[2][1].margins(0.05)

    axarr[0][2].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<# spring transitions(t)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5,
                     label=r"<# spring flips(t)$>_T$")
    axarr[0][2].set(ylabel=r"$\Sigma_T\mathrm{\,\#\, spring\, flips(t)}$")
    axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][2].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][2].margins(0.05)

    axarr[1][2].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['<# long springs(t)>_T'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5,
                     label=r"<# long springs(t)$>_T$")
    axarr[1][2].set(ylabel=r"$\langle \mathrm{\#\,long\,springs(t)}\rangle_T$")
    axarr[1][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][2].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][2].margins(0.05)

    axarr[2][2].plot(plot_data_dict['t/T'][start_index:], plot_data_dict['num_unique_states_tsteps'][start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5,
                     label=r"# unique states(t)")
    axarr[2][2].set(xlabel=r"time, $t/T_{\mathrm{drive}}$", ylabel=r"# unique states(t)")
    axarr[2][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[2][2].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[2][2].margins(0.05)

    ####################### plot spectra related stuff ###################

    eighistplot = axarr[0][3].imshow(plot_data_dict['eig_hist_tsteps'], aspect='auto',
                                     cmap=plt.get_cmap(params_dict['cmap']),
                                     extent=(plot_data_dict['t/T'][start_index], plot_data_dict['t/T'][-1],
                                             plot_data_dict['eig_bins'][-1], plot_data_dict['eig_bins'][0]))
    divider = make_axes_locatable(axarr[0][3])
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(eighistplot, orientation='vertical', label='normal mode density', cax=cax)
    for drive_T in params_dict['T_list']:
        axarr[0][3].axhline(y=2 * np.pi / float(drive_T), c='grey', linestyle='-.', alpha=0.6)
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][3].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][3].set(ylabel=r"$\omega$")
    axarr[0][3].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    #
    forcing_dirn_plot = axarr[1][3].imshow(plot_data_dict['forcing_overlap_tsteps'], aspect='auto',
                                           cmap=plt.get_cmap(params_dict['cmap']),
                                           extent=(plot_data_dict['t/T'][start_index], plot_data_dict['t/T'][-1],
                                                   plot_data_dict['eig_bins'][-1], plot_data_dict['eig_bins'][0]))
    divider = make_axes_locatable(axarr[1][3])
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(forcing_dirn_plot, orientation='vertical',
                 label=r"$\vert\langle \hat{\omega_i}\vert \hat{F}\rangle\vert^2 $", cax=cax)
    axarr[1][3].set(
        ylabel=r"$\omega$")
    for drive_T in params_dict['T_list']:
        axarr[1][3].axhline(y=2 * np.pi / float(drive_T), c='grey', linestyle='-.', alpha=0.6)
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][3].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][3].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    #

    # forcing_perp_dirnplot = axarr[2][3].imshow(plot_data_dict['forcing_perp_overlap_tsteps'], aspect='auto',
    #                                            cmap=plt.get_cmap(params_dict['cmap']),
    #                                            extent=(plot_data_dict['t/T'][start_index], plot_data_dict['t/T'][-1],
    #                                                    plot_data_dict['eig_bins'][-1], plot_data_dict['eig_bins'][0]))
    # divider = make_axes_locatable(axarr[2][3])
    # cax = divider.append_axes("right", "5%", pad="3%")
    # plt.colorbar(forcing_perp_dirnplot, orientation='vertical',
    #              label=r"$\vert\langle \hat{\omega_i}\vert \hat{F}_\perp\rangle\vert^2 $", cax=cax)
    # axarr[2][3].set(xlabel=r"time, $t/T_{\mathrm{drive}}$",
    #     ylabel=r"$\omega$")
    # for drive_T in params_dict['T_list']:
    #     axarr[2][3].axhline(y=2 * np.pi / float(drive_T), c='grey', linestyle='-.', alpha=0.6)
    # for freq_switch_t in params_dict['freq_switch_times']:
    #     axarr[2][3].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    # axarr[2][3].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')


    gs.update(left=0.02, right=0.98, bottom=0.05, top=0.98) # ,wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    plot_fn = os.path.join(params_dict['PLOT_DIR'], 'summary_plot.png')
    fig.savefig(plot_fn)
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(6,6), dpi=100)
    ax.plot(plot_data_dict['t/T'], np.log(np.abs(plot_data_dict['<(Edot + dissipation_rate - work_rate)/(Barrier/T)>_T'])),
                    linestyle=':', c=params_dict['COLORS'][0], marker='o', markerfacecolor=params_dict['COLORS'][0],
                    markeredgecolor='None', markersize=5)
    ax.set(xlabel=r"times, $t/T_{\mathrm{drive}}$",ylabel=r"$\log(<(\dot{E} + \dot{Q} - \dot{W})/(\mathrm{B}/T)>_T)$")
    ax.margins(0.05)
    plt.tight_layout()
    fig.savefig(os.path.join(params_dict['PLOT_DIR'], 'net_E_violation_rate.png'))
    plt.close()



def overlay_save_initial_frames_final_frames(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['PLOT_DIR'] = PLOT_DIR
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    with open(dfn, 'r') as f:
        tlist = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'adjacency_data.json')
    with open(dfn, 'r') as f:
        adj_data = json.load(f)
    ergraph = json_graph.adjacency_graph(adj_data)

    drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
    # drive_start_index = getLeftIndex(tlist, drive_start_t)
    drive_start_index = int(params_dict['T_RELAX'] / (2 * params_dict['DT'] * params_dict['PLOT_DELTA_STEPS']))
    drive_period_delta_index = int(params_dict['FORCE_PERIOD']/(params_dict['DT']*params_dict['PLOT_DELTA_STEPS']))
    print 'force period is %.1f, plot delta steps is %d, drive period delta index is %d'%(params_dict['FORCE_PERIOD'],
                                                                                          params_dict['PLOT_DELTA_STEPS'],
                                                                                          drive_period_delta_index)
    # phi_vals_list = params_dict['phi_values_list']
    # num_cycles_per_phi = params_dict['num_cycles_per_phi']
    num_periods = 1

    colors = []
    for node in range(params_dict['NODES']):
        colors.append('grey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'maroon'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'sandybrown'

    # initial_frame_indices = np.arange(drive_start_index+10*drive_period_delta_index, drive_start_index+(10+num_periods)*drive_period_delta_index)
    # initial_frames_xilist = xilist_tsteps[drive_start_index+10*drive_period_delta_index:drive_start_index+(10+num_periods)*drive_period_delta_index]
    # initial_frames_yilist = yilist_tsteps[drive_start_index+10*drive_period_delta_index:drive_start_index+(10+num_periods)*drive_period_delta_index]
    initial_frame_indices = np.arange(0, (num_periods)*drive_period_delta_index)
    initial_frames_xilist = xilist_tsteps[:(num_periods)*drive_period_delta_index]
    initial_frames_yilist = yilist_tsteps[:(num_periods)*drive_period_delta_index]
    print 'length of initial frame indices is %d' %(initial_frame_indices.size)

    tlength = xilist_tsteps.shape[0]
    final_frame_indices = np.arange(tlength-num_periods*drive_period_delta_index, tlength)
    final_frames_xilist = xilist_tsteps[-num_periods*drive_period_delta_index:]
    final_frames_yilist = yilist_tsteps[-num_periods*drive_period_delta_index:]
    print 'length of final frame indices is %d' % (final_frame_indices.size)
    # final_phi_list = [phi_vals_list[-(i//(num_cycles_per_phi*drive_period_delta_index))] for i in range(final_frame_indices.size)]
    # final_phi_list.reverse()
    # print 'length of final_phi_list is %d' %(len(final_phi_list))

    xmin = min(np.amin(initial_frames_xilist), np.amin(final_frames_xilist))
    ymin = min(np.amin(initial_frames_yilist), np.amin(final_frames_yilist))

    xmax = max(np.amax(initial_frames_xilist), np.amax(final_frames_xilist))
    ymax = max(np.amax(initial_frames_yilist), np.amax(final_frames_yilist))

    fsize = 10
    dpi = 200

    ########## initial frames overlay ##################
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    pos = dict()
    for idx, i in enumerate(initial_frame_indices):
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_tsteps[i][node], yilist_tsteps[i][node])
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=500, width=0.4, with_labels=False, ax=ax, alpha=0.35)
        ax.set_aspect('equal')
        ax.set(ylim=[ymin - 0.2, ymax + 0.2], xlim=[xmin - 0.2, xmax + 0.2],
               aspect=(xmax - xmin + 0.4) / (ymax - ymin + 0.4))

    plot_fn = os.path.join(params_dict['PLOT_DIR'], 'initial_frames_overlay.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()

    ########## final frames overlay ##################
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    pos = dict()
    for idx, i in enumerate(final_frame_indices):
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_tsteps[i][node], yilist_tsteps[i][node])
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=500, width=0.4, with_labels=False, ax=ax, alpha=0.35)
        ax.set_aspect('equal')
        ax.set(ylim=[ymin - 0.2, ymax + 0.2], xlim=[xmin - 0.2, xmax + 0.2],
               aspect=(xmax - xmin + 0.4) / (ymax - ymin + 0.4))

    plot_fn = os.path.join(params_dict['PLOT_DIR'], 'final_frames_overlay.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()


def resave_graph(xilist, yilist, t, save_str, params_dict, ergraph, plot_dir):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('grey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'maroon'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'sandybrown'
    xmin = np.amin(xilist)
    xmax = np.amax(xilist)
    ymin = np.amin(yilist)
    ymax = np.amax(yilist)
    fsize = 10
    dpi = 400

    for node in range(params_dict['NODES']):
        pos[node] = (xilist[node], yilist[node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=500, width=3., with_labels=False, ax=ax)
    ax.set_aspect('equal')
    # ax.grid(True, which='both')
    # ax.axis('on')
    # ax.grid('on')
    # ax.text(xmin, ymax + 0.5, 't=%.3f' % (t), fontsize=12)
    ax.set(ylim=[ymin - 0.2, ymax + 0.2], xlim=[xmin - 0.2, xmax + 0.2],
           aspect=(xmax - xmin + 0.4) / (ymax - ymin + 0.4))
    # time_stamp = '%d' % round(t * 100)
    prefix = save_str
    plot_fn = os.path.join(plot_dir, prefix + '_resaved.png')
    # plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + '_resaved.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()


def save_graph(xilist, yilist, numMoves, save_str, params_dict, ergraph):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'peru'
    xmin = np.amin(xilist)
    xmax = np.amax(xilist)
    ymin = np.amin(yilist)
    ymax = np.amax(yilist)
    fsize = 6
    dpi = 50

    for node in range(params_dict['NODES']):
        pos[node] = (xilist[node], yilist[node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, '%d moves' % (numMoves), fontsize=12)
    ax.set(ylim=[ymin - 0.5, ymax + 0.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
    time_stamp = '%d' %(numMoves)
    prefix = save_str
    for j in range(10 - len(time_stamp)):
        prefix = prefix + '0'
    plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()

def save_graph_fext(xilist, yilist, t, f_ext, phi, save_str, params_dict, ergraph):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'peru'
    xmin = np.amin(xilist)
    xmax = np.amax(xilist)
    ymin = np.amin(yilist)
    ymax = np.amax(yilist)
    fsize = 6
    dpi = 50

    for node in range(params_dict['NODES']):
        pos[node] = (xilist[node], yilist[node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, 't=%.3f' % (t), fontsize=12)
    ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
    f_ext *=(1./params_dict['FORCE_AMP'])
    ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0]-f_ext*np.cos(phi),
                        pos[params_dict['MAX_DEG_VERTEX_I']][1]-f_ext*np.sin(phi)),
                xycoords='data',
                xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                textcoords='data',
                arrowprops=dict(arrowstyle='<|-',
                                color='firebrick',
                                lw=2.5)
                )
    time_stamp = '%d' % round(t * 100)
    prefix = save_str
    for j in range(10 - len(time_stamp)):
        prefix = prefix + '0'
    plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()


def plot_save_graph(tlist, xilist_t, yilist_t, f_ext_tlist, phi_tlist, params_dict, ergraph):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'peru'
    xmin = np.amin(xilist_t[0])
    xmax = np.amax(xilist_t[0])
    ymin = np.amin(yilist_t[0])
    ymax = np.amax(yilist_t[0])
    fsize = 6
    dpi = 50
    fmax = np.amax(f_ext_tlist)
    f_ext_tlist *=(1./(fmax+1e-4))

    for node in range(params_dict['NODES']):
        pos[node] = (xilist_t[0][node], yilist_t[0][node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, 't=%.3f' % (tlist[0]), fontsize=12)
    ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
    ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0]-f_ext_tlist[0]*np.cos(phi_tlist[0]),
                        pos[params_dict['MAX_DEG_VERTEX_I']][1]-f_ext_tlist[0]*np.sin(phi_tlist[0])),
                xycoords='data',
                xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                textcoords='data',
                arrowprops=dict(arrowstyle='<|-',
                                color='firebrick',
                                lw=2.5)
                )
    t = 0
    time_stamp = '%d' % round(t * 100)
    prefix = ''
    for j in range(10 - len(time_stamp)):
        prefix = prefix + '0'
    plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()

    i = params_dict['SAVE_FIG_DELTA_FRAMES']

    while i < tlist.size:
        xmin = np.amin(xilist_t[i])
        xmax = np.amax(xilist_t[i])
        ymin = np.amin(yilist_t[i])
        ymax = np.amax(yilist_t[i])

        t = tlist[i]
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_t[i][node], yilist_t[i][node])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        ax.axis('on')
        ax.grid('on')
        ax.text(xmin, ymax + 0.5, r"t/$T_\mathrm{drive}$=%.3f" % (t), fontsize=12)
        ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
               aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
        ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0] + f_ext_tlist[i]*np.cos(phi_tlist[i]),
                            pos[params_dict['MAX_DEG_VERTEX_I']][1] + f_ext_tlist[i]*np.sin(phi_tlist[i])),
                    xycoords='data',
                    xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                    textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    color='firebrick',
                                    lw=2.5)
                    )

        time_stamp = '%d' % round(t * 100)
        prefix = ''
        for j in range(10 - len(time_stamp)):
            prefix = prefix + '0'
        plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
        fig.savefig(plot_fn, dpi=dpi)
        plt.close()
        i += params_dict['SAVE_FIG_DELTA_FRAMES']

def plot_save_graph_pos_drive(tlist, xilist_t, yilist_t, pos_tlist, phi_tlist, params_dict, ergraph):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'peru'
    xmin = np.amin(xilist_t[0])
    xmax = np.amax(xilist_t[0])
    ymin = np.amin(yilist_t[0])
    ymax = np.amax(yilist_t[0])
    fsize = 6
    dpi = 50
    fmax = np.amax(pos_tlist)
    pos_tlist *=(1./(fmax+1e-4))

    for node in range(params_dict['NODES']):
        pos[node] = (xilist_t[0][node], yilist_t[0][node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, 't=%.3f' % (tlist[0]), fontsize=12)
    ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
    ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0]-pos_tlist[0]*np.cos(phi_tlist[0]),
                        pos[params_dict['MAX_DEG_VERTEX_I']][1]-pos_tlist[0]*np.sin(phi_tlist[0])),
                xycoords='data',
                xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                textcoords='data',
                arrowprops=dict(arrowstyle='<|-',
                                color='firebrick',
                                lw=2.5)
                )
    t = 0
    time_stamp = '%d' % round(t * 100)
    prefix = ''
    for j in range(10 - len(time_stamp)):
        prefix = prefix + '0'
    plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()

    i = params_dict['SAVE_FIG_DELTA_FRAMES']

    while i < tlist.size:
        xmin = np.amin(xilist_t[i])
        xmax = np.amax(xilist_t[i])
        ymin = np.amin(yilist_t[i])
        ymax = np.amax(yilist_t[i])

        t = tlist[i]
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_t[i][node], yilist_t[i][node])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        ax.axis('on')
        ax.grid('on')
        ax.text(xmin, ymax + 0.5, r"t/$T_\mathrm{drive}$=%.3f" % (t), fontsize=12)
        ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
               aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
        ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0] + pos_tlist[i]*np.cos(phi_tlist[i]),
                            pos[params_dict['MAX_DEG_VERTEX_I']][1] + pos_tlist[i]*np.sin(phi_tlist[i])),
                    xycoords='data',
                    xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                    textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    color='firebrick',
                                    lw=2.5)
                    )

        time_stamp = '%d' % round(t * 100)
        prefix = ''
        for j in range(10 - len(time_stamp)):
            prefix = prefix + '0'
        plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + time_stamp + '.png')
        fig.savefig(plot_fn, dpi=dpi)
        plt.close()
        i += params_dict['SAVE_FIG_DELTA_FRAMES']

def plot_save_video_pos_drive(tlist, xilist_t, yilist_t, x_drive_tlist, y_drive_tlist, params_dict, ergraph, fps):
    start_index = params_dict[
        'START_PLOT_INDEX']  # int(float(params_dict['RELAX_STEPS'])/float(params_dict['PLOT_DELTA_STEPS']))
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'peru'
    xmin = np.amin(xilist_t[start_index:])
    xmax = np.amax(xilist_t[start_index:])
    ymin = np.amin(yilist_t[start_index:])
    ymax = np.amax(yilist_t[start_index:])
    fsize = 6
    dpi = 50
    xmax = np.amax(x_drive_tlist)
    ymax = np.amax(y_drive_tlist)
    fmax = max(xmax, ymax)
    x_drive_tlist *= (1. / (fmax+1e-3))
    y_drive_tlist *= (1. / (fmax + 1e-3))
    for node in range(params_dict['NODES']):
        pos[node] = (xilist_t[start_index][node], yilist_t[start_index][node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)

    nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, 't=%.3f' % (tlist[start_index]), fontsize=10)
    ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
    ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0] + x_drive_tlist[0],
                        pos[params_dict['MAX_DEG_VERTEX_I']][1] + y_drive_tlist[0]),
                xycoords='data',
                xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                textcoords='data',
                arrowprops=dict(arrowstyle='-',
                                color='firebrick',
                                lw=2.5)
                )
    # ofn = os.path.join(params_dict['PLOT_DIR'], 'n%d_m%d_equal_E_diff_k_%s_A%.1f_T_%.1f'%(params_dict['NODES'],params_dict['EDGES'],
    #                                                                                     params_dict['FORCE_STRING'],
    #                                                                                     params_dict['FORCE_AMP'],
    #                                                                                     params_dict['FORCE_PERIOD']))
    ofn = params_dict['PLOT_DIR'] +'_fps%d'%(fps) + '_vid.mp4'
    # ofn = ofn.replace('.','p')
    # ofn += '.mp4'
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = np.array(Image.open(buf))
    h, w, layers = im.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ofn, fourcc, fps, (w, h))
    buf.close()
    plt.close()
    # print 'length of tlist is %d'%(tlist.size)
    #
    for j, t in enumerate(tlist[start_index + 1:]):
        i = j + start_index + 1
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_t[i][node], yilist_t[i][node])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        ax.axis('on')
        ax.grid('on')
        ax.text(xmin, ymax + 0.5, r"t/$T_\mathrm{drive}$=%.3f" % (t), fontsize=10)
        ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
               aspect=(xmax - xmin + 1) / (ymax - ymin + 1))
        ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0] + x_drive_tlist[i],
                            pos[params_dict['MAX_DEG_VERTEX_I']][1] + y_drive_tlist[i]),
                    xycoords='data',
                    xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                    textcoords='data',
                    arrowprops=dict(arrowstyle='-',
                                    color='firebrick',
                                    lw=2.5)
                    )
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        plt.close()
        buf.close()
    out.release()


def plot_save_video(tlist, xilist_t, yilist_t, f_ext_tlist, phi_tlist, t_start, t_end, params_dict, ergraph, fps):
    start_index = getLeftIndex(tlist, t_start)
    end_index = getLeftIndex(tlist, t_end)
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('grey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'maroon'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'sandybrown'
    xmin = np.amin(xilist_t[start_index:end_index])
    xmax = np.amax(xilist_t[start_index:end_index])
    ymin = np.amin(yilist_t[start_index:end_index])
    ymax = np.amax(yilist_t[start_index:end_index])
    fsize = 6
    dpi = 100
    fmax = np.amax(f_ext_tlist)
    f_ext_tlist *=(1./fmax)
    for node in range(params_dict['NODES']):
        pos[node] = (xilist_t[start_index][node], yilist_t[start_index][node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    print 'video set up completed'
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=300, width=3., with_labels=False, ax=ax)
    ax.set_aspect('equal')
    # ax.grid(True, which='both')
    # ax.axis('on')
    # ax.grid('on')
    ax.text(xmin, ymax+0.1, r"t/$\tau$=%.1f" % (tlist[start_index]), fontsize=16)
    ax.set(ylim=[ymin - 0.1, ymax + 0.1], xlim=[xmin - 0.1, xmax + 0.1],
           aspect=(xmax - xmin + 0.2) / (ymax - ymin + 0.2))
    ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0]+f_ext_tlist[start_index]*np.cos(phi_tlist[start_index]),
                        pos[params_dict['MAX_DEG_VERTEX_I']][1]+f_ext_tlist[start_index]*np.sin(phi_tlist[start_index])),
                xycoords='data',
                xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                textcoords='data',
                arrowprops=dict(arrowstyle='-|>',
                                color='maroon',
                                lw=5)
                )
    ofn = params_dict['PLOT_DIR']+'_fps%d'%(fps)+'_vid.mp4'
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = np.array(Image.open(buf))
    h,w,layers = im.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ofn, fourcc, fps, (w, h))
    buf.close()
    plt.close()
    print 'video making started'
    for j,t in enumerate(tlist[start_index+1:end_index]):
        i = j+start_index+1
        for node in range(params_dict['NODES']):
            pos[node] = (xilist_t[i][node],yilist_t[i][node])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
        nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=300, width=3., with_labels=False, ax=ax)
        ax.set_aspect('equal')
        # ax.grid(True, which='both')
        # ax.axis('on')
        # ax.grid('on')
        ax.text(xmin, ymax+0.1, r"t/$\tau$=%.1f"%(t), fontsize=16)
        ax.set(ylim=[ymin-0.1, ymax+0.1], xlim=[xmin-0.1, xmax+0.1],
               aspect=(xmax - xmin+0.2)/(ymax - ymin+0.2))
        ax.annotate('', xy=(pos[params_dict['MAX_DEG_VERTEX_I']][0] + f_ext_tlist[i]*np.cos(phi_tlist[i]),
                            pos[params_dict['MAX_DEG_VERTEX_I']][1] + f_ext_tlist[i]*np.sin(phi_tlist[i])),
                    xycoords='data',
                    xytext=pos[params_dict['MAX_DEG_VERTEX_I']],
                    textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    color='maroon',
                                    lw=5)
                    )
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        plt.close()
        buf.close()
    out.release()


def get_first_hop_data(plot_data_dict):
    spring_hops = plot_data_dict['<# spring transitions(t)>_T']
    first_hop_index = next((i for i, x in enumerate(spring_hops) if abs(x) > 1e-3), spring_hops.size-1)
    first_hop_dict = dict()
    first_hop_dict['first_hop_time'] = plot_data_dict['t/T'][first_hop_index]
    first_hop_dict['first_hop_E'] = plot_data_dict['<E(t)/(Barrier)>_T'][first_hop_index]
    first_hop_dict['first_hop_Wdot'] = plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'][first_hop_index]
    first_hop_dict['first_hop_Qdot'] = plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][first_hop_index]
    first_hop_dict['first_hop_U'] = plot_data_dict['<U(t)/(Barrier)>_T'][first_hop_index]
    if first_hop_index > 10:
        first_hop_qdot_avg = np.mean(
                                plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][first_hop_index-10:first_hop_index+1])
        first_hop_wdot_avg = np.mean(
                                plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'][first_hop_index-10:first_hop_index+1])
    else:
        first_hop_qdot_avg = np.mean(
                                plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][:first_hop_index+1])
        first_hop_wdot_avg = np.mean(
                                plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'][:first_hop_index+1])

    first_hop_dict['first_hop_Wdot_avg'] = first_hop_wdot_avg

    first_hop_dict['first_hop_Qdot_avg'] = first_hop_qdot_avg

    return first_hop_dict

def run_undriven_sim(params_dict, xilist_0, yilist_0, vxilist_0, vyilist_0, num_cycles=50):
    params_dict['T_END'] = (params_dict['FORCE_PERIOD']*num_cycles)
    params_dict['T_RELAX'] = 0.
    TLIST = np.arange(0., params_dict['T_END'], params_dict['sim_DT'])
    SIM_STEPS = TLIST.size
    f_ext_tlist = np.zeros(SIM_STEPS)
    # phi_tlist = np.zeros(SIM_STEPS)
    params_dict['AMAT'] = np.array(params_dict['AMAT'])
    NSTEPS = np.int(np.ceil((params_dict['T_END'] - params_dict['T_START'])/params_dict['DT']))
    params_dict['NSTEPS'] = NSTEPS
    PLOT_DELTA_STEPS = np.int(params_dict['PLOT_DELTA_STEPS'])
    params_dict['PLOT_STEPS'] = np.arange(0, params_dict['NSTEPS'], PLOT_DELTA_STEPS)

    xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    wdot_tsteps = np.zeros(NSTEPS)
    qdot_tsteps = np.zeros(NSTEPS)

    xilists_tsteps[0] = xilist_0
    yilists_tsteps[0] = yilist_0
    vxilists_tsteps[0] = vxilist_0
    vyilists_tsteps[0] = vyilist_0

    # get_phi_tlist(TLIST, phi_tlist, params_dict)
    verlet_iterate(xilists_tsteps, yilists_tsteps, vxilists_tsteps,
                   vyilists_tsteps, f_ext_tlist, wdot_tsteps, qdot_tsteps, params_dict)

    edges = params_dict['EDGES']
    Amat = np.array(params_dict['AMAT'])
    spring_lengths_tsteps = np.empty([NSTEPS, edges])
    get_spring_lengths_tsteps(xilists_tsteps, yilists_tsteps, Amat, spring_lengths_tsteps)
    spring_bit_flips_tsteps = np.empty([NSTEPS, edges], dtype=int)
    get_spring_bit_flips_tsteps(spring_lengths_tsteps, params_dict, spring_bit_flips_tsteps)
    total_bit_flips = np.sum(spring_bit_flips_tsteps)

    return [xilists_tsteps[-1], yilists_tsteps[-1], vxilists_tsteps[-1], vyilists_tsteps[-1], total_bit_flips]

def run_ramped_perturbed_force_sim_save_plots(params_dict, ERGRAPH,
                                              xilist_0, yilist_0, vxilist_0, vyilist_0, num_cycles, perturb_string,
                                              is_degenerate_perturbed=False, perturb_rotation=0., degenerate_lambda_info=[]):
    PLOT_DIR = params_dict['PLOT_DIR'].replace('.','p') + perturb_string

    # new_seed2 = np.random.randint(1, 2 ** 32)
    # PLOT_DIR = PLOT_DIR + '_new_seed2_%d'%(new_seed2)
    # np.random.seed(new_seed2)
    if os.path.isdir(PLOT_DIR):
        return PLOT_DIR
    else:
        os.mkdir(PLOT_DIR)
    # else:
    #     raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))
        params_dict['PLOT_DIR'] = PLOT_DIR
        TLIST = np.arange(0., params_dict['T_END'], params_dict['sim_DT'])
        SIM_STEPS = TLIST.size
        f_ext_tlist = np.empty(SIM_STEPS)
        phi_tlist = np.empty(SIM_STEPS)
        params_dict['AMAT'] = np.array(params_dict['AMAT'])
        NSTEPS = np.int(np.ceil((params_dict['T_END'] - params_dict['T_START'])/params_dict['DT']))
        params_dict['NSTEPS'] = NSTEPS
        PLOT_DELTA_STEPS = np.int(params_dict['PLOT_DELTA_STEPS'])
        params_dict['PLOT_STEPS'] = np.arange(0, params_dict['NSTEPS'], PLOT_DELTA_STEPS)

        xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        wdot_tsteps = np.zeros(NSTEPS)
        qdot_tsteps = np.zeros(NSTEPS)

        xilists_tsteps[0] = xilist_0
        yilists_tsteps[0] = yilist_0
        vxilists_tsteps[0] = vxilist_0
        vyilists_tsteps[0] = vyilist_0

        save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
        print 'init graph saved'

        # get_phi_tlist(TLIST, phi_tlist, params_dict)
        get_ramp_f_ext_tlist(TLIST, f_ext_tlist, params_dict)

        verlet_iterate_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps,
               vyilists_tsteps, f_ext_tlist, wdot_tsteps, qdot_tsteps, params_dict)

        # f_ext_tlist = np.array(np.take(f_ext_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
        # phi_tlist = np.array(np.take(phi_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
        TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
        print 'simulation done'
        save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
        print 'final graph saved'

        plot_data_dict = get_perturbed_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                            TLIST, wdot_tsteps, qdot_tsteps, params_dict)
        print 'plot_data_dict ready'
        first_hop_dict = get_first_hop_data(plot_data_dict)
        if is_degenerate_perturbed:
            first_hop_dict['perturb_rotation'] = perturb_rotation
            first_hop_dict['degenerate_lambda'] = degenerate_lambda_info[0]
            first_hop_dict['degenerate_delta_lambda'] = degenerate_lambda_info[1]
        dfn = os.path.join(params_dict['PLOT_DIR'], 'first_hop_dict.json')
        with open(dfn, 'w') as f:
            json.dump(first_hop_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        dfn = os.path.join(params_dict['PLOT_DIR'], 'params_dict.json')
        with open(dfn, 'w') as f:
            json.dump(params_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        for k,v in plot_data_dict.iteritems():
            dfn = os.path.join(params_dict['PLOT_DIR'], str(k).replace('/','_by_')+'.json')
            with open(dfn, 'w') as f:
                json.dump(v, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        save_perturbed_plot(plot_data_dict, params_dict)
        return params_dict['PLOT_DIR']

def run_ramped_temperature_sim_save_plots(params_dict, ERGRAPH,
                                          xilist_0, yilist_0, vxilist_0, vyilist_0, num_cycles, temp_string):
    PLOT_DIR = params_dict['PLOT_DIR'].replace('.','p') + temp_string

    # new_seed2 = np.random.randint(1, 2 ** 32)
    # PLOT_DIR = PLOT_DIR + '_new_seed2_%d'%(new_seed2)
    # np.random.seed(new_seed2)
    if os.path.isdir(PLOT_DIR):
        return PLOT_DIR
    else:
        os.mkdir(PLOT_DIR)
    # else:
    #     raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))
        params_dict['PLOT_DIR'] = PLOT_DIR
        TLIST = np.arange(0., params_dict['T_END'], params_dict['sim_DT'])
        SIM_STEPS = TLIST.size
        f_ext_tlist = np.empty(SIM_STEPS)
        phi_tlist = np.empty(SIM_STEPS)
        params_dict['AMAT'] = np.array(params_dict['AMAT'])
        if '_node' in temp_string:
            temp_tlist = np.linspace(0.01, 10., SIM_STEPS)
        elif '_network' in temp_string:
            temp_tlist = np.linspace(0.001, 0.75, SIM_STEPS)
        NSTEPS = np.int(np.ceil((params_dict['T_END'] - params_dict['T_START'])/params_dict['DT']))
        params_dict['NSTEPS'] = NSTEPS
        PLOT_DELTA_STEPS = np.int(params_dict['PLOT_DELTA_STEPS'])
        params_dict['PLOT_STEPS'] = np.arange(0, params_dict['NSTEPS'], PLOT_DELTA_STEPS)

        xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
        wdot_tsteps = np.zeros(NSTEPS)
        qdot_tsteps = np.zeros(NSTEPS)

        xilists_tsteps[0] = xilist_0
        yilists_tsteps[0] = yilist_0
        vxilists_tsteps[0] = vxilist_0
        vyilists_tsteps[0] = vyilist_0

        save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
        print 'init graph saved'

        # get_phi_tlist(TLIST, phi_tlist, params_dict)
        # get_ramp_f_ext_tlist(TLIST, f_ext_tlist, params_dict)

        if 'network' in temp_string:
            temperature_drive_network_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps,
                           vyilists_tsteps, temp_tlist, wdot_tsteps, qdot_tsteps, params_dict)
        elif 'node' in temp_string:
            temperature_drive_node_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps,
                           vyilists_tsteps, temp_tlist, wdot_tsteps, qdot_tsteps, params_dict)

        TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
        print 'simulation done'
        save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
        print 'final graph saved'

        plot_data_dict = get_perturbed_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                            TLIST, wdot_tsteps, qdot_tsteps, params_dict)
        print 'plot_data_dict ready'
        first_hop_dict = get_first_hop_data(plot_data_dict)

        dfn = os.path.join(params_dict['PLOT_DIR'], 'first_hop_dict.json')
        with open(dfn, 'w') as f:
            json.dump(first_hop_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        dfn = os.path.join(params_dict['PLOT_DIR'], 'params_dict.json')
        with open(dfn, 'w') as f:
            json.dump(params_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        for k,v in plot_data_dict.iteritems():
            dfn = os.path.join(params_dict['PLOT_DIR'], str(k).replace('/','_by_')+'.json')
            with open(dfn, 'w') as f:
                json.dump(v, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        save_perturbed_plot(plot_data_dict, params_dict)
        return params_dict['PLOT_DIR']


def run_force_sim_save_plots(args_dict):
    [ERGRAPH, params_dict] = initialize_params_dict(args_dict)

    PLOT_DIR = params_dict['PLOT_DIR'].replace('.','p')
    # PLOT_DIR = PLOT_DIR.replace('pos_drive_correct', 'force_drive_correct')
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    else:
        raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))
    params_dict['PLOT_DIR'] = PLOT_DIR

    max_deg_dist = nx.single_source_shortest_path_length(ERGRAPH, params_dict['MAX_DEG_VERTEX_I']).values()
    max_deg_dist_list = sorted(max_deg_dist)[-3:]
    max_dist_nodes = list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-1])[0])
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-2])[0]))
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-3])[0]))
    max_dist_nodes = list(set(max_dist_nodes))
    max_dist_nodes = np.array(max_dist_nodes[:args_dict['num_anchors']])
    params_dict['ANCHORED_NODES'] = max_dist_nodes

    # RELAX_STEPS = params_dict['RELAX_STEPS']
    TLIST = np.arange(params_dict['T_START'], params_dict['T_END'], params_dict['sim_DT'])
    SIM_STEPS = TLIST.size
    f_ext_tlist = np.zeros(SIM_STEPS)
    # phi_tlist = np.zeros(SIM_STEPS)
    # beta_tlist = 1e6*np.ones(SIM_STEPS)

    NSTEPS = params_dict['NSTEPS']
    # print 'true nsteps is %d, while ceiling estimate is %d'%(NSTEPS, params_dict['NSTEPS'])
    xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    wdot_tsteps = np.zeros(NSTEPS)
    qdot_tsteps = np.zeros(NSTEPS)

    # init_pos = nx.spring_layout(ERGRAPH, k=1.,scale=NODES*1.)
    [xilists_tsteps[0], yilists_tsteps[0]] = init_positions(params_dict)
    vxilists_tsteps[0] = initialize_velocityvec(params_dict)
    vyilists_tsteps[0] = initialize_velocityvec(params_dict)

    save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
    print 'init graph saved'
    RELAX_STEPS_sim_DT = int(np.ceil(params_dict['RELAX_STEPS']*params_dict['SIM_STEPS_PER_DT']))
    RELAX_STEPS = params_dict['RELAX_STEPS']

    # get_phi_tlist(TLIST[RELAX_STEPS_sim_DT:], phi_tlist[RELAX_STEPS_sim_DT:], params_dict)
    # directional_temperature_drive_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
    #                              beta_tlist, phi_tlist, params_dict)
    get_f_ext_tlist(TLIST[RELAX_STEPS_sim_DT:], f_ext_tlist[RELAX_STEPS_sim_DT:], params_dict)

    undriven_relaxation_with_noise(xilists_tsteps[:RELAX_STEPS + 1], yilists_tsteps[:RELAX_STEPS + 1], vxilists_tsteps[:RELAX_STEPS + 1],
          vyilists_tsteps[:RELAX_STEPS + 1], params_dict)
    print 'relaxation over'
    save_graph(xilists_tsteps[RELAX_STEPS], yilists_tsteps[RELAX_STEPS], TLIST[RELAX_STEPS_sim_DT], 'relaxed_state', params_dict, ERGRAPH)
    print 'relaxed graph saved'

    # np.random.seed()

    verlet_iterate_with_noise(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], vxilists_tsteps[RELAX_STEPS:],
           vyilists_tsteps[RELAX_STEPS:], f_ext_tlist[RELAX_STEPS_sim_DT:], wdot_tsteps[RELAX_STEPS:],
            qdot_tsteps[RELAX_STEPS:], params_dict)

    f_ext_tlist = np.array(np.take(f_ext_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    # phi_tlist = np.array(np.take(phi_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    print 'simulation done'
    save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
    print 'final graph saved'

    plot_data_dict = get_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                        TLIST, f_ext_tlist, wdot_tsteps, qdot_tsteps, params_dict)
    print 'plot_data_dict ready'
    save_spring_transitions_data(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], TLIST[RELAX_STEPS:],
                                 params_dict, ERGRAPH, plot_data_dict)

    plot_tlist = np.take(TLIST, params_dict['PLOT_STEPS'], axis=0)
    plot_tlist /= params_dict['FORCE_PERIOD']
    plot_xilist_tsteps = np.take(xilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_yilist_tsteps = np.take(yilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vxilist_tsteps = np.take(vxilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vyilist_tsteps = np.take(vyilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_f_ext_tlist = np.take(f_ext_tlist, params_dict['PLOT_STEPS'], axis=0)
    # plot_phi_tlist = np.take(phi_tlist, params_dict['PLOT_STEPS'], axis=0)

    # params_dict['phi_final'] = phi_tlist[-1]
    save_sim_data(plot_data_dict, plot_xilist_tsteps, plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps,
                  plot_tlist, plot_f_ext_tlist, params_dict, ERGRAPH) #  plot_phi_tlist,
    print 'sim data saved'
    plot_data_dict['sampled_tlist'] = plot_tlist
    try:
        save_plots(plot_data_dict, params_dict)
    except ValueError as err:
        print 'error when saving plots for %s, is: %s'%(params_dict['PLOT_DIR'], err)
    print 'plots saved'
    # check_equilibration(plot_vxilist_tsteps, plot_vyilist_tsteps, params_dict)
    # f_ext_tlist *= params_dict['force_scale_factor']
    # plot_save_graph(plot_tlist, plot_xilist_tsteps, plot_yilist_tsteps, plot_f_ext_tlist, plot_phi_tlist, params_dict, ERGRAPH)
    # print 'graph snapshots saved'
    return params_dict['PLOT_DIR']

def run_force_sim_changing_dirn_save_plots(args_dict):
    [ERGRAPH, params_dict] = initialize_params_dict(args_dict)

    PLOT_DIR = params_dict['PLOT_DIR'].replace('.','p')
    # PLOT_DIR = PLOT_DIR.replace('pos_drive_correct', 'force_drive_correct')
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    else:
        raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))
    params_dict['PLOT_DIR'] = PLOT_DIR

    max_deg_dist = nx.single_source_shortest_path_length(ERGRAPH, params_dict['MAX_DEG_VERTEX_I']).values()
    max_deg_dist_list = sorted(max_deg_dist)[-3:]
    max_dist_nodes = list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-1])[0])
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-2])[0]))
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-3])[0]))
    max_dist_nodes = list(set(max_dist_nodes))
    max_dist_nodes = np.array(max_dist_nodes[:args_dict['num_anchors']])
    params_dict['ANCHORED_NODES'] = max_dist_nodes

    # RELAX_STEPS = params_dict['RELAX_STEPS']
    TLIST = np.arange(params_dict['T_START'], params_dict['T_END'], params_dict['sim_DT'])
    SIM_STEPS = TLIST.size
    f_ext_tlist = np.zeros(SIM_STEPS)
    # phi_tlist = np.zeros(SIM_STEPS)
    # beta_tlist = 1e6*np.ones(SIM_STEPS)

    NSTEPS = params_dict['NSTEPS']
    # print 'true nsteps is %d, while ceiling estimate is %d'%(NSTEPS, params_dict['NSTEPS'])
    xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    wdot_tsteps = np.zeros(NSTEPS)
    qdot_tsteps = np.zeros(NSTEPS)

    # init_pos = nx.spring_layout(ERGRAPH, k=1.,scale=NODES*1.)
    [xilists_tsteps[0], yilists_tsteps[0]] = init_positions(params_dict)
    vxilists_tsteps[0] = initialize_velocityvec(params_dict)
    vyilists_tsteps[0] = initialize_velocityvec(params_dict)

    save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
    print 'init graph saved'
    RELAX_STEPS_sim_DT = int(np.ceil(params_dict['RELAX_STEPS']*params_dict['SIM_STEPS_PER_DT']))
    RELAX_STEPS = params_dict['RELAX_STEPS']
    SIM_STEPS_PER_DT = params_dict['SIM_STEPS_PER_DT']

    # get_phi_tlist(TLIST[RELAX_STEPS_sim_DT:], phi_tlist[RELAX_STEPS_sim_DT:], params_dict)
    # directional_temperature_drive_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
    #                              beta_tlist, phi_tlist, params_dict)
    get_f_ext_tlist(TLIST[RELAX_STEPS_sim_DT:], f_ext_tlist[RELAX_STEPS_sim_DT:], params_dict)

    undriven_relaxation_with_noise(xilists_tsteps[:RELAX_STEPS + 1], yilists_tsteps[:RELAX_STEPS + 1], vxilists_tsteps[:RELAX_STEPS + 1],
          vyilists_tsteps[:RELAX_STEPS + 1], params_dict)
    print 'relaxation over'
    save_graph(xilists_tsteps[RELAX_STEPS], yilists_tsteps[RELAX_STEPS], TLIST[RELAX_STEPS_sim_DT], 'relaxed_state', params_dict, ERGRAPH)
    print 'relaxed graph saved'

    # np.random.seed()
    phi = params_dict['phi_values_list'][0]
    forcing_direction = np.zeros(2*params_dict['NODES'])
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']] = np.cos(phi)
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']+1] = np.sin(phi)
    params_dict['FORCING_VEC'] = forcing_direction

    ########## assuming dirn is switched according to freq switch steps #############
    freq_switch_steps = params_dict['freq_switch_steps']
    #################################################################################

    verlet_iterate_with_noise(xilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                              yilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                              vxilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                              vyilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                              f_ext_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:
                                          (RELAX_STEPS+freq_switch_steps[0])*SIM_STEPS_PER_DT],
                              wdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
                              qdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
                              params_dict)

    for i in range(1, len(freq_switch_steps)):
        phi = params_dict['phi_values_list'][i]
        forcing_direction = np.zeros(2 * params_dict['NODES'])
        forcing_direction[2 * params_dict['MAX_DEG_VERTEX_I']] = np.cos(phi)
        forcing_direction[2 * params_dict['MAX_DEG_VERTEX_I'] + 1] = np.sin(phi)
        params_dict['FORCING_VEC'] = forcing_direction

        verlet_iterate_with_noise(
            xilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            yilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            vxilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            vyilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            f_ext_tlist[(RELAX_STEPS+freq_switch_steps[i-1])*SIM_STEPS_PER_DT:
                            (RELAX_STEPS+freq_switch_steps[i])*SIM_STEPS_PER_DT],
            wdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            qdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
            params_dict)

    f_ext_tlist = np.array(np.take(f_ext_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    # phi_tlist = np.array(np.take(phi_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    print 'simulation done'
    save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
    print 'final graph saved'

    plot_data_dict = get_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                        TLIST, f_ext_tlist, wdot_tsteps, qdot_tsteps, params_dict)
    print 'plot_data_dict ready'
    # save_spring_transitions_data(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], TLIST[RELAX_STEPS:],
    #                              params_dict, ERGRAPH, plot_data_dict)

    plot_tlist = np.take(TLIST, params_dict['PLOT_STEPS'], axis=0)
    plot_tlist /= params_dict['FORCE_PERIOD']
    plot_xilist_tsteps = np.take(xilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_yilist_tsteps = np.take(yilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vxilist_tsteps = np.take(vxilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vyilist_tsteps = np.take(vyilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_f_ext_tlist = np.take(f_ext_tlist, params_dict['PLOT_STEPS'], axis=0)
    # plot_phi_tlist = np.take(phi_tlist, params_dict['PLOT_STEPS'], axis=0)

    # params_dict['phi_final'] = phi_tlist[-1]
    save_sim_data(plot_data_dict, plot_xilist_tsteps, plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps,
                  plot_tlist, plot_f_ext_tlist, params_dict, ERGRAPH) #  plot_phi_tlist,
    print 'sim data saved'
    plot_data_dict['sampled_tlist'] = plot_tlist
    try:
        save_plots(plot_data_dict, params_dict)
    except ValueError as err:
        print 'error when saving plots for %s, is: %s'%(params_dict['PLOT_DIR'], err)
    print 'plots saved'
    # check_equilibration(plot_vxilist_tsteps, plot_vyilist_tsteps, params_dict)
    # f_ext_tlist *= params_dict['force_scale_factor']
    # plot_save_graph(plot_tlist, plot_xilist_tsteps, plot_yilist_tsteps, plot_f_ext_tlist, plot_phi_tlist, params_dict, ERGRAPH)
    # print 'graph snapshots saved'
    return params_dict['PLOT_DIR']


def run_pos_drive_sim_save_plots(args_dict):
    [ERGRAPH, params_dict] = initialize_params_dict(args_dict)

    PLOT_DIR = params_dict['PLOT_DIR'].replace('.','p')

    # if args_dict['is_anchored']:
    #     PLOT_DIR = PLOT_DIR + '_anchored'
    # else:
    #     PLOT_DIR = PLOT_DIR + '_unanchored'

    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    else:
        raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))

    params_dict['PLOT_DIR'] = PLOT_DIR
    # params_dict['is_anchored'] = args_dict['is_anchored']
    max_deg_dist = nx.single_source_shortest_path_length(ERGRAPH, params_dict['MAX_DEG_VERTEX_I']).values()
    max_deg_dist_list = sorted(max_deg_dist)[-3:]
    max_dist_nodes = list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-1])[0])
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-2])[0]))
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-3])[0]))
    max_dist_nodes = list(set(max_dist_nodes))
    max_dist_nodes = np.array(max_dist_nodes[:args_dict['num_anchors']])
    params_dict['ANCHORED_NODES'] = max_dist_nodes
    # RELAX_STEPS = params_dict['RELAX_STEPS']
    TLIST = np.arange(params_dict['T_START'], params_dict['T_END'], params_dict['sim_DT'])
    SIM_STEPS = TLIST.size

    pos_drive_tlist = np.zeros(SIM_STEPS)
    phi_tlist = np.zeros(SIM_STEPS)
    # beta_tlist = 1e6*np.ones(SIM_STEPS)

    NSTEPS = params_dict['NSTEPS']
    # print 'true nsteps is %d, while ceiling estimate is %d'%(NSTEPS, params_dict['NSTEPS'])
    xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    wdot_tsteps = np.zeros(NSTEPS)
    qdot_tsteps = np.zeros(NSTEPS)

    # init_pos = nx.spring_layout(ERGRAPH, k=1.,scale=NODES*1.)
    [xilists_tsteps[0], yilists_tsteps[0]] = init_positions(params_dict)
    vxilists_tsteps[0] = initialize_velocityvec(params_dict)
    vyilists_tsteps[0] = initialize_velocityvec(params_dict)

    save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
    print 'init graph saved'
    RELAX_STEPS_sim_DT = int(np.ceil(params_dict['RELAX_STEPS']*params_dict['SIM_STEPS_PER_DT']))
    RELAX_STEPS = params_dict['RELAX_STEPS']

    get_phi_tlist(TLIST, phi_tlist, params_dict)
    get_pos_tlist(TLIST[RELAX_STEPS_sim_DT:], pos_drive_tlist[RELAX_STEPS_sim_DT:], params_dict)

    undriven_relaxation_with_noise(xilists_tsteps[:RELAX_STEPS + 1], yilists_tsteps[:RELAX_STEPS + 1], vxilists_tsteps[:RELAX_STEPS + 1],
          vyilists_tsteps[:RELAX_STEPS + 1], params_dict)
    print 'relaxation over'
    save_graph(xilists_tsteps[RELAX_STEPS], yilists_tsteps[RELAX_STEPS], TLIST[RELAX_STEPS_sim_DT], 'relaxed_state', params_dict, ERGRAPH)
    print 'relaxed graph saved'
    np.random.seed()

    # if args_dict['is_anchored']:
    verlet_pos_drive_with_noise_anchored(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], vxilists_tsteps[RELAX_STEPS:],
                    vyilists_tsteps[RELAX_STEPS:], pos_drive_tlist[RELAX_STEPS_sim_DT:], phi_tlist[RELAX_STEPS_sim_DT:],
                    wdot_tsteps[RELAX_STEPS:], qdot_tsteps[RELAX_STEPS:], params_dict)
    # else:
    #     verlet_pos_drive_with_noise_unanchored(xilists_tsteps, yilists_tsteps, vxilists_tsteps,
    #                    vyilists_tsteps, pos_drive_tlist, phi_tlist, params_dict)
    save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state',
               params_dict, ERGRAPH)
    pos_drive_tlist = np.array(np.take(pos_drive_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    phi_tlist = np.array(np.take(phi_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))

    print 'simulation done'
    save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
    print 'final graph saved'

    plot_data_dict = get_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                        TLIST, pos_drive_tlist, wdot_tsteps, qdot_tsteps, params_dict)
    print 'plot_data_dict ready'
    save_spring_transitions_data(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], TLIST[RELAX_STEPS:],
                                 params_dict, ERGRAPH, plot_data_dict)

    plot_tlist = np.take(TLIST, params_dict['PLOT_STEPS'], axis=0)
    plot_tlist /= params_dict['FORCE_PERIOD']
    plot_xilist_tsteps = np.take(xilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_yilist_tsteps = np.take(yilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vxilist_tsteps = np.take(vxilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vyilist_tsteps = np.take(vyilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_pos_drive_tlist = np.take(pos_drive_tlist, params_dict['PLOT_STEPS'], axis=0)
    # plot_phi_tlist = np.take(phi_tlist, params_dict['PLOT_STEPS'], axis=0)
    # params_dict['phi_final'] = phi_tlist[-1]

    save_sim_data(plot_data_dict, plot_xilist_tsteps, plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps,
                  plot_tlist, plot_pos_drive_tlist, params_dict, ERGRAPH)
    print 'sim data saved'
    plot_data_dict['sampled_tlist'] = plot_tlist
    try:
        save_plots(plot_data_dict, params_dict)
    except ValueError as err:
        print 'error when saving plots for %s, is: %s'%(params_dict['PLOT_DIR'], err)
    print 'plots saved'
    # plot_save_graph_pos_drive(plot_tlist, plot_xilist_tsteps, plot_yilist_tsteps, plot_pos_drive_tlist, plot_phi_tlist, params_dict, ERGRAPH)
    # print 'graph snapshots saved'
    return params_dict['PLOT_DIR']


def switch_force_pos_drive_sim_save_plots(args_dict):
    [ERGRAPH, params_dict] = initialize_params_dict(args_dict)

    if args_dict['pos_force_order']:
        fp_order_string = 'switch_pf'
    else:
        fp_order_string = 'switch_fp'
######################################## get work rate separately for each, and concatenate ###################################
    PLOT_DIR = os.path.expanduser('/Volumes/HK_data/force_pos_drive_sims/'+datetime.datetime.now().strftime("%y%m%d%H%M")+
                                    '_erg_n%d_m%d_seed1_%d_%s_force_A%.2f_pos_A%.2f_T_%sseed2_%d_soft_stiff_spring_dt_%.3f_Tend_%d_beta_1e%d'%(params_dict['NODES'],
                                                                                                    params_dict['EDGES'],
                                                                                                    params_dict['SEED1'],
                                                                                                        fp_order_string,
                                                                                              params_dict['FORCE_AMP'],
                                                                                              params_dict['POS_AMP'],
                                                                                              params_dict['T_list_string'],
                                                                                              # args_dict['beta_drive'],
                                                                                              params_dict['SEED2'],
                                                                                              args_dict['sim_dt'],
                                                                                              # params_dict['T_END'],
                                                                                              # params_dict['spring_config'],
                                                                                              # params_dict['phi_string'],
                                                                                              # params_dict['num_cycles_per_phi'],
                                                                                              # np.int(args_dict['num_drive_cycles']),
                                                                                              # np.int(params_dict['SPRING_K']),
                                                                                              int(params_dict['T_END']),
                                                                                              np.int(np.log10(args_dict['beta']))
                                                                                             # params_dict['phi_values_list'].size,
                                                                                              # _beta_1e%d_corrected int(np.log10(params_dict['BETA']))
                                                                                              ))


    PLOT_DIR = PLOT_DIR.replace('.','p')
    params_dict['PLOT_DIR'] = PLOT_DIR
    # PLOT_DIR = PLOT_DIR.replace('pos_drive_correct', 'force_drive_correct')
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    else:
        raise ValueError("PLOT_DIR:%s already exists"%(PLOT_DIR))
    params_dict['PLOT_DIR'] = PLOT_DIR
    params_dict['pos_force_order'] = args_dict['pos_force_order']

    max_deg_dist = nx.single_source_shortest_path_length(ERGRAPH, params_dict['MAX_DEG_VERTEX_I']).values()
    max_deg_dist_list = sorted(max_deg_dist)[-2:]
    max_dist_nodes = list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-1])[0])
    max_dist_nodes.extend(list(np.where(np.array(max_deg_dist)==max_deg_dist_list[-2])[0]))
    max_dist_nodes = list(set(max_dist_nodes))
    max_dist_nodes = np.array(max_dist_nodes[:2])
    params_dict['ANCHORED_NODES'] = max_dist_nodes

    # RELAX_STEPS = params_dict['RELAX_STEPS']
    TLIST = np.arange(params_dict['T_START'], params_dict['T_END'], params_dict['sim_DT'])
    SIM_STEPS = TLIST.size
    f_ext_tlist = np.zeros(SIM_STEPS)
    pos_drive_tlist = np.zeros(SIM_STEPS)
    phi_tlist = np.zeros(SIM_STEPS)
    # beta_tlist = 1e6*np.ones(SIM_STEPS)

    NSTEPS = params_dict['NSTEPS']
    # print 'true nsteps is %d, while ceiling estimate is %d'%(NSTEPS, params_dict['NSTEPS'])
    xilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    yilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vxilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    vyilists_tsteps = np.empty([NSTEPS, params_dict['NODES']])
    wdot_tsteps = np.zeros(NSTEPS)
    qdot_tsteps = np.zeros(NSTEPS)

    # init_pos = nx.spring_layout(ERGRAPH, k=1.,scale=NODES*1.)
    [xilists_tsteps[0], yilists_tsteps[0]] = init_positions(params_dict)
    vxilists_tsteps[0] = initialize_velocityvec(params_dict)
    vyilists_tsteps[0] = initialize_velocityvec(params_dict)

    save_graph(xilists_tsteps[0], yilists_tsteps[0], 0., 'init_state', params_dict, ERGRAPH)
    print 'init graph saved'
    RELAX_STEPS_sim_DT = params_dict['RELAX_STEPS']*params_dict['SIM_STEPS_PER_DT']
    SIM_STEPS_PER_DT = params_dict['SIM_STEPS_PER_DT']
    RELAX_STEPS = params_dict['RELAX_STEPS']

    get_phi_tlist(TLIST[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], phi_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], params_dict)
    # directional_temperature_drive_with_noise(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
    #                              beta_tlist, phi_tlist, params_dict)
    get_f_ext_tlist(TLIST[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], f_ext_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], params_dict)
    get_pos_tlist(TLIST[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], pos_drive_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:], params_dict)

    undriven_relaxation_with_noise(xilists_tsteps[:RELAX_STEPS + 1], yilists_tsteps[:RELAX_STEPS + 1], vxilists_tsteps[:RELAX_STEPS + 1],
          vyilists_tsteps[:RELAX_STEPS + 1], params_dict)
    print 'relaxation over'
    save_graph(xilists_tsteps[RELAX_STEPS], yilists_tsteps[RELAX_STEPS], TLIST[RELAX_STEPS_sim_DT], 'relaxed_state', params_dict, ERGRAPH)
    print 'relaxed graph saved'
    freq_switch_steps = params_dict['freq_switch_steps']
    if not params_dict['pos_force_order']:
        verlet_iterate_with_noise(xilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                                  yilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                                  vxilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                                  vyilists_tsteps[RELAX_STEPS:RELAX_STEPS+freq_switch_steps[0]],
                                  f_ext_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:
                                              (RELAX_STEPS+freq_switch_steps[0])*SIM_STEPS_PER_DT],
                                  wdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
                                  qdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
                                  params_dict)
    else:
        verlet_pos_drive_with_noise_anchored(
            xilists_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            yilists_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            vxilists_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            vyilists_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            pos_drive_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:
                          (RELAX_STEPS+freq_switch_steps[0])*SIM_STEPS_PER_DT],
            phi_tlist[(RELAX_STEPS+1)*SIM_STEPS_PER_DT:
                      (RELAX_STEPS+freq_switch_steps[0])*SIM_STEPS_PER_DT],
            wdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            qdot_tsteps[RELAX_STEPS:RELAX_STEPS + freq_switch_steps[0]],
            params_dict)

    for i in range(1, len(freq_switch_steps)):
        if (i%2 != params_dict['pos_force_order']):
            verlet_pos_drive_with_noise_anchored(
                xilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                yilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                vxilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                vyilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                pos_drive_tlist[(RELAX_STEPS+freq_switch_steps[i-1])*SIM_STEPS_PER_DT:
                                (RELAX_STEPS+freq_switch_steps[i])*SIM_STEPS_PER_DT],
                phi_tlist[(RELAX_STEPS+freq_switch_steps[i-1])*SIM_STEPS_PER_DT:
                                (RELAX_STEPS+freq_switch_steps[i])*SIM_STEPS_PER_DT],
                wdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                qdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                params_dict)
        else:
            verlet_iterate_with_noise(
                xilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                yilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                vxilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                vyilists_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                f_ext_tlist[(RELAX_STEPS+freq_switch_steps[i-1])*SIM_STEPS_PER_DT:
                                (RELAX_STEPS+freq_switch_steps[i])*SIM_STEPS_PER_DT],
                wdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                qdot_tsteps[RELAX_STEPS+freq_switch_steps[i-1]-1:RELAX_STEPS+freq_switch_steps[i]],
                params_dict)

    f_ext_tlist = np.array(np.take(f_ext_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    pos_drive_tlist = np.array(np.take(pos_drive_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    phi_tlist = np.array(np.take(phi_tlist, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    TLIST = np.array(np.take(TLIST, np.arange(0, SIM_STEPS, np.int(params_dict['SIM_STEPS_PER_DT']))))
    print 'simulation done'
    save_graph(xilists_tsteps[-1], yilists_tsteps[-1], TLIST[-1], 'final_state', params_dict, ERGRAPH)
    print 'final graph saved'

    plot_data_dict = get_plot_data_dict(xilists_tsteps, yilists_tsteps, vxilists_tsteps, vyilists_tsteps,
                                        TLIST, f_ext_tlist, wdot_tsteps, qdot_tsteps, params_dict)
    print 'plot_data_dict ready'
    save_spring_transitions_data(xilists_tsteps[RELAX_STEPS:], yilists_tsteps[RELAX_STEPS:], TLIST[RELAX_STEPS:],
                                 params_dict, ERGRAPH, plot_data_dict)

    plot_tlist = np.take(TLIST, params_dict['PLOT_STEPS'], axis=0)
    plot_tlist /= params_dict['FORCE_PERIOD']
    plot_xilist_tsteps = np.take(xilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_yilist_tsteps = np.take(yilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vxilist_tsteps = np.take(vxilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_vyilist_tsteps = np.take(vyilists_tsteps, params_dict['PLOT_STEPS'], axis=0)
    plot_f_ext_tlist = np.take(f_ext_tlist, params_dict['PLOT_STEPS'], axis=0)
    plot_phi_tlist = np.take(phi_tlist, params_dict['PLOT_STEPS'], axis=0)
    plot_pos_drive_tlist = np.take(pos_drive_tlist, params_dict['PLOT_STEPS'], axis=0)

    plot_data_dict['sampled_tlist'] = plot_tlist
    dfn = os.path.join(params_dict['PLOT_DIR'], 'sampled_pos_drive_tlist.json')
    with open(dfn, 'w') as f:
        json.dump(plot_pos_drive_tlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    # params_dict['phi_final'] = phi_tlist[-1]
    save_sim_data(plot_data_dict, plot_xilist_tsteps, plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps,
                  plot_tlist, plot_f_ext_tlist, params_dict, ERGRAPH)
    print 'sim data saved'
    save_plots(plot_data_dict, params_dict)
    print 'plots saved'
    # check_equilibration(plot_vxilist_tsteps, plot_vyilist_tsteps, params_dict)
    # f_ext_tlist *= params_dict['force_scale_factor']
    # plot_save_graph(plot_tlist, plot_xilist_tsteps, plot_yilist_tsteps, plot_f_ext_tlist, plot_phi_tlist, params_dict, ERGRAPH)
    print 'graph snapshots saved'
    return params_dict['PLOT_DIR']


def initialize_params_dict(args_dict):
    params_dict = {
        #smalled iteration time
        'sim_DT': args_dict['sim_dt'],
        'SIM_STEPS_PER_DT': args_dict['sim_steps_per_dt'],
        'T_START': 0.,
        #relax time
        'T_RELAX': 200.,
        #amp of the interaction between nodes
        'SPRING_K': args_dict['spring_k'],
        #constrain spring constant
        'SPRING_K0': args_dict['spring_k0'],
        'MASS': 1.,
        'DAMPING_RATE': args_dict['damping_rate'],
        'FORCE_STRING': args_dict['force_string'], #'step', 'sine','square','pulse'
        'FORCE_AMP': args_dict['force_amp'],
        'POS_AMP': args_dict['pos_amp'],
        'FORCE_PHASE': args_dict['force_phase'],
        #period of the driven force
        'FORCE_PERIOD': args_dict['force_period'],
        'L1': 1.,
        'Lbarrier': 2.,
        'L2': 3.,
        'K1': 0.5,
        'K2': 5.,
        'B': 1.,
        'OFFSET': args_dict['offset'],
        #for smallest distance between nodes
        'ERR': 1E-12,
        #num of nodes/edges
        'NODES': args_dict['nodes'],
        'EDGES': args_dict['edges'],
        #seed1 graph config, seed2 is the initialization
        'SEED1': args_dict['seed1'],
        'SEED2': args_dict['seed2'],
        #plot parameter, for force plotting 20 points each force period
        # 'num_samples_per_period': 20,
        'spring_config': args_dict['spring_config'],
        'phi_string': args_dict['phi_string'],
        'num_cycles_per_phi': args_dict['num_cycles_per_phi'],
        'phi_values_list': args_dict['phi_values_list'],
        'BETA': args_dict['beta'],
        'beta_drive': args_dict['beta_drive'],
        'T_list': np.array(args_dict['T_list']),
        'num_cycles_list': np.array(args_dict['num_cycles_list']),
        'freq_switch_times': args_dict['freq_switch_times'],
        'num_anchors':args_dict['num_anchors'],
    }

    params_dict['T_END'] = params_dict['freq_switch_times'][-1] + params_dict['T_RELAX']
    params_dict['T_list_string'] = ''.join([str(Ti)+'_' for Ti in params_dict['T_list']])

    params_dict['phi_list_string'] = ''.join([str('%.1f'%(phi)).replace('.','p') + '_' for phi in params_dict['phi_values_list']])

    params_dict['DT'] = params_dict['SIM_STEPS_PER_DT']*params_dict['sim_DT']
    params_dict['RELAX_STEPS'] = np.int(np.ceil(params_dict['T_RELAX']/params_dict['DT']))
    params_dict['freq_switch_steps'] = [ np.int(np.ceil(
                                (params_dict['freq_switch_times'][i] - params_dict['T_START'])/params_dict['DT']))
                                for i in range(len(params_dict['freq_switch_times']))]
    params_dict['NSTEPS'] = np.int(np.ceil((params_dict['T_END']-params_dict['T_START'])/params_dict['DT']))

    #to cut the initialization
    RGRAPH = nx.gnm_random_graph(20,50)
    params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX'] / (2. * params_dict['FORCE_PERIOD']))
    SEED1 = args_dict['seed1']
    random.seed(SEED1)  # controls graph generation
    # create  a connected graph
    if (args_dict['graph_type'] == 'erg'):
        RGRAPH = nx.gnm_random_graph(args_dict['nodes'], args_dict['edges'])
        while not nx.is_connected(RGRAPH):
            SEED1 = random.randint(0, 2 ** 32 - 1)
            random.seed(SEED1)
            RGRAPH = nx.gnm_random_graph(args_dict['nodes'], args_dict['edges'])
    elif (args_dict['graph_type'] == 'bag'):
        RGRAPH = nx.barabasi_albert_graph(args_dict['nodes'], args_dict['bag_m'])
    elif (args_dict['graph_type'] == 'wsg'):
        RGRAPH = nx.connected_watts_strogatz_graph(args_dict['nodes'], args_dict['wsg_k'], args_dict['wsg_p'])

    np.random.seed(args_dict['seed2']) #controls initial positions, velocities
    AMAT = nx.adjacency_matrix(RGRAPH).toarray()
    SORT_DEG = sorted(list(dict(nx.degree(RGRAPH)).values()))
    #for max deg
    MAX_DEG_VERTEX_I = int(list(dict(nx.degree(RGRAPH)).values()).index(SORT_DEG[-1]))
    params_dict['AMAT'] = AMAT
    params_dict['MAX_DEG_VERTEX_I'] = MAX_DEG_VERTEX_I

    ########## assume fixed phi, i.e. fixed driving direction ##########
    unperturbed_phi = params_dict['phi_values_list'][0]
    forcing_direction = np.zeros(2*params_dict['NODES'])
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']] = np.cos(unperturbed_phi)
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']+1] = np.sin(unperturbed_phi)
    params_dict['FORCING_VEC'] = forcing_direction

    PLOT_DIR = os.path.expanduser('/Volumes/bistable_data/force_drive_switch_freq_sims_new/'+datetime.datetime.now().strftime("%y%m%d%H%M")+
                                  '_%s_n%d_m%d_seed1_%d_F_max_deg_A%.2f_T_%s_phi_%s_seed2_%d_soft_stiff_spring_dt_%.3f_Tend_%d_beta_1e%d' % (
                                                                                                    args_dict['graph_type'],
                                                                                                    params_dict['NODES'],
                                                                                                    params_dict['EDGES'],
                                                                                                    params_dict['SEED1'],
                                                                                              # params_dict['FORCE_STRING'],
                                                                                              params_dict['FORCE_AMP'],
                                                                                              params_dict['T_list_string'],
                                                                                              # args_dict['beta_drive'],
                                                                                              params_dict['phi_list_string'],
                                                                                              # params_dict['num_cycles_per_phi'],
                                                                                              # params_dict['phi_string'],
                                                                                              params_dict['SEED2'],
                                                                                              args_dict['sim_dt'],
                                                                                              # params_dict['T_END'],
                                                                                              # params_dict['spring_config'],
                                                                                              # params_dict['num_cycles_per_phi'],
                                                                                              # np.int(args_dict['num_drive_cycles']),
                                                                                              # params_dict['SPRING_K'],
                                                                                              int(params_dict['T_END']),
                                                                                              # args_dict['overdamped_gamma'], overdamped_%d_
                                                                                              np.int(np.log10(args_dict['beta'])),
                                                                                              # args_dict['permute_string'] ######### only for permuted drive ensemble ########
                                                                                              # args_dict['rep'],
                                                                                             # params_dict['phi_values_list'].size,
                                                                                              # _beta_1e%d_corrected int(np.log10(params_dict['BETA']))
                                                                                              ))
    #Save data for every PLOT_DELTA_STEPS
    PLOT_DELTA_STEPS = args_dict['plot_delta_steps']
    params_dict['PLOT_DELTA_STEPS'] = PLOT_DELTA_STEPS
    params_dict['PLOT_DIR'] = PLOT_DIR
    #pick out the values for each PLOT_DELTA_STEPS
    params_dict['PLOT_STEPS'] = np.arange(np.int(params_dict['RELAX_STEPS']/2.), params_dict['NSTEPS'], PLOT_DELTA_STEPS)
    #for saving snapshots every SAVE_FIG_DELTA_FRAMES*PLOT_DELTA_STEPS
    params_dict['SAVE_FIG_DELTA_FRAMES'] = 200 ########### CHANGE WHEN RUNNING SHORTER SIMULATIONS
    params_dict['COLORS'] = ['peru','olive']
    return [RGRAPH, params_dict]

def resave_graph(xilist, yilist, t, save_str, params_dict, ergraph):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('grey')
    colors[params_dict['MAX_DEG_VERTEX_I']] = 'maroon'
    for node in params_dict['ANCHORED_NODES']:
        colors[node] = 'sandybrown'
    xmin = np.amin(xilist)
    xmax = np.amax(xilist)
    ymin = np.amin(yilist)
    ymax = np.amax(yilist)
    fsize = 10
    dpi = 400

    for node in range(params_dict['NODES']):
        pos[node] = (xilist[node], yilist[node])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fsize, fsize), dpi=dpi)
    nx.draw_networkx(ergraph, pos=pos, node_color=colors, node_size=500, width=3., with_labels=False, ax=ax)
    ax.set_aspect('equal')
    ax.set(ylim=[ymin - 0.1, ymax + 0.1], xlim=[xmin - 0.1, xmax + 0.1],
           aspect=(xmax - xmin + 0.2) / (ymax - ymin + 0.2))
    prefix = save_str
    plot_fn = os.path.join(params_dict['PLOT_DIR'], prefix + '_resaved.png')
    fig.savefig(plot_fn, dpi=dpi)
    plt.close()



if __name__ == "__main__":
    args_dict = {
                'nodes': 20,
                'edges': 50,
                'force_string': 'sine',
                #constrain spring constant
                'graph_type': 'erg',
                'bag_m':2,
                'wsg_k':4,
                'wsg_p':0.5,
                'spring_k0': 0.05,
                'pos_force_order': False,
                'phi_string': 'random',
                'num_cycles_per_phi': 600,
                'phi_values_list':[np.pi/4],
                'T_list': [2.1],
                'num_cycles_list': [600],
                'freq_switch_times': [1260.],
                'force_amp': 5.5,
                'pos_amp': 0.2,
                'force_period': 2.0,
                # 'force_dirn': np.pi/4.,
                # 'num_drive_cycles': 1,#500,
                'num_anchors':3,
                'force_phase': 0.,
                'damping_rate': 0.1,
                'overdamped_gamma':30.,
                'steps_per_beta':1000,
                'step_size_factor':1.,
                'beta': 1e4,
                'sim_dt': 0.001,
                'sim_steps_per_dt': 25,
                'plot_delta_steps': 4,
                'spring_k': 2.,
                'lbarrier': 1.9,
                'offset': 0.06,
                'spring_config': 'right_low_E_stiffer',
                'beta_drive': 0.5,
                # 'hot_degree': 4,
                'hot_beta': 0.5,
                'cold_beta': 1e2,
                'hot_damping_rate': 0.1,
                'cold_damping_rate': 0.2,
                'seed1': random.randint(0, 2**32-1),
                'seed2': random.randint(1, 2**32),
                }

    # run_pos_drive_sim_save_plots(args_dict)
    # run_temperature_sim_save_plots(args_dict)
    # run_force_sim_save_plots(args_dict)
    # run_pos_drive_sim_save_plots(args_dict)
    # period_list = [2., 2.]  # [ 4.5,  2., 1. ]
    # num_cycles_list = [600, 600]  # [1800, 5000, 8000 ]
    # args_dict['T_list'] = period_list
    # args_dict['num_cycles_list'] = num_cycles_list
    # switch_force_pos_drive_sim_save_plots(args_dict)

    # period_list = [ 6.28, 2.1, 1.26, 0.9]
    # period_list = [2., 1.2]  # [ 4.5,  2., 1. ]
    # num_cycles_list = [600, 1000]  # [1800, 5000, 8000 ]
    # args_dict['T_list'] = period_list
    # args_dict['num_cycles_list'] = num_cycles_list
    # run_force_sim_save_plots(args_dict)


###### RUNNING SWITCH FREQ SIMS FOR DIFFERENT SET OF FREQ.S ######
    num_graphs = 2#120
    period_list = np.array([3.14, 1.57, 1.047])  # np.array([6.28,  0.9, 1.26, 2.1, ]) #[ 4.5,  2., 1. ]
    args_dict['sim_steps_per_dt'] = 25
    args_dict['plot_delta_steps'] = 4
    args_dict['nodes'] = 20
    args_dict['edges'] = 50
    args_dict['force_amp'] = 5.5
    args_dict['T_list'] = period_list
    num_cycles_list = np.array([350])
    drive_duration_list = np.array(period_list)*np.array(num_cycles_list)
    freq_switch_times = [sum(drive_duration_list[:i + 1]) for i in range(len(period_list))]
    args_dict['freq_switch_times'] = freq_switch_times
    args_dict['num_cycles_list'] = num_cycles_list

    args_dict_list = []

    for graph in range(num_graphs):
        args_dict['seed1'] = random.randint(0, 2 ** 32 - 1)
        # for n_iter in range(num_iter):
        args_dict['seed2'] = np.random.randint(1, 2 ** 32)
        args_dict['phi_values_list'] = 2 * np.pi * np.random.random_sample(1)
        args_dict_list.append(dict(args_dict))

    p = multiprocessing.Pool(processes=2)
    for result in p.imap_unordered(run_force_sim_save_plots, args_dict_list):
        print '%s is ready' % (result)
    p.close()
    p.join()
