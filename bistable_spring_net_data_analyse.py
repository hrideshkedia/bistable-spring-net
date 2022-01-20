import matplotlib
matplotlib.use('Agg')
from bistable_spring_net_fns import *
from bistable_spring_net import *
from networkx.readwrite import json_graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from scipy import linalg
import shutil

plot_params = {
    'axes.linewidth': 0.5,
    'axes.labelsize': 16,
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
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_node_distances(erg, params_dict):
    node_dist_dict = nx.single_source_shortest_path_length(erg, params_dict['MAX_DEG_VERTEX_I'])
    return node_dist_dict

def get_spring_distances(RGRAPH, params_dict, spring_distances):
    nodes = params_dict['NODES']
    Amat = np.array(params_dict['AMAT'])
    node_dist_dict = get_node_distances(RGRAPH, params_dict)
    (i_nonzero, j_nonzero) = np.nonzero(Amat)
    num_nonzero = i_nonzero.shape[0]

    edgeIndex = 0
    for i in range(num_nonzero):
        if i_nonzero[i]>j_nonzero[i]:
            spring_distances[edgeIndex] = min(node_dist_dict[i_nonzero[i]],node_dist_dict[j_nonzero[i]])
            edgeIndex +=1

def get_spring_degrees(RGRAPH, params_dict, spring_degrees):
    nodes = params_dict['NODES']
    Amat = np.array(params_dict['AMAT'])
    (i_nonzero, j_nonzero) = np.nonzero(Amat)
    num_nonzero = i_nonzero.shape[0]

    edgeIndex = 0
    for i in range(num_nonzero):
        if i_nonzero[i]>j_nonzero[i]:
            spring_degrees[edgeIndex] = min(nx.degree(RGRAPH, i_nonzero[i]), nx.degree(RGRAPH, j_nonzero[i]))
            edgeIndex +=1


def read_sim_data_check_equilibration(PLOT_DIR, is_hot_cold=False):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn,'r') as f:
        params_dict = json.load(f)

    params_dict['PLOT_DIR'] = PLOT_DIR

    dfn = os.path.join(PLOT_DIR, 'sampled_vxilist.json')
    with open(dfn,'r') as f:
        plot_vxilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vyilist.json')
    with open(dfn,'r') as f:
        plot_vyilist_tsteps = np.array(json.load(f))

    dfn = os.path.join(PLOT_DIR, 'adjacency_data.json')
    with open(dfn,'r') as f:
        adj_data = json.load(f)
    erg = json_graph.adjacency_graph(adj_data)

    if not is_hot_cold:
        check_equilibration(plot_vxilist_tsteps, plot_vyilist_tsteps, params_dict)
    else:
        check_equilibration_hot_cold(plot_vxilist_tsteps,plot_vyilist_tsteps, erg, params_dict)
        check_node_T_distribution(plot_vxilist_tsteps,plot_vyilist_tsteps, params_dict)

    return PLOT_DIR

def read_sim_data_check_beta_dist(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn,'r') as f:
        params_dict = json.load(f)

    params_dict['PLOT_DIR'] = PLOT_DIR

    dfn = os.path.join(PLOT_DIR, 'sampled_vxilist.json')
    with open(dfn,'r') as f:
        plot_vxilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vyilist.json')
    with open(dfn,'r') as f:
        plot_vyilist_tsteps = np.array(json.load(f))

    check_node_beta_distribution(plot_vxilist_tsteps,plot_vyilist_tsteps, params_dict)

    return PLOT_DIR


def read_sim_data_save_plots(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn,'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()

    keys_list = ['t/T', 'tlist_sampled',
                 '<# long springs(t)>_T', '<distance from relaxed state(t)>_T',
                '<# spring transitions(t)>_T', '<U(t)/(Barrier)>_T', '<KE(t)/(Barrier)>_T',
                '<KE(t)_{c.o.m.}/KE>_T', '<E(t)/(Barrier)>_T', '<F(t).v(t)/(Barrier/T)>_T',
                '<dissipation_rate(t)/(Barrier/T)>_T', '<(Edot + dissipation_rate - work_rate)/(Barrier/T)>_T',
                 'eig_hist_tsteps', 'eig_bins', 'forcing_overlap_tsteps', 'forcing_perp_overlap_tsteps',
                 'num_unique_states_tsteps', 'max_diverging_rates_sampled'
                 'num_diverging_dims_sampled', 'spring_flips_tlist', 'flip_times',
                 'spring_flips_tsteps_by_distance', 'dwell_max_diverging_rates', 'dwell_times',
                 'dwell_diverging_dims', 'dwell_diss_rate_lr',
                 ]
    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
    params_dict['PLOT_DIR'] = PLOT_DIR

    save_plots(plot_data_dict, params_dict)

    return PLOT_DIR

def read_sim_data_save_dwell_plot(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()

    keys_list = ['t/T', '<dissipation_rate(t)/(Barrier/T)>_T', 'dwell_times', 'flip_times', 'dwell_diss_rate_lr']

    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
    params_dict['PLOT_DIR'] = PLOT_DIR

    start_index = params_dict['START_PLOT_INDEX']
    drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']

    fig = plt.figure(figsize=(12, 12), dpi=100)
    gs = gridspec.GridSpec(2, 2)
    axarr = [   [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])],
                [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])] ]

    axarr[0][0].plot(plot_data_dict['dwell_diss_rate_lr'], plot_data_dict['dwell_times'],
                    linestyle=':', c=params_dict['COLORS'][0], marker='o', markerfacecolor=params_dict['COLORS'][0],
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][0].set_yscale('log')
    axarr[0][0].set_xscale('log')
    axarr[0][0].set(xlabel=r"avg dissipation rate (linear response)",ylabel=r"dwell time")
    axarr[0][0].margins(0.05)

    flip_times = plot_data_dict['flip_times']
    if flip_times.size != plot_data_dict['dwell_times'].size:
        flip_times = flip_times[1:]

    axarr[1][0].plot(flip_times, plot_data_dict['dwell_times'],
                    linestyle=':', c=params_dict['COLORS'][0], marker='o', markerfacecolor=params_dict['COLORS'][0],
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[1][0].set_yscale('log')
    axarr[1][0].set(xlabel=r"flip time",ylabel=r"dwell time")
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][0].axvline(x=freq_switch_t, c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][0].margins(0.05)

    axarr[0][1].plot(flip_times, plot_data_dict['dwell_diss_rate_lr'],
                    linestyle=':', c=params_dict['COLORS'][0], marker='o', markerfacecolor=params_dict['COLORS'][0],
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][1].set_yscale('log')
    axarr[0][1].set(xlabel=r"flip time",ylabel=r"avg dissipation rate (linear response)")
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][1].axvline(x=freq_switch_t, c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][1].margins(0.05)

    axarr[1][1].plot(plot_data_dict['t/T'][start_index:],
                     plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'][start_index:], linestyle=':',
                     c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][1].set(ylabel=r"$\langle\mathrm{Dissipation\,rate}(t)/(\mathrm{Barrier}/T)\rangle_T$")
    axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][1].axvline(x=freq_switch_t/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][1].margins(0.05)

    plt.tight_layout()
    fig.savefig(os.path.join(params_dict['PLOT_DIR'], 'dwell_time_vs_diss_rate_lr.png'))
    plt.close()

def read_sim_data_save_ang_momentum(PLOT_DIR):
    # print 'plot_dir is %s' %(PLOT_DIR)
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vxilist.json')
    with open(dfn, 'r') as f:
        plot_vxilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vyilist.json')
    with open(dfn, 'r') as f:
        plot_vyilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    with open(dfn, 'r') as f:
        plot_tlist = np.array(json.load(f))

    Amat = np.array(params_dict['AMAT'])
    tlength = plot_xilist_tsteps.shape[0]
    L_anchor_sampled = np.empty(tlength)
    # print 'loaded data'
    # print 'shape of vxilist is %d, %d, size of L is %d'%(plot_vxilist_tsteps.shape[0], plot_vxilist_tsteps.shape[1], tlength)
    get_driven_angular_momentum(plot_xilist_tsteps, plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps,
                                params_dict, L_anchor_sampled)
    # print 'calculated angular momentum'
    avg_block_size = int(np.ceil(params_dict['FORCE_PERIOD']/params_dict['DT'])/params_dict['PLOT_DELTA_STEPS'])

    L_anchor_block_avg = np.empty(int(np.ceil(tlength/avg_block_size)))
    avg_sampled_tlist = np.empty(int(np.ceil(tlength/avg_block_size)))

    get_scaled_block_avg(L_anchor_sampled, L_anchor_block_avg, avg_block_size, 1.)
    get_scaled_block_avg(plot_tlist, avg_sampled_tlist, avg_block_size, 1.)

    # print 'calculated scaled block avgs'
    dfn = os.path.join(params_dict['PLOT_DIR'], 'L_anchor_avg.json')
    with open(dfn, 'w') as f:
        json.dump(L_anchor_block_avg, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'L_anchor_sampled.json')
    with open(dfn, 'w') as f:
        json.dump(L_anchor_sampled, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'avg_sampled_tlist.json')
    with open(dfn, 'w') as f:
        json.dump(avg_sampled_tlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))


def read_sim_data_save_allostery_force_transmission_measures(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    # dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    # with open(dfn, 'r') as f:
    #     plot_tlist = np.array(json.load(f))

    Amat = np.array(params_dict['AMAT'])
    omega_plot = np.array(params_dict['omega_plot'])
    params_dict['PLOT_DIR'] = PLOT_DIR
    plot_data_dict = dict()

    drive_start_index = int(params_dict['T_RELAX']/(2*params_dict['DT']*params_dict['PLOT_DELTA_STEPS']))
    allostery_measure_start = np.zeros(omega_plot.size)
    force_transmission_start = np.zeros(omega_plot.size)

    allostery_measure_end = np.zeros(omega_plot.size)
    force_transmission_end = np.zeros(omega_plot.size)

    get_allostery_force_transmission_measure(plot_xilist_tsteps[drive_start_index],
                                             plot_yilist_tsteps[drive_start_index],
                                             omega_plot, params_dict, Amat, allostery_measure_start,
                                             force_transmission_start, 'drive_start_')

    get_allostery_force_transmission_measure(plot_xilist_tsteps[-1], plot_yilist_tsteps[-1],
                                             omega_plot, params_dict, Amat, allostery_measure_end,
                                             force_transmission_end, 'drive_end_')

    plot_data_dict['allostery_measure_drive_start'] = allostery_measure_start
    plot_data_dict['force_transmission_measure_drive_start'] = force_transmission_start

    plot_data_dict['allostery_measure_drive_end'] = allostery_measure_end
    plot_data_dict['force_transmission_measure_drive_end'] = force_transmission_end

    dfn = os.path.join(params_dict['PLOT_DIR'], 'params_dict.json')
    with open(dfn, 'w') as f:
        json.dump(params_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    for k,v in plot_data_dict.iteritems():
        dfn = os.path.join(params_dict['PLOT_DIR'], str(k).replace('/','_by_')+'.json')
        with open(dfn, 'w') as f:
            json.dump(v, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    fig = plt.figure(figsize=(10, 10), dpi=100)
    gs = gridspec.GridSpec(2, 2)
    axarr = [  [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])],
               [plt.subplot(gs[1, 0]), plt.subplot(gs[1,1]) ] ]

    axarr[0][0].plot(omega_plot, allostery_measure_start+1E-6,
                    linestyle=':', c='cadetblue', marker='o', markerfacecolor='cadetblue',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][0].set(xlabel=r"$\omega$",ylabel=r"$\frac{1}{N^2}\sum_{i\neq j}\vert \langle i\vert G(\omega)\vert j\rangle\vert^2$ at start")
    axarr[0][0].axvline(x=2*np.pi/params_dict['T_list'][0], c=params_dict['COLORS'][1], linestyle='-.')
    axarr[0][0].set_yscale('log')
    axarr[0][0].margins(0.05)

    axarr[1][0].plot(omega_plot, allostery_measure_end+1E-6,
                    linestyle=':', c='cadetblue', marker='o', markerfacecolor='cadetblue',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[1][0].set(xlabel=r"$\omega$",ylabel=r"$\frac{1}{N^2}\sum_{i\neq j}\vert \langle i\vert G(\omega)\vert j\rangle\vert^2$ at end")
    axarr[1][0].axvline(x=2 * np.pi / params_dict['T_list'][0], c=params_dict['COLORS'][1], linestyle='-.')
    axarr[1][0].set_yscale('log')
    axarr[1][0].margins(0.05)

    ymin0, ymax0 = axarr[0][0].get_ylim()
    ymin1, ymax1 = axarr[1][0].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)
    axarr[0][0].set_ylim([ymin, ymax])
    axarr[1][0].set_ylim([ymin, ymax])

    axarr[0][1].plot(omega_plot, force_transmission_start+1E-6,
                    linestyle=':', c='cadetblue', marker='o', markerfacecolor='cadetblue',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][1].set(xlabel=r"$\omega$",
        ylabel=r"$\frac{1}{N}\sum_{j\neq i}\vert \langle j\vert G(\omega)\vert i_\mathrm{driven}\rangle\vert^2$ at start")
    axarr[0][1].axvline(x=2 * np.pi / params_dict['T_list'][0], c=params_dict['COLORS'][1], linestyle='-.')
    axarr[0][1].set_yscale('log')
    axarr[0][1].margins(0.05)

    axarr[1][1].plot(omega_plot, force_transmission_end+1E-6,
                    linestyle=':', c='cadetblue', marker='o', markerfacecolor='cadetblue',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[1][1].set(xlabel=r"$\omega$",
        ylabel=r"$\frac{1}{N}\sum_{j\neq i}\vert \langle j\vert G(\omega)\vert i_\mathrm{driven}\rangle\vert^2$ at end")
    axarr[1][1].axvline(x=2 * np.pi / params_dict['T_list'][0], c=params_dict['COLORS'][1], linestyle='-.')
    axarr[1][1].set_yscale('log')
    axarr[1][1].margins(0.05)

    ymin0, ymax0 = axarr[0][1].get_ylim()
    ymin1, ymax1 = axarr[1][1].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)
    axarr[0][1].set_ylim([ymin, ymax])
    axarr[1][1].set_ylim([ymin, ymax])

    plt.tight_layout()
    fig.savefig(os.path.join(params_dict['PLOT_DIR'], 'allostery_measures.png'))
    plt.close()

    return params_dict['PLOT_DIR']

def read_sim_data_save_contraction_plot(PLOT_DIR, tol=1e-8):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    with open(dfn, 'r') as f:
        plot_tlist = np.array(json.load(f))

    Amat = np.array(params_dict['AMAT'])

    num_contracting_dims = np.empty(plot_tlist.size)
    num_stationary_dims = np.empty(plot_tlist.size)
    num_diverging_dims = np.empty(plot_tlist.size)

    get_num_phase_space_contracting_dims(plot_xilist_tsteps, plot_yilist_tsteps, params_dict, Amat, tol, num_contracting_dims,
                                           num_stationary_dims, num_diverging_dims)

    avg_block_size = int((params_dict['FORCE_PERIOD']/params_dict['DT'])/params_dict['PLOT_DELTA_STEPS'])
    num_contracting_dims_block_avg = np.empty(int(plot_tlist.size/avg_block_size))
    num_stationary_dims_block_avg = np.empty(int(plot_tlist.size / avg_block_size))
    num_diverging_dims_block_avg = np.empty(int(plot_tlist.size / avg_block_size))
    contraction_tlist = np.empty(int(plot_tlist.size / avg_block_size))

    # get_scaled_block_avg(data_array, avg_array, block_size, scale_factor)
    get_scaled_block_avg(num_contracting_dims, num_contracting_dims_block_avg, avg_block_size, 1.)
    get_scaled_block_avg(num_stationary_dims, num_stationary_dims_block_avg, avg_block_size, 1.)
    get_scaled_block_avg(num_diverging_dims, num_diverging_dims_block_avg, avg_block_size, 1.)
    get_scaled_block_avg(plot_tlist, contraction_tlist, avg_block_size, 1.)

    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_contracting_dims_sampled.json')
    with open(dfn, 'w') as f:
        json.dump(num_contracting_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_stationary_dims_sampled.json')
    with open(dfn, 'w') as f:
        json.dump(num_stationary_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_diverging_dims_sampled.json')
    with open(dfn, 'w') as f:
        json.dump(num_diverging_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
#####################
    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_contracting_dims_T_avg.json')
    with open(dfn, 'w') as f:
        json.dump(num_contracting_dims_block_avg, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_stationary_dims_T_avg.json')
    with open(dfn, 'w') as f:
        json.dump(num_stationary_dims_block_avg, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'num_diverging_dims_T_avg.json')
    with open(dfn, 'w') as f:
        json.dump(num_diverging_dims_block_avg, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'contraction_tlist.json')
    with open(dfn, 'w') as f:
        json.dump(contraction_tlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    fig = plt.figure(figsize=(15, 5), dpi=100)
    gs = gridspec.GridSpec(1, 3)
    axarr = [  [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]) ] ]

    axarr[0][0].plot(contraction_tlist, num_contracting_dims_block_avg,
                    linestyle=':', c='cadetblue', marker='o', markerfacecolor='cadetblue',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][0].set(xlabel=r"t/T",ylabel=r"# contracting dim.s")
    axarr[0][0].margins(0.05)

    axarr[0][1].plot(contraction_tlist, num_stationary_dims_block_avg,
                    linestyle=':', c='olivedrab', marker='o', markerfacecolor='olivedrab',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][1].set(xlabel=r"t/T",ylabel=r"# stationary dim.s")
    axarr[0][1].margins(0.05)

    axarr[0][2].plot(contraction_tlist, num_diverging_dims_block_avg,
                    linestyle=':', c='indianred', marker='o', markerfacecolor='indianred',
                    markeredgecolor='None', markersize=5, alpha = 0.5)
    axarr[0][2].set(xlabel=r"t/T",ylabel=r"# diverging dim.s")
    axarr[0][2].margins(0.05)

    plt.tight_layout()
    fig.savefig(os.path.join(params_dict['PLOT_DIR'], 'contraction_plot.png'))
    plt.close()

    plot_data_dict = dict()

    keys_list = [ 't/T', 'dwell_times', 'flip_times']

    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            plot_data_dict[k] = np.array(json.load(f))

    flip_times = plot_data_dict['flip_times'] / params_dict['FORCE_PERIOD']
    # dwell_times = plot_data_dict['dwell_times'] / params_dict['FORCE_PERIOD']

    if flip_times.size != plot_data_dict['dwell_times'].size:
        start_dwell = flip_times[0]
        flip_times = flip_times[1:]
    else:
        start_dwell = params_dict['T_RELAX']

    dwell_contracting_dims = []
    dwell_stationary_dims = []
    dwell_diverging_dims = []

    for i, flip_time in enumerate(flip_times):
        start_dwell_idx = find_nearest_idx(plot_tlist, start_dwell)
        end_dwell = flip_time
        end_dwell_idx = find_nearest_idx(plot_tlist, end_dwell)

        if end_dwell_idx > plot_tlist.size - 1:
            end_dwell_idx = plot_tlist.size - 1
        elif end_dwell_idx == start_dwell_idx:
            end_dwell_idx += 1

        dwell_contracting_dims.append(np.sum(num_contracting_dims[start_dwell_idx:end_dwell_idx])/(end_dwell_idx-start_dwell_idx+1e-2))
        dwell_stationary_dims.append(
            np.sum(num_stationary_dims[start_dwell_idx:end_dwell_idx])/(end_dwell_idx - start_dwell_idx + 1e-2))
        dwell_diverging_dims.append(
            np.sum(num_diverging_dims[start_dwell_idx:end_dwell_idx])/(end_dwell_idx - start_dwell_idx + 1e-2))

        start_dwell = flip_time

    dfn = os.path.join(params_dict['PLOT_DIR'], 'dwell_contracting_dims.json')
    with open(dfn, 'w') as f:
        json.dump(dwell_contracting_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'dwell_stationary_dims.json')
    with open(dfn, 'w') as f:
        json.dump(dwell_stationary_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'dwell_diverging_dims.json')
    with open(dfn, 'w') as f:
        json.dump(dwell_diverging_dims, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    ########## save dwell contraction data on reading, in get_plot_data_dict, and  in agg_plot ###########

def read_sim_data_save_dwell_diss_rate(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()

    keys_list = [ 't/T', '<dissipation_rate(t)/(Barrier/T)>_T', 'dwell_times', 'flip_times', 'dwell_diss_rate_lr',
                '<dissipation_rate_lr(t)/(Barrier/T)>_T', ]

    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
    params_dict['PLOT_DIR'] = PLOT_DIR

    flip_times = plot_data_dict['flip_times'] / params_dict['FORCE_PERIOD']
    # dwell_times = plot_data_dict['dwell_times'] / params_dict['FORCE_PERIOD']

    if flip_times.size != plot_data_dict['dwell_times'].size:
        start_dwell = flip_times[0]
        flip_times = flip_times[1:]
    else:
        start_dwell = params_dict['T_RELAX']

    raw_diss_rate = plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T']
    smooth_diss_rate_lr = plot_data_dict['<dissipation_rate_lr(t)/(Barrier/T)>_T']
    dwell_diss_rate_actual = []
    dwell_diss_rate_lr = []
    # long_dwell_diss_rate_actual = []
    # long_dwell_diss_rate_lr_smooth = []

    for i, flip_time in enumerate(flip_times):
        start_dwell_idx = find_nearest_idx(plot_data_dict['t/T'], start_dwell)
        end_dwell = flip_time
        end_dwell_idx = find_nearest_idx(plot_data_dict['t/T'], end_dwell)

        if end_dwell_idx > plot_data_dict['t/T'].size - 1:
            end_dwell_idx = plot_data_dict['t/T'].size - 1

        dwell_diss_rate = np.sum(raw_diss_rate[start_dwell_idx:end_dwell_idx])/(end_dwell_idx-start_dwell_idx+1e-2)
        dwell_diss_rate_actual.append(dwell_diss_rate)

        dwell_diss_rate_lr_smooth = np.sum(smooth_diss_rate_lr[start_dwell_idx:end_dwell_idx])/(
                                        end_dwell_idx - start_dwell_idx + 1e-2)
        dwell_diss_rate_lr.append(dwell_diss_rate_lr_smooth)

        start_dwell = flip_time

    dfn = os.path.join(params_dict['PLOT_DIR'], 'dwell_diss_rate_actual.json')
    with open(dfn, 'w') as f:
        json.dump(np.array(dwell_diss_rate_actual), f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join(params_dict['PLOT_DIR'], 'dwell_diss_rate_lr_smooth.json')
    with open(dfn, 'w') as f:
        json.dump(np.array(dwell_diss_rate_lr), f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_sim_data_save_KE_mode_plots(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()

    keys_list = ['t/T', '<dissipation_rate(t)/(Barrier/T)>_T', 'dwell_times', 'flip_times', 'dwell_diss_rate_lr',
                 'eigvals_drive_start', 'forcing_overlap_drive_start', 'eigvals_drive_end',
                 'forcing_overlap_drive_end', '<U(t)/(Barrier)>_T', '<# spring transitions(t)>_T']

    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
    params_dict['PLOT_DIR'] = PLOT_DIR

    omega_drive = 2*np.pi/params_dict['T_list'][0]
    damping_rate = params_dict['DAMPING_RATE']
    plot_data_dict['KE_mode_drive_start'] = np.empty(plot_data_dict['eigvals_drive_start'].size)
    for eigval_i, eigval in enumerate(plot_data_dict['eigvals_drive_start']):
        plot_data_dict['KE_mode_drive_start'][eigval_i] = (plot_data_dict['forcing_overlap_drive_start'][eigval_i]/(
            (eigval - omega_drive*omega_drive)**2 + (damping_rate*omega_drive/params_dict['MASS'])**2) )

    omega_drive = 2 * np.pi / params_dict['T_list'][-1]
    plot_data_dict['KE_mode_drive_end'] = np.empty(plot_data_dict['eigvals_drive_end'].size)
    for eigval_i, eigval in enumerate(plot_data_dict['eigvals_drive_end']):
        plot_data_dict['KE_mode_drive_end'][eigval_i] = (plot_data_dict['forcing_overlap_drive_end'][eigval_i]/(
            (eigval - omega_drive*omega_drive)**2 + (damping_rate*omega_drive/params_dict['MASS'])**2))

    dfn = os.path.join(params_dict['PLOT_DIR'], 'KE_mode_drive_start.json')
    with open(dfn, 'w') as f:
        json.dump(plot_data_dict['KE_mode_drive_start'], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(params_dict['PLOT_DIR'], 'KE_mode_drive_end.json')
    with open(dfn, 'w') as f:
        json.dump(plot_data_dict['KE_mode_drive_end'], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))


def read_sim_data_save_lr_diss_rate(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    plot_data_dict = dict()

    keys_list = ['t/T', ]

    for k in keys_list:
        dfn = os.path.join(PLOT_DIR, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            plot_data_dict[k] = np.array(json.load(f))
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
    dfn = os.path.join(PLOT_DIR, 'sampled_philist.json')
    with open(dfn,'r') as f:
        plot_phi_tlist = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'adjacency_data.json')
    with open(dfn,'r') as f:
        adj_data = json.load(f)
    ergraph = json_graph.adjacency_graph(adj_data)

    tlength = plot_xilist_tsteps.shape[0]
    forcing_direction = np.zeros([2*params_dict['NODES'], tlength])
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I'], :] = np.cos(plot_phi_tlist)
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I'] + 1, :] = np.sin(plot_phi_tlist)

    sampled_diss_rate_lr = np.empty(tlength)

    get_lr_diss_rate(plot_xilist_tsteps, plot_yilist_tsteps, plot_tlist, forcing_direction, params_dict,
                             np.array(params_dict['AMAT']), sampled_diss_rate_lr)

    sampled_diss_rate_lr = sampled_diss_rate_lr*params_dict['power_scale_factor']

    window_size = 10
    convolve_window = np.ones((window_size,)) / (1. * window_size)
    smooth_sampled_diss_rate_lr = np.convolve(sampled_diss_rate_lr, convolve_window, mode='valid')
    smooth_plot_tlist = np.convolve(plot_tlist, convolve_window, mode='valid')

    dfn = os.path.join(PLOT_DIR, 'smooth_sampled_diss_rate_lr.json')
    with open(dfn, 'w') as f:
        json.dump(smooth_sampled_diss_rate_lr, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join(PLOT_DIR, 'smooth_sampled_tlist.json')
    with open(dfn, 'w') as f:
        json.dump(smooth_plot_tlist, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

def read_sim_data_save_mode_participation_ratios(data_dir, tol=1e-8):
    # print 'data dir is %s' %(data_dir)
    dfn = os.path.join(data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(data_dir, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        sampled_xilist = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        sampled_yilist = np.array(json.load(f))

    plot_data_dict = dict()
    drive_start_index = int(params_dict['T_RELAX']/(2*params_dict['DT']*params_dict['PLOT_DELTA_STEPS']))
    participation_ratios_start = np.empty(2*params_dict['NODES'])
    participation_ratios_end = np.empty(2*params_dict['NODES'])

    get_normal_mode_participation_ratios(sampled_xilist[drive_start_index], sampled_yilist[drive_start_index],
                                         participation_ratios_start, params_dict, np.array(params_dict['AMAT']), tol)

    get_normal_mode_participation_ratios(sampled_xilist[-1], sampled_yilist[-1],
                                         participation_ratios_end, params_dict, np.array(params_dict['AMAT']), tol)

    plot_data_dict['participation_ratios_start'] = participation_ratios_start
    plot_data_dict['participation_ratios_end'] = participation_ratios_end

    for k, v in plot_data_dict.iteritems():
        dfn = os.path.join(params_dict['PLOT_DIR'], str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'w') as f:
            json.dump(v, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))


def get_spring_params_dicts():
    spring_config_dicts = {'equal_E_right_stiffer':{'SPRING_K': 2.2, 'Lbarrier': 1.9, 'OFFSET': 0.14, 'L1':1., 'L2':3.},
                           'equal_E_left_stiffer': {'SPRING_K': 2.2, 'Lbarrier': 2.1, 'OFFSET': -0.15, 'L1':1., 'L2':3.},
                           'right_low_E_stiffer': {'SPRING_K': 2.5, 'Lbarrier': 1.9, 'OFFSET': 0.06, 'L1':1., 'L2':3.},
                           'left_low_E_stiffer': {'SPRING_K': 2.5, 'Lbarrier': 2.1, 'OFFSET': -0.08, 'L1':1., 'L2':3.},
                           'equal_E_equal_stiffness': {'SPRING_K': 2.2, 'Lbarrier': 2., 'OFFSET': 0., 'L1':1., 'L2':3.}}
    keylist = spring_config_dicts.keys()
    for k in keylist:
        spring_dict = spring_config_dicts[k]
        L1 = spring_dict['L1']
        L2 = spring_dict['L2']
        OFFSET = spring_dict['OFFSET']
        SPRING_K = spring_dict['SPRING_K']
        Lbarrier = spring_dict['Lbarrier']
        U_L1 = L1*OFFSET-SPRING_K*L1*L1*(L1*L1+6*L2*Lbarrier-2*L1*(L2+Lbarrier))/12.
        U_L2 = L2*OFFSET-SPRING_K*L2*L2*(L2*L2+6*L1*Lbarrier-2*L2*(L1+Lbarrier))/12.
        if U_L1 <= U_L2:
            spring_dict['set_U_min_zero'] = -U_L1
        else:
            spring_dict['set_U_min_zero'] = -U_L2
        spring_config_dicts[k] = spring_dict
    return spring_config_dicts


def get_spring_potential_plots(spring_config_dicts, plot_param_dict, col_dict, axarr):
    for spring_config,spring_dict in spring_config_dicts.iteritems():
        col = col_dict[spring_config]
        plot_params = plot_param_dict[spring_config]
        xlist = np.linspace(0.3,3.7,200)
        Ulist = map(lambda x: U_spring(x, spring_dict), xlist)
        axarr[col].plot(xlist, Ulist, color=plot_params['color'],
                        linestyle='None', marker=plot_params['marker'], markersize=plot_params['markersize'],
                      markeredgecolor=plot_params['color'])
        axarr[col].set(xlabel=r"spring length $l$", ylabel=r"spring potential $U(l)$")


def generate_collated_spectrum_plots_erg(data_dicts_erg, plot_keys_list, plot_strings_list,
                                    plot_labels_list, col_dict, spring_config_dicts, plot_param_dict,
                                     base_ofn):
    data_dirs = data_dicts_erg.keys()
    for i, plot_key in enumerate(plot_keys_list):
        fig = plt.figure(figsize=(20,20), dpi=100)
        gs = gridspec.GridSpec(3,len(data_dirs))
        axarr = [   [plt.subplot(gs[0,i]) for i in range(len(data_dirs))],
                    [plt.subplot(gs[1,i]) for i in range(len(data_dirs))],
                    [plt.subplot(gs[2,i]) for i in range(len(data_dirs))], ]

        for data_dir in data_dirs:
            seed2_start = data_dir.rfind('_seed2_')+7
            seed2_len = data_dir[seed2_start:].find('_')
            seed2 = data_dir[seed2_start:seed2_start+seed2_len]
            spring_config_string = data_dir[data_dir.rfind('_bistable_')+10:]

            col = col_dict[seed2]
            params_dict = data_dicts_erg[data_dir]['params_dict']
            drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
            start_index = params_dict['START_PLOT_INDEX']
            plot_data_dict = data_dicts_erg[data_dir]['plot_data_dict']

            histplot = axarr[row][col].imshow(plot_data_dict[plot_key], aspect='auto',
                                             cmap=plt.get_cmap('plasma'),
                                             extent=(plot_data_dict['t/T'][start_index], plot_data_dict['t/T'][-1],
                                                     plot_data_dict['eig_bins'][-1], plot_data_dict['eig_bins'][0]))
            divider = make_axes_locatable(axarr[row][col])
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(histplot, orientation='vertical', label=plot_labels_list[i], cax=cax)
            axarr[row][col].axhline(y=2 * np.pi / float(params_dict['FORCE_PERIOD']), c='gray', linestyle='-.',
                                    linewidth = 5.)
            axarr[row][col].axvline(x=drive_start_t, c='white', linestyle='-.', linewidth=3.)
            # axarr[0][3].margins(0.05)

            if row==4:
                axarr[row][col].set(xlabel=r"time, $t/T_{\mathrm{drive}}$")

            if col==0:
                axarr[row][col].set(ylabel=r"normal mode frequency $\omega$")
            elif col==4:
                axarr[row][col].set(ylabel=seed2)
                axarr[row][col].yaxis.set_label_coords(1.4,0.5)

        get_spring_potential_plots(spring_config_dicts, plot_param_dict, col_dict, axarr[0][:])

        gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        gs.tight_layout(fig)
        ofn = base_ofn+plot_strings_list[i]
        fig.savefig(ofn)
        plt.close()

def generate_collated_data_plots_erg(data_dicts_erg, plot_keys_list,
                               plot_labels_list, base_ofn):
    data_dirs = data_dicts_erg.keys()
    fig = plt.figure(figsize=(15, 25), dpi=100)
    gs = gridspec.GridSpec(1+len(plot_keys_list),len(data_dirs))
    axarr = [[plt.subplot(gs[i,j]) for j in range(len(data_dirs))] for i in range(1+len(plot_keys_list))]

    for i,data_dir in enumerate(data_dirs):
        # seed2_start = data_dir.rfind('_seed2_')+7
        # seed2_len = data_dir[seed2_start:].find('_')
        # seed2 = data_dir[seed2_start:seed2_start+seed2_len]

        params_dict = data_dicts_erg[data_dir]['params_dict']
        seed2 = params_dict['SEED2']
        drive_start_t = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
        start_index = params_dict['START_PLOT_INDEX']
        plot_data_dict = data_dicts_erg[data_dir]['plot_data_dict']

        xilist = data_dicts_erg[data_dir]['xilist']
        yilist = data_dicts_erg[data_dir]['yilist']
        t_end = data_dicts_erg[data_dir]['t_end']
        ergraph = data_dicts_erg[data_dir]['ergraph']
        generate_graph_plot(ergraph, axarr[0][i], params_dict, xilist, yilist, t_end)

        for j,plot_key in enumerate(plot_keys_list):
            axarr[j+1][i].plot(plot_data_dict['t/T'][start_index:], plot_data_dict[plot_key][start_index:],
                             linestyle=':', c=params_dict['COLORS'][0],
                             marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
            axarr[j+1][i].set(ylabel=plot_labels_list[j])
            axarr[j+1][i].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
            axarr[j+1][i].margins(0.05)

        axarr[-1][i].set(xlabel=r"time, $t/T_{\mathrm{drive}}$")

        axarr[0][i].set(xlabel=str(seed2))# +'_hot_ beta_'+str(params_dict['HOT_BETA'])) ############ additional labels, change as needed #############
        axarr[0][i].xaxis.set_label_coords(0.5,1.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+'_U_diss_rate.png' # 'graph_hot_cold_KE_diss_rate.png'
    fig.savefig(ofn)
    plt.close()


def generate_graph_plot(ergraph, ax, params_dict, xilist, yilist, t ):
    pos = dict()
    colors = []
    for node in range(params_dict['NODES']):
        colors.append('lightgrey')
    colors[params_dict['HOT_NODE']] = 'darkorange'
    colors[params_dict['COLD_NODE']] = 'deepskyblue'
    # colors[params_dict['MAX_DEG_VERTEX_I']] = 'darkorange'

    xmin = np.amin(xilist)
    xmax = np.amax(xilist)
    ymin = np.amin(yilist)
    ymax = np.amax(yilist)

    for node in range(params_dict['NODES']):
        pos[node] = (xilist[node], yilist[node])

    nx.draw(ergraph, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axis('on')
    ax.grid('on')
    ax.text(xmin, ymax + 0.5, r"t/$T_\mathrm{drive}$=%.3f" % (t), fontsize=12)
    ax.set(ylim=[ymin - 0.5, ymax + 1.5], xlim=[xmin - 0.5, xmax + 0.5],
           aspect=(xmax - xmin + 1) / (ymax - ymin + 1))

def get_spring_bits(data_dir, spring_bits_tsteps):
    dfn = os.path.join(data_dir, 'params_dict.json')
    with open(dfn,'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(data_dir, 'sampled_xilist.json')
    with open(dfn,'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'sampled_yilist.json')
    with open(dfn,'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    Amat = np.array(params_dict['AMAT'])
    spring_lengths_tsteps = np.empty([plot_xilist_tsteps.shape[0], params_dict['EDGES']], dtype=np.float64)
    # spring_bits_tsteps = np.empty([plot_xilist_tsteps.shape[0], params_dict['EDGES']], dtype=np.int64)
    get_spring_lengths_tsteps(plot_xilist_tsteps, plot_yilist_tsteps, Amat, spring_lengths_tsteps)
    get_spring_bits_tsteps(spring_lengths_tsteps, spring_bits_tsteps, params_dict)

    return spring_bits_tsteps

def plot_rep_distance_amp_variation(list_of_data_dirs, amplist, plot_dir, suffix_string):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    base_ofn = os.path.join(plot_dir, 'agg_rep_distance_amp_vary_plot_')

    cm = plt.cm.get_cmap('PuBuGn') #plt.cm.get_cmap('Blues')
    Amin = min(amplist)
    Amax = max(amplist)
    # contrastA = [0.45 + i*(0.35/(len(amplist)-1)) for i in range(len(amplist))]
    contrastA = [ 0.9, 0.7, 0.4 ]
    clist = [cm(Ai) for Ai in contrastA]
    # clist.reverse()

    for n, data_dirs in enumerate(list_of_data_dirs):
        distance_trajectories_i_j_list = []
        ci = clist[n]

        init_dir = data_dirs[0]

        dfn = os.path.join(init_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            init_params_dict = json.load(f)
        dfn = os.path.join(init_dir, 'sampled_tlist.json')
        with open(dfn, 'r') as f:
            init_tlist = np.array(json.load(f))

        edges = init_params_dict['EDGES']
        num_trajectory_samples = (np.array(init_params_dict['PLOT_STEPS']).size -
                                  np.int(init_params_dict['RELAX_STEPS'] / (2 * init_params_dict['PLOT_DELTA_STEPS'])))
        plot_tlist = init_tlist[-num_trajectory_samples:]

        seed1_list = [dirname[dirname.find('seed1_') + 6:dirname.find('_F_max_deg')] for dirname in data_dirs]
        unique_seeds = list(set(seed1_list))

        for k, unique_seed in enumerate(unique_seeds):
            same_seed_dirs = filter(lambda data_dir: str('seed1_' + unique_seed) in data_dir, data_dirs)

            spring_bit_trajectories = np.empty([len(same_seed_dirs), num_trajectory_samples, edges], dtype=np.int64)
            for i, same_seed_dir in enumerate(same_seed_dirs):
                spring_bit_trajectory = np.empty([init_tlist.size, edges], dtype=np.int64)
                get_spring_bits(same_seed_dir, spring_bit_trajectory)
                spring_bit_trajectories[i] = spring_bit_trajectory[-num_trajectory_samples:]

            for i in range(len(same_seed_dirs)):
                for j in range(i+1,len(same_seed_dirs)):
                    bit_trajectory_distance = np.empty(num_trajectory_samples)
                    get_distance_between_trajectories_tsteps(spring_bit_trajectories[i], spring_bit_trajectories[j],
                                                                                        bit_trajectory_distance)
                    distance_trajectories_i_j_list.append(bit_trajectory_distance)

        distance_trajectories_i_j_arr = np.array(distance_trajectories_i_j_list)

        dist_trajectories_mean = np.mean(distance_trajectories_i_j_arr, axis=0)
        dist_trajectories_std = np.std(distance_trajectories_i_j_arr, axis=0)

        rep_data_dir = '/Volumes/wd/hridesh/force_drive_sims/rep_agg_data'
        dfn = os.path.join(rep_data_dir, 'dist_mean_'+ str(n)+str('_A%.1f'%(amplist[n])).replace('.','p')+'.json')
        with open(dfn, 'w') as f:
            json.dump(dist_trajectories_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        rep_data_dir = '/Volumes/wd/hridesh/force_drive_sims/rep_agg_data'
        dfn = os.path.join(rep_data_dir, 'dist_std_'+ str(n)+str('_A%.1f'%(amplist[n])).replace('.','p')+'.json')
        with open(dfn, 'w') as f:
            json.dump(dist_trajectories_std, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

        zeros_ref = np.zeros(dist_trajectories_mean.size)
        ax.fill_between(plot_tlist,
            np.maximum(dist_trajectories_mean-dist_trajectories_std, zeros_ref), dist_trajectories_mean+dist_trajectories_std,
                                 color=ci, alpha=0.1)
        ax.plot(plot_tlist, dist_trajectories_mean,linestyle=':', c=ci,
                 marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[n]))
        ax.set(ylabel=r"$\left\langle \delta d_{ij}(t)\right\rangle$",
                        xlabel=r"$t/T_\mathrm{drive}$")

    # ax.legend(loc='best')
    ax.margins(0.05)
    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_dij_same_init_conditions.png'
    fig.savefig(ofn)
    plt.close()

def read_sim_data_plot_rep_aggregated(data_dirs, suffix_string):
    init_dir = data_dirs[0]
    base_ofn = os.path.join(os.path.split(init_dir)[0], 'agg_plot_')
    dfn = os.path.join(init_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        init_params_dict = json.load(f)
    dfn = os.path.join(init_dir, 'sampled_tlist.json')
    with open(dfn,'r') as f:
        init_tlist = np.array(json.load(f))

    seed1_list = [ dirname[dirname.find('seed1_')+6:dirname.find('_f_max_deg')] for dirname in data_dirs ]
    unique_seeds = list(set(seed1_list))

    num_always_same_bits = [] # 1 number per unique seed
    num_always_diff_bits = [] # 1 number per unique seed
    num_sometimes_diff_bits = [] # 1 number per unique seed

    degs_always_same_bits = [] # list of variable length per unique seed
    degs_always_diff_bits = [] # list of variable length per unique seed
    degs_sometimes_diff_bits = [] # list of variable length per unique seed

    dists_always_same_bits = [] # list of variable length per unique seed
    dists_always_diff_bits = [] # list of variable length per unique seed
    dists_sometimes_diff_bits = [] # list of variable length per unique seed

    num_trajectory_samples = (np.array(init_params_dict['PLOT_STEPS']).size -
                                np.int(init_params_dict['RELAX_STEPS']/(2*init_params_dict['PLOT_DELTA_STEPS'])) )
    plot_tlist = init_tlist[-num_trajectory_samples:]
    avg_bit_trajectory_distances = np.empty([len(unique_seeds), num_trajectory_samples]) # list of length ~ simulation time per unique seed

    for k, unique_seed in enumerate(unique_seeds):
        same_seed_dirs = filter(lambda data_dir: unique_seed in data_dir, data_dirs)
        seed_dir = same_seed_dirs[0]
        dfn = os.path.join(seed_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            seed_params_dict = json.load(f)
        edges = int(seed_params_dict['EDGES'])
        dfn = os.path.join(seed_dir, 'adjacency_data.json')
        with open(dfn, 'r') as f:
            adj_data = json.load(f)
        RGRAPH = json_graph.adjacency_graph(adj_data)

        end_states = np.empty([len(same_seed_dirs), edges ])
        end_state_xors = np.empty([len(same_seed_dirs)*(len(same_seed_dirs)-1)/2, edges ])
        degs_seed_graph = np.empty(edges)
        dists_seed_graph = np.empty(edges)

        get_spring_degrees(RGRAPH, seed_params_dict, degs_seed_graph)
        get_spring_distances(RGRAPH, seed_params_dict, dists_seed_graph)
        spring_bit_trajectories = np.empty([len(same_seed_dirs), num_trajectory_samples, edges], dtype=np.int64)
        bit_trajectory_distances = np.empty([len(same_seed_dirs)*(len(same_seed_dirs)-1)/2, num_trajectory_samples])

        for i, same_seed_dir in enumerate(same_seed_dirs):
            spring_bit_trajectory = get_spring_bits(same_seed_dir)
            spring_bit_trajectories[i]= spring_bit_trajectory[-num_trajectory_samples:]
            end_states[i] = spring_bit_trajectories[i][-1]

        pair_index = 0
        for i in range(len(same_seed_dirs)):
            for j in range(i+1,len(same_seed_dirs)):
                get_distance_between_trajectories_tsteps(spring_bit_trajectories[i], spring_bit_trajectories[j],
                                                            bit_trajectory_distances[pair_index])
                end_state_xors[pair_index] = np.logical_xor(end_states[i], end_states[j])
                pair_index += 1

        avg_bit_trajectory_distances[k] = np.mean(bit_trajectory_distances, axis=0)
        avg_end_state_xors = np.mean(end_state_xors, axis=0)
        seed_always_same_spring_indices = []
        seed_always_diff_spring_indices = []
        seed_sometimes_diff_spring_indices = []

        for i in range(edges):
            if avg_end_state_xors[i]>0.99:
                seed_always_diff_spring_indices.append(i)
            elif avg_end_state_xors[i]<0.01:
                seed_always_same_spring_indices.append(i)
            else:
                seed_sometimes_diff_spring_indices.append(i)

        num_always_same_bits.append(len(seed_always_same_spring_indices))
        print '%d always same bits' %(len(seed_always_same_spring_indices))
        num_always_diff_bits.append(len(seed_always_diff_spring_indices))
        print '%d always diff bits' % (len(seed_always_diff_spring_indices))
        num_sometimes_diff_bits.append(len(seed_sometimes_diff_spring_indices))
        print '%d sometimes diff bits' % (len(seed_sometimes_diff_spring_indices))

        for spring_i in seed_always_same_spring_indices:
            degs_always_same_bits.append(degs_seed_graph[spring_i])
            dists_always_same_bits.append(dists_seed_graph[spring_i])

        for spring_i in seed_always_diff_spring_indices:
            degs_always_diff_bits.append(degs_seed_graph[spring_i])
            dists_always_diff_bits.append(dists_seed_graph[spring_i])

        for spring_i in seed_sometimes_diff_spring_indices:
            degs_sometimes_diff_bits.append(degs_seed_graph[spring_i])
            dists_sometimes_diff_bits.append(dists_seed_graph[spring_i])

    print 'total always same bits: %d'%(sum(num_always_same_bits))
    num_always_same_hist = np.histogram(num_always_same_bits, bins='sqrt')
    print 'num bins for always same bits: %d' %(len(num_always_same_hist[1])-1)
    print 'total always diff bits: %d'%(sum(num_always_diff_bits))
    num_always_diff_hist = np.histogram(num_always_diff_bits, bins='sqrt')
    print 'num bins for always diff bits: %d' % (len(num_always_diff_hist[1]) - 1)
    print 'total sometimes diff bits: %d' % (sum(num_sometimes_diff_bits))
    num_sometimes_diff_hist = np.histogram(num_sometimes_diff_bits, bins='sqrt')
    print 'num bins for sometimes diff bits: %d' % (len(num_sometimes_diff_hist[1]) - 1)

    degs_always_same_hist = np.histogram(degs_always_same_bits, bins='sqrt')
    degs_always_diff_hist = np.histogram(degs_always_diff_bits, bins='sqrt')
    degs_sometimes_diff_hist = np.histogram(degs_sometimes_diff_bits, bins='sqrt')

    dists_always_same_hist = np.histogram(dists_always_same_bits, bins='sqrt')
    dists_always_diff_hist = np.histogram(dists_always_diff_bits, bins='sqrt')
    dists_sometimes_diff_hist = np.histogram(dists_sometimes_diff_bits, bins='sqrt')

    avg_bit_trajectory_distances_mean = np.mean(avg_bit_trajectory_distances, axis=0)
    avg_bit_trajectory_distances_std = np.std(avg_bit_trajectory_distances, axis=0)

    fig = plt.figure(figsize=(20, 5), dpi=100)
    gs = gridspec.GridSpec(1, 4)
    axarr = [ [ plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3]) ]]

    hist = num_sometimes_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    num_sometimes_diff_plt = axarr[0][0].bar(hist_coords, hist[0], color='olivedrab',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = num_always_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    num_always_diff_plt = axarr[0][0].bar(hist_coords, hist[0], color='indianred',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = num_always_same_hist
    hist_coords = (hist[1][:-1]+hist[1][1:])/2.
    hist_width = (hist[1][1]-hist[1][0]) if len(hist[1])>2 else 0.5
    num_always_same_plt = axarr[0][0].bar(hist_coords, hist[0], color='cadetblue',
                width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)

    axarr[0][0].legend((num_always_same_plt[0], num_sometimes_diff_plt[0], num_always_diff_plt[0]),
                       ('unchanging bits', 'sometimes changing bits', 'always changing bits'))
    axarr[0][0].set(ylabel=r"frequency of occurence", xlabel=r"# of bits")

    hist = degs_sometimes_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    degs_sometimes_diff_plt = axarr[0][1].bar(hist_coords, hist[0], color='olivedrab',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = degs_always_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    degs_always_diff_plt = axarr[0][1].bar(hist_coords, hist[0], color='indianred',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = degs_always_same_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    degs_always_same_plt = axarr[0][1].bar(hist_coords, hist[0], color='cadetblue',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    axarr[0][1].legend((degs_always_same_plt[0], degs_sometimes_diff_plt[0], degs_always_diff_plt[0]),
                       ('unchanging bits', 'sometimes changing bits', 'always changing bits'))
    axarr[0][1].set(ylabel=r"frequency of occurence", xlabel=r"degree of bits")

    hist = dists_sometimes_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    dists_sometimes_diff_plt = axarr[0][2].bar(hist_coords, hist[0], color='olivedrab',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = dists_always_diff_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    dists_always_diff_plt = axarr[0][2].bar(hist_coords, hist[0], color='indianred',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    hist = dists_always_same_hist
    hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
    hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.5
    dists_always_same_plt = axarr[0][2].bar(hist_coords, hist[0], color='cadetblue',
                    width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)
    axarr[0][2].legend((dists_always_same_plt[0], dists_sometimes_diff_plt[0], dists_always_diff_plt[0]),
                       ('unchanging bits', 'sometimes changing bits', 'always changing bits'))
    axarr[0][2].set(ylabel=r"frequency of occurence", xlabel=r"spring distance from driven node")

    axarr[0][3].fill_between(plot_tlist,
        avg_bit_trajectory_distances_mean-avg_bit_trajectory_distances_std, avg_bit_trajectory_distances_mean+avg_bit_trajectory_distances_std,
                             color='cadetblue', alpha = 0.4)
    axarr[0][3].plot(plot_tlist, avg_bit_trajectory_distances_mean,linestyle=':', c='cadetblue',
                     marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5)
    axarr[0][3].set(ylabel=r"$\left\langle \delta d_{ij}(t)\right\rangle$",
                    xlabel=r"$t/T_\mathrm{drive}$")
    axarr[0][3].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_repetition_analyse.png'
    fig.savefig(ofn)
    plt.close()

def read_sim_data_plot_L_KE_com_fraction_aggregated(data_dirs, suffix_string):
    num_dirs = len(data_dirs)
    print 'num dirs is %d, suffix_string is %s'%(num_dirs, suffix_string)
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    keys_list = ['L_anchor_sampled', 'L_anchor_avg', '<KE(t)_{c.o.m.}_by_KE>_T']
    init_data_dir = data_dirs[0]

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)

    for k in keys_list:
        dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
            agg_plot_data_dict[k][0] = k_val

    dfn = os.path.join(init_data_dir, 't_by_T'+'.json')
    with open(dfn,'r') as f:
        agg_plot_data_dict['t/T'] = np.array(json.load(f))

    dfn = os.path.join(init_data_dir, 'sampled_tlist'+'.json')
    with open(dfn,'r') as f:
        agg_plot_data_dict['sampled_tlist'] = np.array(json.load(f))

    dfn = os.path.join(init_data_dir, 'avg_sampled_tlist'+'.json')
    with open(dfn,'r') as f:
        agg_plot_data_dict['avg_sampled_tlist'] = np.array(json.load(f))

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_list:
            dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k][i+1] = np.array(json.load(f))

    sampled_L_mean = np.mean(agg_plot_data_dict['L_anchor_sampled'], axis=0)
    sampled_L_std = np.std(agg_plot_data_dict['L_anchor_sampled'], axis=0)
    T_avg_L_mean = np.mean(agg_plot_data_dict['L_anchor_avg'], axis=0)
    T_avg_L_std = np.std(agg_plot_data_dict['L_anchor_avg'], axis=0)
    KE_com_frac_mean = np.mean(agg_plot_data_dict['<KE(t)_{c.o.m.}_by_KE>_T'], axis=0)
    KE_com_frac_std = np.std(agg_plot_data_dict['<KE(t)_{c.o.m.}_by_KE>_T'], axis=0)

    fig = plt.figure(figsize=(30,10), dpi=100)
    # fig = plt.figure(figsize=(30,15), dpi=100)
    # gs = gridspec.GridSpec(2,3)
    gs = gridspec.GridSpec(1, 3)
    axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])]]
    start_index = params_dict['START_PLOT_INDEX']
    drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']

    axarr[0][0].fill_between(agg_plot_data_dict['sampled_tlist'],
                            sampled_L_mean - sampled_L_std, sampled_L_mean + sampled_L_std,
                             color=params_dict['COLORS'][0], alpha = 0.5)
    axarr[0][0].plot(agg_plot_data_dict['sampled_tlist'],
                            sampled_L_mean,linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][0].set(ylabel=r"$L(t)$",
                    xlabel=r"$t/T_\mathrm{drive}$")
    axarr[0][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][0].margins(0.05)

    axarr[0][1].fill_between(agg_plot_data_dict['avg_sampled_tlist'],
                             T_avg_L_mean - T_avg_L_std, T_avg_L_mean + T_avg_L_std,
                             color=params_dict['COLORS'][0], alpha=0.5)
    axarr[0][1].plot(agg_plot_data_dict['avg_sampled_tlist'],
                     T_avg_L_mean, linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][1].set(ylabel=r"$\langle L(t)\rangle_T$",
                    xlabel=r"$t/T_\mathrm{drive}$")
    axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][1].axvline(x=(freq_switch_t + params_dict['T_RELAX']) / params_dict['FORCE_PERIOD'],
                            c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][1].margins(0.05)

    axarr[0][2].fill_between(agg_plot_data_dict['t/T'][start_index:],
        KE_com_frac_mean[start_index:]-KE_com_frac_std[start_index:], KE_com_frac_mean[start_index:]+KE_com_frac_std[start_index:],
                             color=params_dict['COLORS'][0], alpha=0.5)
    axarr[0][2].plot(agg_plot_data_dict['t/T'][start_index:], KE_com_frac_mean[start_index:],
                     linestyle=':',c=params_dict['COLORS'][0],
                     marker='o',markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None',markersize=5)
    axarr[0][2].set(ylabel=r"$\langle \mathrm{KE}_{c.o.m.}\,\mathrm{fraction}\rangle_T$")
    axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][2].axvline(x=(freq_switch_t + params_dict['T_RELAX']) / params_dict['FORCE_PERIOD'],
                            c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][2].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_L_KE_com_frac_avg.png'
    # ofn = base_ofn+suffix_string+'_hot_cold_diss_U_KE_d_relaxed_avg.png'
    fig.savefig(ofn)
    plt.close()

def read_tstop(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    return params_dict['t_stop']

def read_total_hops(PLOT_DIR):
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(PLOT_DIR, '<# spring transitions(t)>_T.json')
    with open(dfn, 'r') as f:
        spring_hops = np.array(json.load(f))
    start_index = params_dict['START_PLOT_INDEX']

    return np.sum(spring_hops[start_index:])


def read_sim_data_save_tstop_unstable_KE_phase_flow(PLOT_DIR):
    print 'loading data_dir %s'%(PLOT_DIR)
    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    dfn = os.path.join(PLOT_DIR, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vxilist.json')
    with open(dfn, 'r') as f:
        plot_vxilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_vyilist.json')
    with open(dfn, 'r') as f:
        plot_vyilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_tlist.json')
    with open(dfn, 'r') as f:
        plot_tlist = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 'sampled_philist.json')
    with open(dfn, 'r') as f:
        sampled_philist = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, '<# spring transitions(t)>_T.json')
    with open(dfn, 'r') as f:
        num_spring_transitions_tsteps = np.array(json.load(f))
    dfn = os.path.join(PLOT_DIR, 't_by_T.json')
    with open(dfn, 'r') as f:
        t_by_T = np.array(json.load(f))

    print 'done loading data_dir %s' % (PLOT_DIR)
    get_tstop(num_spring_transitions_tsteps, t_by_T, params_dict)

    forcing_direction = np.zeros([2 * params_dict['NODES'], plot_xilist_tsteps.shape[0]])
    forcing_direction[2 * params_dict['MAX_DEG_VERTEX_I'], :] = np.cos(sampled_philist)
    forcing_direction[2 * params_dict['MAX_DEG_VERTEX_I'] + 1, :] = np.sin(sampled_philist)
    Amat = np.array(params_dict['AMAT'])
    diss_rate_lr = np.empty(plot_tlist.size)
    fraction_div_phase_space_flow_per_dirn = np.empty(plot_tlist.size)
    fraction_unstable_KE_per_dirn = np.empty(plot_tlist.size)
    max_unstable_KE = np.empty(plot_tlist.size)
    max_KE_all_dirns = np.empty(plot_tlist.size)

    get_sampled_lr_diss_rate_phase_flow_div_dirns_KE_unstable_dirns(plot_xilist_tsteps,
        plot_yilist_tsteps, plot_vxilist_tsteps, plot_vyilist_tsteps, plot_tlist, forcing_direction,
        params_dict, Amat, diss_rate_lr, fraction_div_phase_space_flow_per_dirn, fraction_unstable_KE_per_dirn,
                                            max_unstable_KE, max_KE_all_dirns, 1e-8)

    dfn = os.path.join(PLOT_DIR, 'params_dict.json')
    with open(dfn, 'w') as f:
        json.dump(params_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(PLOT_DIR, 'sampled_diss_rate_lr.json')
    with open(dfn, 'w') as f:
        json.dump(diss_rate_lr*params_dict['power_scale_factor'], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(PLOT_DIR, 'fraction_phase_space_flow_per_div_dirn.json')
    with open(dfn, 'w') as f:
        json.dump(fraction_div_phase_space_flow_per_dirn, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(PLOT_DIR, 'fraction_KE_per_unstable_dirn.json')
    with open(dfn, 'w') as f:
        json.dump(fraction_unstable_KE_per_dirn, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(PLOT_DIR, 'max_unstable_directions_KE.json')
    with open(dfn, 'w') as f:
        json.dump(max_unstable_KE*params_dict['energy_scale_factor'], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    dfn = os.path.join(PLOT_DIR, 'max_all_directions_KE.json')
    with open(dfn, 'w') as f:
        json.dump(max_KE_all_dirns*params_dict['energy_scale_factor'], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    return PLOT_DIR

def read_sim_data_plot_agg_phase_flow_KE_tstop(all_data_dirs, suffix_string):
    t_stop_list = [read_tstop(data_dir) for data_dir in all_data_dirs]
    early_t_stop = np.percentile(t_stop_list, 25)
    median_t_stop = np.percentile(t_stop_list, 50)
    late_t_stop = np.percentile(t_stop_list, 75)
    early_stop_dirs = [all_data_dirs[i] for i in range(len(t_stop_list)) if t_stop_list[i] <= early_t_stop]
    median_stop_dirs = [all_data_dirs[i] for i in range(len(t_stop_list)) if early_t_stop < t_stop_list[i] <= median_t_stop ]
    late_stop_dirs =  [all_data_dirs[i] for i in range(len(t_stop_list)) if median_t_stop < t_stop_list[i] < late_t_stop ]
    latest_stop_dirs = [all_data_dirs[i] for i in range(len(t_stop_list)) if t_stop_list[i] >= late_t_stop]
    tstop_text_list = [ 't_stop <= %.1f'%(early_t_stop), '%.1f < t_stop <= %.1f'%(early_t_stop, median_t_stop),
                        '%.1f < t_stop < %.1f'%(median_t_stop, late_t_stop), 't_stop >= %.1f' %(late_t_stop)]
    suffix_tstop_list = ['_stops_early', '_stops_median', '_stops_late', '_stops_last']
    t_stop_thresholds = [ early_t_stop, median_t_stop, late_t_stop]
    ######  get all other data, plot histogram and 5 other plots, call the
    for j, data_dirs in enumerate([early_stop_dirs, median_stop_dirs, late_stop_dirs, latest_stop_dirs]):
        agg_plot_data_dict = dict()
        base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
        keys_list = ['sampled_diss_rate_lr', 'fraction_phase_space_flow_per_div_dirn', 'fraction_KE_per_unstable_dirn',
                     'max_unstable_directions_KE', 'max_all_directions_KE' ]
        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)

        for k in keys_list:
            dfn = os.path.join(init_data_dir, str(k)+'.json')
            with open(dfn,'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
                agg_plot_data_dict[k][0] = k_val

        dfn = os.path.join(init_data_dir, 'sampled_tlist'+'.json')
        with open(dfn,'r') as f:
            agg_plot_data_dict['plot_tlist'] = np.array(json.load(f))

        for i, data_dir in enumerate(data_dirs[1:]):
            for k in keys_list:
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    agg_plot_data_dict[k][i+1] = np.array(json.load(f))

        tstop_hist = np.histogram(t_stop_list, bins='sqrt')
        sampled_qdot_lr_mean = np.mean(agg_plot_data_dict['sampled_diss_rate_lr'], axis=0)
        sampled_qdot_lr_std = np.std(agg_plot_data_dict['sampled_diss_rate_lr'], axis=0)
        frac_div_phase_flow_mean = np.mean(agg_plot_data_dict['fraction_phase_space_flow_per_div_dirn'], axis=0)
        frac_div_phase_flow_std = np.std(agg_plot_data_dict['fraction_phase_space_flow_per_div_dirn'], axis=0)
        frac_unstable_KE_mean = np.mean(agg_plot_data_dict['fraction_KE_per_unstable_dirn'], axis=0)
        frac_unstable_KE_std = np.std(agg_plot_data_dict['fraction_KE_per_unstable_dirn'], axis=0)
        max_unstable_KE_mean = np.mean(agg_plot_data_dict['max_unstable_directions_KE'], axis=0)
        max_unstable_KE_std = np.std(agg_plot_data_dict['max_unstable_directions_KE'], axis=0)
        max_dirn_KE_mean = np.mean(agg_plot_data_dict['max_all_directions_KE'], axis=0)
        max_dirn_KE_std = np.std(agg_plot_data_dict['max_all_directions_KE'], axis=0)

        fig = plt.figure(figsize=(30,20), dpi=100)
        gs = gridspec.GridSpec(2, 3)
        axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])],
                  [plt.subplot(gs[1,0]), plt.subplot(gs[1,1]), plt.subplot(gs[1,2])] ]
        start_index = params_dict['START_PLOT_INDEX']
        drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']

        hist = tstop_hist
        hist_coords = (hist[1][:-1]+hist[1][1:])/2.
        hist_width = (hist[1][1]-hist[1][0]) if len(hist[1])>2 else 0.5
        axarr[0][0].bar(hist_coords, hist[0], color='cadetblue',
                    width=hist_width, edgecolor='k', linewidth=1)
        axarr[0][0].set(ylabel=r"frequency of occurence", xlabel=r"stopping time")
        for t_stop_thresh in t_stop_thresholds:
            axarr[0][0].axvline(x=t_stop_thresh, c='indianred', linestyle='-.')
        axarr[0][0].text(0.35, 0.9, tstop_text_list[j], transform=axarr[0][0].transAxes)

        axarr[0][1].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                sampled_qdot_lr_mean[start_index:] - sampled_qdot_lr_std[start_index:],
                sampled_qdot_lr_mean[start_index:] + sampled_qdot_lr_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[0][1].plot(agg_plot_data_dict['plot_tlist'][start_index:], sampled_qdot_lr_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[0][1].set(ylabel=r"$\dot{Q}_\mathrm{lr}/(E_b/\tau)$", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[0][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][1].margins(0.05)

        axarr[1][0].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                frac_div_phase_flow_mean[start_index:] - frac_div_phase_flow_std[start_index:],
                frac_div_phase_flow_mean[start_index:] + frac_div_phase_flow_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][0].plot(agg_plot_data_dict['plot_tlist'][start_index:], frac_div_phase_flow_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][0].set(ylabel=r"fraction diverging phase flow", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][0].margins(0.05)

        axarr[1][1].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                frac_unstable_KE_mean[start_index:] - frac_unstable_KE_std[start_index:],
                frac_unstable_KE_mean[start_index:] + frac_unstable_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][1].plot(agg_plot_data_dict['plot_tlist'][start_index:], frac_unstable_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][1].set(ylabel=r"fraction unstable KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][1].margins(0.05)

        axarr[0][2].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                max_dirn_KE_mean[start_index:] - max_dirn_KE_std[start_index:],
                max_dirn_KE_mean[start_index:] + max_dirn_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[0][2].plot(agg_plot_data_dict['plot_tlist'][start_index:], max_dirn_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[0][2].set(ylabel=r"max mode KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[0][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][2].margins(0.05)

        axarr[1][2].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                max_unstable_KE_mean[start_index:] - max_unstable_KE_std[start_index:],
                max_unstable_KE_mean[start_index:] + max_unstable_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][2].plot(agg_plot_data_dict['plot_tlist'][start_index:], max_unstable_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][2].set(ylabel=r"max unstable KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][2].margins(0.05)

        gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        gs.tight_layout(fig)

        ofn = base_ofn+suffix_string+suffix_tstop_list[j]+'_KE_phase_flow_updated.png'
        fig.savefig(ofn)
        plt.close()
        read_sim_data_plot_aggregated(data_dirs, suffix_string+suffix_tstop_list[j])

def read_sim_data_plot_agg_phase_flow_KE_total_hops(all_data_dirs, suffix_string):
    total_hops_list = [read_total_hops(data_dir) for data_dir in all_data_dirs]
    low_hops = np.percentile(total_hops_list, 25)
    median_hops = np.percentile(total_hops_list, 50)
    more_hops = np.percentile(total_hops_list, 75)
    low_hop_dirs = [all_data_dirs[i] for i in range(len(total_hops_list)) if total_hops_list[i] <= low_hops]
    median_hop_dirs = [all_data_dirs[i] for i in range(len(total_hops_list)) if low_hops < total_hops_list[i] <= median_hops ]
    more_hop_dirs =  [all_data_dirs[i] for i in range(len(total_hops_list)) if median_hops < total_hops_list[i] < more_hops ]
    most_hop_dirs = [all_data_dirs[i] for i in range(len(total_hops_list)) if total_hops_list[i] >= more_hops]
    hops_text_list = [ 'total hops <= %.1f'%(low_hops), '%.1f < total hops <= %.1f'%(low_hops, median_hops),
                        '%.1f < total hops < %.1f'%(median_hops, more_hops), 'total hops >= %.1f' %(more_hops)]
    suffix_hops_list = ['_low_hops', '_median_hops', '_more_hops', '_most_hops']
    hop_thresholds = [ low_hops, median_hops, more_hops]
    ######  get all other data, plot histogram and 5 other plots, call the
    for j, data_dirs in enumerate([low_hop_dirs, median_hop_dirs, more_hop_dirs, most_hop_dirs]):
        agg_plot_data_dict = dict()
        base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
        keys_list = ['sampled_diss_rate_lr', 'fraction_phase_space_flow_per_div_dirn', 'fraction_KE_per_unstable_dirn',
                     'max_unstable_directions_KE', 'max_all_directions_KE' ]
        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)

        for k in keys_list:
            dfn = os.path.join(init_data_dir, str(k)+'.json')
            with open(dfn,'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
                agg_plot_data_dict[k][0] = k_val

        dfn = os.path.join(init_data_dir, 'sampled_tlist'+'.json')
        with open(dfn,'r') as f:
            agg_plot_data_dict['plot_tlist'] = np.array(json.load(f))

        for i, data_dir in enumerate(data_dirs[1:]):
            for k in keys_list:
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    agg_plot_data_dict[k][i+1] = np.array(json.load(f))

        hops_hist = np.histogram(total_hops_list, bins='sqrt')
        sampled_qdot_lr_mean = np.mean(agg_plot_data_dict['sampled_diss_rate_lr'], axis=0)
        sampled_qdot_lr_std = np.std(agg_plot_data_dict['sampled_diss_rate_lr'], axis=0)
        frac_div_phase_flow_mean = np.mean(agg_plot_data_dict['fraction_phase_space_flow_per_div_dirn'], axis=0)
        frac_div_phase_flow_std = np.std(agg_plot_data_dict['fraction_phase_space_flow_per_div_dirn'], axis=0)
        frac_unstable_KE_mean = np.mean(agg_plot_data_dict['fraction_KE_per_unstable_dirn'], axis=0)
        frac_unstable_KE_std = np.std(agg_plot_data_dict['fraction_KE_per_unstable_dirn'], axis=0)
        max_unstable_KE_mean = np.mean(agg_plot_data_dict['max_unstable_directions_KE'], axis=0)
        max_unstable_KE_std = np.std(agg_plot_data_dict['max_unstable_directions_KE'], axis=0)
        max_dirn_KE_mean = np.mean(agg_plot_data_dict['max_all_directions_KE'], axis=0)
        max_dirn_KE_std = np.std(agg_plot_data_dict['max_all_directions_KE'], axis=0)

        fig = plt.figure(figsize=(30,20), dpi=100)
        gs = gridspec.GridSpec(2, 3)
        axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])],
                  [plt.subplot(gs[1,0]), plt.subplot(gs[1,1]), plt.subplot(gs[1,2])] ]
        start_index = params_dict['START_PLOT_INDEX']
        drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']

        hist = hops_hist
        hist_coords = (hist[1][:-1]+hist[1][1:])/2.
        hist_width = (hist[1][1]-hist[1][0]) if len(hist[1])>2 else 0.5
        axarr[0][0].bar(hist_coords, hist[0], color='cadetblue',
                    width=hist_width, edgecolor='k', linewidth=1)
        axarr[0][0].set(ylabel=r"frequency of occurence", xlabel=r"stopping time")
        for t_stop_thresh in hop_thresholds:
            axarr[0][0].axvline(x=t_stop_thresh, c='indianred', linestyle='-.')
        axarr[0][0].text(0.35, 0.9, hops_text_list[j], transform=axarr[0][0].transAxes)

        axarr[0][1].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                sampled_qdot_lr_mean[start_index:] - sampled_qdot_lr_std[start_index:],
                sampled_qdot_lr_mean[start_index:] + sampled_qdot_lr_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[0][1].plot(agg_plot_data_dict['plot_tlist'][start_index:], sampled_qdot_lr_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[0][1].set(ylabel=r"$\dot{Q}_\mathrm{lr}/(E_b/\tau)$", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[0][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][1].margins(0.05)

        axarr[1][0].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                frac_div_phase_flow_mean[start_index:] - frac_div_phase_flow_std[start_index:],
                frac_div_phase_flow_mean[start_index:] + frac_div_phase_flow_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][0].plot(agg_plot_data_dict['plot_tlist'][start_index:], frac_div_phase_flow_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][0].set(ylabel=r"fraction diverging phase flow", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][0].margins(0.05)

        axarr[1][1].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                frac_unstable_KE_mean[start_index:] - frac_unstable_KE_std[start_index:],
                frac_unstable_KE_mean[start_index:] + frac_unstable_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][1].plot(agg_plot_data_dict['plot_tlist'][start_index:], frac_unstable_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][1].set(ylabel=r"fraction unstable KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][1].margins(0.05)

        axarr[0][2].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                max_dirn_KE_mean[start_index:] - max_dirn_KE_std[start_index:],
                max_dirn_KE_mean[start_index:] + max_dirn_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[0][2].plot(agg_plot_data_dict['plot_tlist'][start_index:], max_dirn_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[0][2].set(ylabel=r"max mode KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[0][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][2].margins(0.05)

        axarr[1][2].fill_between(agg_plot_data_dict['plot_tlist'][start_index:],
                max_unstable_KE_mean[start_index:] - max_unstable_KE_std[start_index:],
                max_unstable_KE_mean[start_index:] + max_unstable_KE_std[start_index:],
                                 color=params_dict['COLORS'][0], alpha = 0.5)
        axarr[1][2].plot(agg_plot_data_dict['plot_tlist'][start_index:], max_unstable_KE_mean[start_index:],linestyle=':',
                         c=params_dict['COLORS'][0],
                         marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
        axarr[1][2].set(ylabel=r"max unstable KE", xlabel=r"$t/T_\mathrm{drive}$")
        axarr[1][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        for freq_switch_t in params_dict['freq_switch_times']:
            axarr[1][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][2].margins(0.05)

        gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        gs.tight_layout(fig)

        ofn = base_ofn+suffix_string+suffix_hops_list[j]+'_KE_phase_flow_updated.png'
        fig.savefig(ofn)
        plt.close()
        read_sim_data_plot_aggregated(data_dirs, suffix_string+suffix_hops_list[j])

def get_rot_angle_data_dir(data_dir):
    rot_str = str(data_dir[data_dir.find('_cycles__')+9:data_dir.find('_cycles__')+13]).replace('p','.')
    rot_angle = np.float64(rot_str)
    return rot_angle

def read_sim_data_agg_plot_strongly_perturbed_data(data_dirs, suffix_string, freq_change=0.5, delta_phi=np.pi/2.,
                                          qdot_list = 0.05*np.array(range(1,51)), wdot_list = 0.1*np.array(range(1,61))):
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_perturbed_')
    seed2_list = list(set([ data_dir[data_dir.find('_seed2_')+7:data_dir.find('_soft_stiff')] for data_dir in data_dirs]))
    delta_t_hop_temp_node_unperturb = []
    delta_t_hop_temp_network_unperturb = []
    delta_t_hop_perturb_phi_unperturb = []
    delta_t_hop_perturb_freq_unperturb = []

    delta_Qdot_hop_temp_node_unperturb = []
    delta_Qdot_hop_temp_network_unperturb = []
    delta_Qdot_hop_perturb_phi_unperturb = []
    delta_Qdot_hop_perturb_freq_unperturb = []

    delta_Wdot_hop_temp_node_unperturb = []
    delta_Wdot_hop_temp_network_unperturb = []
    delta_Wdot_hop_perturb_phi_unperturb = []
    delta_Wdot_hop_perturb_freq_unperturb = []

    hop_or_not_unperturb_qdot_lists = []
    hop_or_not_temp_node_qdot_lists = []
    hop_or_not_temp_network_qdot_lists = []
    hop_or_not_perturb_phi_qdot_lists = []
    hop_or_not_perturb_freq_qdot_lists = []

    hop_or_not_unperturb_wdot_lists = []
    hop_or_not_temp_node_wdot_lists = []
    hop_or_not_temp_network_wdot_lists = []
    hop_or_not_perturb_phi_wdot_lists = []
    hop_or_not_perturb_freq_wdot_lists = []

    orig_wdot_end_list = []
    orig_qdot_end_list = []
    for seed2 in seed2_list:
        same_seed_dirs = filter(lambda data_dir: seed2 in data_dir, data_dirs)
        [orig_wdot_end, orig_qdot_end] = read_perturb_sim_data_return_qdot_wdot_orig(same_seed_dirs[0])
        orig_wdot_end_list.append(orig_wdot_end)
        orig_qdot_end_list.append(orig_qdot_end)
        temp_node_dirs = filter(lambda data_dir: '_1e4_perturb_strong_temp_node' in data_dir, same_seed_dirs)
        print '# temp node dirs is %d' %(len(temp_node_dirs))
        temp_network_dirs = filter(lambda data_dir: '_1e4_perturb_strong_temp_network' in data_dir, same_seed_dirs)
        print '# temp network dirs is %d' %(len(temp_network_dirs))
        perturb_phi_dirs = filter(lambda data_dir: '_1e4_perturb_strong_delta_phi_w_noise' in data_dir, same_seed_dirs)
        print '# perturb phi dirs is %d' %(len(perturb_phi_dirs))
        perturb_freq_dirs = filter(lambda data_dir: '_1e4_perturb_strong_force_period_w_noise' in data_dir, same_seed_dirs)
        print '# perturb freq dirs is %d' %(len(perturb_freq_dirs))
        unperturbed_dirs = filter(lambda data_dir: '_1e4_unperturbed_2000_cycles_w_noise' in data_dir, same_seed_dirs)
        print '# unperturb dirs is %d' %(len(unperturbed_dirs))
        for i, unperturbed_dir in enumerate(unperturbed_dirs):
            dfn = os.path.join(unperturbed_dir, 'first_hop_dict.json')
            with open(dfn, 'r') as f:
                unperturbed_hop_dict = json.load(f)
            unperturbed_data_dict = dict()
            keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                         '<F(t).v(t)_by_(Barrier_by_T)>_T']
            for k in keys_list:
                dfn = os.path.join(unperturbed_dir, str(k).replace('/', '_by_') + '.json')
                with open(dfn, 'r') as f:
                    unperturbed_data_dict[k] = np.array(json.load(f))
            if unperturbed_hop_dict['first_hop_time']>1999.:
                print '%s doesn\'t hop' %(unperturbed_dir)
                # continue

            unperturbed_qdot_indices = [ getLeftIndex(unperturbed_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                      qdot_list[i]) for i in range(qdot_list.size)]
            hop_or_not_unperturbed_qdot_list = [ np.int(np.sum(unperturbed_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                        >0.1) for i in unperturbed_qdot_indices]
            hop_or_not_unperturb_qdot_lists.append(hop_or_not_unperturbed_qdot_list)

            unperturbed_wdot_indices = [ getLeftIndex(unperturbed_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                      wdot_list[i]) for i in range(wdot_list.size)]
            hop_or_not_unperturbed_wdot_list = [ np.int(np.sum(unperturbed_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                        >0.1) for i in unperturbed_wdot_indices]
            hop_or_not_unperturb_wdot_lists.append(hop_or_not_unperturbed_wdot_list)

            for j, perturb_dir in enumerate(temp_node_dirs):
                dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                with open(dfn, 'r') as f:
                    perturb_hop_dict = json.load(f)
                perturb_data_dict = dict()
                keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                             '<F(t).v(t)_by_(Barrier_by_T)>_T']
                for k in keys_list:
                    dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                    with open(dfn, 'r') as f:
                        perturb_data_dict[k] = np.array(json.load(f))

                if perturb_hop_dict['first_hop_time'] > 1999.:
                    print '%s doesn\'t hop' % (perturb_dir)
                    # continue

                delta_t_hop_temp_node_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                delta_Qdot_hop_temp_node_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                delta_Wdot_hop_temp_node_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                         qdot_list[i]) for i in range(qdot_list.size)]
                hop_or_not_temp_node_qdot_list = [
                    np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                           > 0.1) for i in perturb_qdot_indices]
                hop_or_not_temp_node_qdot_lists.append(hop_or_not_temp_node_qdot_list)

                perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                          wdot_list[i]) for i in range(wdot_list.size)]
                hop_or_not_temp_node_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in perturb_wdot_indices ]
                hop_or_not_temp_node_wdot_lists.append(hop_or_not_temp_node_wdot_list)

            for j, perturb_dir in enumerate(temp_network_dirs):
                dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                with open(dfn, 'r') as f:
                    perturb_hop_dict = json.load(f)
                perturb_data_dict = dict()
                keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                             '<F(t).v(t)_by_(Barrier_by_T)>_T']
                for k in keys_list:
                    dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                    with open(dfn, 'r') as f:
                        perturb_data_dict[k] = np.array(json.load(f))

                if perturb_hop_dict['first_hop_time'] > 1999.:
                    print '%s doesn\'t hop' % (perturb_dir)
                    # continue

                delta_t_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                delta_Qdot_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                delta_Wdot_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                         qdot_list[i]) for i in range(qdot_list.size)]
                hop_or_not_temp_network_qdot_list = [
                    np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                           > 0.1) for i in perturb_qdot_indices]
                hop_or_not_temp_network_qdot_lists.append(hop_or_not_temp_network_qdot_list)

                perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                          wdot_list[i]) for i in range(wdot_list.size)]
                hop_or_not_temp_network_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in perturb_wdot_indices ]
                hop_or_not_temp_network_wdot_lists.append(hop_or_not_temp_network_wdot_list)

            for j, perturb_dir in enumerate(perturb_phi_dirs):
                dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                with open(dfn, 'r') as f:
                    perturb_hop_dict = json.load(f)
                perturb_data_dict = dict()
                keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                             '<F(t).v(t)_by_(Barrier_by_T)>_T']
                for k in keys_list:
                    dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                    with open(dfn, 'r') as f:
                        perturb_data_dict[k] = np.array(json.load(f))

                if perturb_hop_dict['first_hop_time'] > 1999.:
                    print '%s doesn\'t hop' % (perturb_dir)
                    # continue

                delta_t_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                delta_Qdot_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                delta_Wdot_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                         qdot_list[i]) for i in range(qdot_list.size)]
                hop_or_not_perturb_phi_qdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                           > 0.1) for i in perturb_qdot_indices]
                hop_or_not_perturb_phi_qdot_lists.append(hop_or_not_perturb_phi_qdot_list)

                perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                          wdot_list[i]) for i in range(wdot_list.size)]
                hop_or_not_perturb_phi_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in perturb_wdot_indices ]
                hop_or_not_perturb_phi_wdot_lists.append(hop_or_not_perturb_phi_wdot_list)

            for j, perturb_dir in enumerate(perturb_freq_dirs):
                # perturb_period = np.float64(str(perturb_dir[perturb_dir.find('_cycles_T_')+10:perturb_dir.find('_cycles_T_')+14]).replace('p','.'))
                dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                with open(dfn, 'r') as f:
                    perturb_hop_dict = json.load(f)
                perturb_data_dict = dict()
                keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T', '<F(t).v(t)_by_(Barrier_by_T)>_T']
                for k in keys_list:
                    dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                    with open(dfn, 'r') as f:
                        perturb_data_dict[k] = np.array(json.load(f))

                if perturb_hop_dict['first_hop_time'] > 1999.:
                    print '%s doesn\'t hop' % (perturb_dir)
                    # continue
                delta_t_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                delta_Qdot_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                delta_Wdot_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                         qdot_list[i]) for i in range(qdot_list.size)]
                hop_or_not_perturb_freq_qdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                           > 0.1) for i in perturb_qdot_indices]
                hop_or_not_perturb_freq_qdot_lists.append(hop_or_not_perturb_freq_qdot_list)

                perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                          wdot_list[i]) for i in range(wdot_list.size)]
                hop_or_not_perturb_freq_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in perturb_wdot_indices ]
                hop_or_not_perturb_freq_wdot_lists.append(hop_or_not_perturb_freq_wdot_list)

    orig_wdot_end_list = np.array(orig_wdot_end_list)
    ref_y = np.linspace(0.,1.,11)
    ref_ones = np.ones(ref_y.size)
    orig_wdot_mean = np.mean(orig_wdot_end_list)
    orig_wdot_std = np.std(orig_wdot_end_list)
    orig_qdot_end_list = np.array(orig_qdot_end_list)
    orig_qdot_mean = np.mean(orig_qdot_end_list)
    orig_qdot_std = np.std(orig_qdot_end_list)

    fig = plt.figure(figsize=(50, 40), dpi=100)
    gs = gridspec.GridSpec(4, 5)
    axarr = [[ plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3]), plt.subplot(gs[0,4])],
              [ plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4])],
              [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4])],
              [plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2]), plt.subplot(gs[3, 3]), plt.subplot(gs[3, 4])]]
    data_lists = [[delta_t_hop_temp_node_unperturb, delta_Qdot_hop_temp_node_unperturb, delta_Wdot_hop_temp_node_unperturb],
                  [delta_t_hop_temp_network_unperturb, delta_Qdot_hop_temp_network_unperturb, delta_Wdot_hop_temp_network_unperturb],
                  [delta_t_hop_perturb_phi_unperturb, delta_Qdot_hop_perturb_phi_unperturb, delta_Wdot_hop_perturb_phi_unperturb],
                  [delta_t_hop_perturb_freq_unperturb, delta_Qdot_hop_perturb_freq_unperturb, delta_Wdot_hop_perturb_freq_unperturb]]
    xlabels = [ [r"$\Delta t_\mathrm{hop}$ (T node $-$unperturb)", r"$\Delta\dot{Q}_\mathrm{hop}$ (T node $-$unperturb)",
                 r"$\Delta\dot{W}_\mathrm{hop}$ (T node $-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (T network $-$unperturb)",
                 r"$\Delta\dot{Q}_\mathrm{hop}$ (T network $-$unperturb)",
                 r"$\Delta\dot{W}_\mathrm{hop}$ (T network $-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (perturb $\lambda-$unperturb)",
                 r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\lambda-$unperturb)",
                 r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\lambda-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (perturb $\phi-$unperturb)", r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\phi-$unperturb)",
                 r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\phi-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (perturb $\omega-$unperturb)", r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\omega-$unperturb)",
                 r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\omega-$unperturb)"] ]
    ylabel = r"frequency (# simulations)"

    for i in range(len(data_lists)):
        for j in range(len(data_lists[i])):
            data = data_lists[i][j]
            print 'xlabel is %s, data length is %d'%(xlabels[i][j], len(data))

            data_mean = np.mean(data)
            data_median = np.median(data)
            hist = np.histogram(data, bins='rice')
            hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
            hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.05

            axarr[i][j].text(0.15, 0.96, r"$\mu=$%.2f, $\tilde{\mu}=$%.2f" % (data_mean, data_median), transform=axarr[i][j].transAxes)

            axarr[i][j].bar(hist_coords, hist[0], color='cadetblue',
                            width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)

            axarr[i][j].set(ylabel=ylabel, xlabel=xlabels[i][j])

    unperturbed_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_unperturb_qdot_lists),axis=0)
    temp_node_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_temp_node_qdot_lists), axis=0)
    temp_network_hop_fraction_qdot_mean = np.mean(1. * np.array(hop_or_not_temp_network_qdot_lists), axis=0)
    perturb_phi_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_perturb_phi_qdot_lists), axis=0)
    perturb_freq_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_perturb_freq_qdot_lists), axis=0)

    axarr[0][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[0][3].plot(qdot_list, temp_node_hop_fraction_qdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='T node')
    axarr[0][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[0][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[0][3].axvline(x=orig_qdot_mean, color='gray', linestyle=':')
    axarr[0][3].legend()
    axarr[0][3].margins(0.05)

    axarr[1][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[1][3].plot(qdot_list, temp_network_hop_fraction_qdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='T network')
    axarr[1][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[1][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[1][3].axvline(x=orig_qdot_mean, color='gray', linestyle=':')
    axarr[1][3].legend()
    axarr[1][3].margins(0.05)

    axarr[2][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[2][3].plot(qdot_list, perturb_phi_hop_fraction_qdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='perturb forcing direction')
    axarr[2][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[2][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[2][3].axvline(x=orig_qdot_mean, color='gray', linestyle=':')
    axarr[2][3].legend()
    axarr[2][3].margins(0.05)

    axarr[3][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[3][3].plot(qdot_list, perturb_freq_hop_fraction_qdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='perturb frequency')
    axarr[3][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[3][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[3][3].axvline(x=orig_qdot_mean, color='gray', linestyle=':')
    axarr[3][3].legend()
    axarr[3][3].margins(0.05)

    unperturbed_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_unperturb_wdot_lists),axis=0)
    temp_node_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_temp_node_wdot_lists), axis=0)
    temp_network_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_temp_network_wdot_lists), axis=0)
    perturb_phi_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_phi_wdot_lists), axis=0)
    perturb_freq_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_freq_wdot_lists), axis=0)

    axarr[0][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[0][4].plot(wdot_list, temp_node_hop_fraction_wdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='T node')
    axarr[0][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[0][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[0][4].axvline(x=orig_wdot_mean, color='gray', linestyle=':')
    axarr[0][4].legend()
    axarr[0][4].margins(0.05)

    axarr[1][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[1][4].plot(wdot_list, temp_network_hop_fraction_wdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='T network')
    axarr[1][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[1][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[1][4].axvline(x=orig_wdot_mean, color='gray', linestyle=':')
    axarr[1][4].legend()
    axarr[1][4].margins(0.05)

    axarr[2][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[2][4].plot(wdot_list, perturb_phi_hop_fraction_wdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='perturb forcing direction')
    axarr[2][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[2][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[2][4].axvline(x=orig_wdot_mean, color='gray', linestyle=':')
    axarr[2][4].legend()
    axarr[2][4].margins(0.05)

    axarr[3][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':',
                     c='cadetblue', marker='o', markerfacecolor='cadetblue', markeredgecolor='None', markersize=5,
                     label='unperturbed')
    axarr[3][4].plot(wdot_list, perturb_freq_hop_fraction_wdot_mean, linestyle=':',
                     c='indianred', marker='o', markerfacecolor='indianred', markeredgecolor='None', markersize=5,
                     label='perturb frequency')
    axarr[3][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[3][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color='grey', alpha = 0.5)
    axarr[3][4].axvline(x=orig_wdot_mean, color='gray', linestyle=':')
    axarr[3][4].legend()
    axarr[3][4].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    suffix_string = suffix_string + str('_domega_%.2f'%(freq_change)).replace('.','p') + str('_dphi_%.2f'%(delta_phi)).replace('.','p')
    ofn = base_ofn+suffix_string+'_strong_perturb_analyse_qdot_wdot.png'
    fig.savefig(ofn)
    plt.close()


def read_sim_data_agg_plot_perturbed_data(data_dirs, suffix_string, rot_angle=np.pi, delta_phi=np.pi/3.,
                                          qdot_list = 0.05*np.array(range(1,51)), wdot_list = 0.03*np.array(range(1,151))):
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_perturbed_')
    seed2_list = list(set([ data_dir[data_dir.find('_seed2_')+7:data_dir.find('_soft_stiff')] for data_dir in data_dirs]))

    delta_t_hop_temp_network_unperturb = []
    delta_t_hop_perturb_eig_unperturb = []
    delta_t_hop_perturb_phi_unperturb = []
    delta_t_hop_perturb_freq_unperturb = []

    hop_or_not_unperturb_qdot_lists = []
    hop_or_not_temp_network_qdot_lists = []
    hop_or_not_perturb_eig_qdot_lists = []
    hop_or_not_perturb_phi_qdot_lists = []
    hop_or_not_perturb_freq_qdot_lists = []

    hop_or_not_unperturb_wdot_lists = []
    hop_or_not_temp_network_wdot_lists = []
    hop_or_not_perturb_eig_wdot_lists = []
    hop_or_not_perturb_phi_wdot_lists = []
    hop_or_not_perturb_freq_wdot_lists = []
    count = 0

    orig_wdot_end_list = []
    orig_qdot_end_list = []
    for seed2 in seed2_list:
        same_seed_diff_rep_dirs = filter(lambda data_dir: ('_seed2_'+seed2) in data_dir, data_dirs)
        rep_list = list(set([ data_dir[data_dir.find('_1e4_rep_')+9:data_dir.find('_1e4_rep_')+10] for data_dir in same_seed_diff_rep_dirs] ))
        for rep_str in rep_list:
            same_seed_dirs = filter(lambda data_dir: ('_1e4_rep_'+rep_str) in data_dir, same_seed_diff_rep_dirs)

            if len(same_seed_dirs) == 0:
                continue

            [orig_wdot_end, orig_qdot_end] = read_perturb_sim_data_return_qdot_wdot_orig(same_seed_dirs[0])
            orig_wdot_end_list.append(orig_wdot_end)
            orig_qdot_end_list.append(orig_qdot_end)

            temp_network_dirs = filter(lambda data_dir: '_perturb_w_noise_temp_network' in data_dir, same_seed_dirs)
            print '# temp network dirs is %d' %(len(temp_network_dirs))
            perturb_eig_dirs = filter(lambda data_dir: '_perturb_w_noise_degenerate_dirns' in data_dir, same_seed_dirs)
            print '# perturb dirn dirs is %d' %(len(perturb_eig_dirs))
            perturb_phi_dirs = filter(lambda data_dir: '_perturb_w_noise_delta_phi' in data_dir, same_seed_dirs)
            print '# perturb dirn dirs is %d' %(len(perturb_eig_dirs))
            perturb_freq_dirs = filter(lambda data_dir: '_perturb_w_noise_domega' in data_dir, same_seed_dirs)
            print '# perturb freq dirs is %d' %(len(perturb_freq_dirs))
            unperturbed_dirs = filter(lambda data_dir: '_unperturbed_w_noise' in data_dir, same_seed_dirs)
            print '# unperturb dirs is %d' %(len(unperturbed_dirs))
            for i, unperturbed_dir in enumerate(unperturbed_dirs):
                dfn = os.path.join(unperturbed_dir, 'first_hop_dict.json')
                with open(dfn, 'r') as f:
                    unperturbed_hop_dict = json.load(f)
                unperturbed_data_dict = dict()
                keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                             '<F(t).v(t)_by_(Barrier_by_T)>_T']
                for k in keys_list:
                    dfn = os.path.join(unperturbed_dir, str(k).replace('/', '_by_') + '.json')
                    with open(dfn, 'r') as f:
                        unperturbed_data_dict[k] = np.array(json.load(f))
                        if np.sum(np.isnan(unperturbed_data_dict[k]))>0:
                            print 'unperturbed sims have NaN values'
                            print 'PROBLEMATIC DIR IS %s'%(unperturbed_dir)
                            continue
                        if np.sum(np.isinf(unperturbed_data_dict[k]))>0:
                            print 'unperturbed sims have Inf values'
                            print 'PROBLEMATIC DIR IS %s'%(unperturbed_dir)
                            continue

                unperturbed_qdot_indices = [ getLeftIndex(unperturbed_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                          qdot_list[i]) for i in range(qdot_list.size)]
                hop_or_not_unperturbed_qdot_list = [ np.int(np.sum(unperturbed_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in unperturbed_qdot_indices]
                hop_or_not_unperturb_qdot_lists.append(hop_or_not_unperturbed_qdot_list)

                unperturbed_wdot_indices = [ getLeftIndex(unperturbed_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                          wdot_list[i]) for i in range(wdot_list.size)]
                hop_or_not_unperturbed_wdot_list = [ np.int(np.sum(unperturbed_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                            >0.1) for i in unperturbed_wdot_indices]
                hop_or_not_unperturb_wdot_lists.append(hop_or_not_unperturbed_wdot_list)

                for j, perturb_dir in enumerate(temp_network_dirs):
                    dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                    with open(dfn, 'r') as f:
                        perturb_hop_dict = json.load(f)
                    perturb_data_dict = dict()
                    keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                                 '<F(t).v(t)_by_(Barrier_by_T)>_T']
                    for k in keys_list:
                        dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                        with open(dfn, 'r') as f:
                            perturb_data_dict[k] = np.array(json.load(f))
                            if np.sum(np.isnan(perturb_data_dict[k]))>0:
                                print 'temperature sims have NaN values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue
                            if np.sum(np.isnan(perturb_data_dict[k]))>0:
                                print 'temperature sims have Inf values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue


                    if perturb_hop_dict['first_hop_time'] > 1999.:
                        print '%s doesn\'t hop' % (perturb_dir)
                        # continue
                    print 'data dir is %s'%(perturb_dir)
                    delta_t_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                    # delta_Qdot_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                    # delta_Wdot_hop_temp_network_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                    perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                             qdot_list[i]) for i in range(qdot_list.size)]
                    hop_or_not_temp_network_qdot_list = [
                        np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                               > 0.1) for i in perturb_qdot_indices]
                    hop_or_not_temp_network_qdot_lists.append(hop_or_not_temp_network_qdot_list)

                    perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                              wdot_list[i]) for i in range(wdot_list.size)]
                    hop_or_not_temp_network_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                                >0.1) for i in perturb_wdot_indices ]
                    hop_or_not_temp_network_wdot_lists.append(hop_or_not_temp_network_wdot_list)

                for j, perturb_dir in enumerate(perturb_eig_dirs):
                    dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                    with open(dfn, 'r') as f:
                        perturb_hop_dict = json.load(f)
                    perturb_data_dict = dict()
                    keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                                 '<F(t).v(t)_by_(Barrier_by_T)>_T']
                    for k in keys_list:
                        dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                        with open(dfn, 'r') as f:
                            perturb_data_dict[k] = np.array(json.load(f))
                            if np.sum(np.isnan(perturb_data_dict[k]))>0:
                                print 'eig sims have NaN values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue
                            if np.sum(np.isinf(perturb_data_dict[k]))>0:
                                print 'eig sims have Inf values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue

                    if perturb_hop_dict['first_hop_time'] > 1999.:
                        print '%s doesn\'t hop' % (perturb_dir)
                        # continue

                    delta_t_hop_perturb_eig_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                    # delta_Qdot_hop_perturb_eig_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                    # delta_Wdot_hop_perturb_eig_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                    perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                             qdot_list[i]) for i in range(qdot_list.size)]
                    hop_or_not_perturb_eig_qdot_list = [
                        np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                               > 0.1) for i in perturb_qdot_indices]
                    hop_or_not_perturb_eig_qdot_lists.append(hop_or_not_perturb_eig_qdot_list)

                    perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                              wdot_list[i]) for i in range(wdot_list.size)]
                    hop_or_not_perturb_eig_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                                >0.1) for i in perturb_wdot_indices ]
                    hop_or_not_perturb_eig_wdot_lists.append(hop_or_not_perturb_eig_wdot_list)

                for j, perturb_dir in enumerate(perturb_phi_dirs):
                    dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                    with open(dfn, 'r') as f:
                        perturb_hop_dict = json.load(f)
                    perturb_data_dict = dict()
                    keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                                 '<F(t).v(t)_by_(Barrier_by_T)>_T']
                    for k in keys_list:
                        dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                        with open(dfn, 'r') as f:
                            perturb_data_dict[k] = np.array(json.load(f))
                            if np.sum(np.isnan(perturb_data_dict[k]))>0:
                                print 'phi sims have NaN values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue
                            if np.sum(np.isinf(perturb_data_dict[k]))>0:
                                print 'phi sims have Inf values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue

                    if perturb_hop_dict['first_hop_time'] > 1999.:
                        print '%s doesn\'t hop' % (perturb_dir)
                        # continue

                    delta_t_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                    # delta_Qdot_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                    # delta_Wdot_hop_perturb_phi_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                    perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                             qdot_list[i]) for i in range(qdot_list.size)]
                    hop_or_not_perturb_phi_qdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                               > 0.1) for i in perturb_qdot_indices]
                    hop_or_not_perturb_phi_qdot_lists.append(hop_or_not_perturb_phi_qdot_list)

                    perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                              wdot_list[i]) for i in range(wdot_list.size)]
                    hop_or_not_perturb_phi_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                                >0.1) for i in perturb_wdot_indices ]
                    hop_or_not_perturb_phi_wdot_lists.append(hop_or_not_perturb_phi_wdot_list)

                for j, perturb_dir in enumerate(perturb_freq_dirs):
                    # perturb_period = np.float64(str(perturb_dir[perturb_dir.find('_cycles_T_')+10:perturb_dir.find('_cycles_T_')+14]).replace('p','.'))
                    dfn = os.path.join(perturb_dir, 'first_hop_dict.json')
                    with open(dfn, 'r') as f:
                        perturb_hop_dict = json.load(f)
                    perturb_data_dict = dict()
                    keys_list = ['t/T', '<# spring transitions(t)>_T', '<dissipation_rate(t)/(Barrier/T)>_T', '<F(t).v(t)_by_(Barrier_by_T)>_T']
                    for k in keys_list:
                        dfn = os.path.join(perturb_dir, str(k).replace('/', '_by_') + '.json')
                        with open(dfn, 'r') as f:
                            perturb_data_dict[k] = np.array(json.load(f))
                            if np.sum(np.isnan(perturb_data_dict[k]))>0:
                                print 'freq sims have NaN values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue
                            if np.sum(np.isinf(perturb_data_dict[k]))>0:
                                print 'freq sims have Inf values'
                                print 'PROBLEMATIC DIR IS %s'%(perturb_dir)
                                continue

                    if perturb_hop_dict['first_hop_time'] > 1999.:
                        print '%s doesn\'t hop' % (perturb_dir)
                        # continue
                    delta_t_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_time']-unperturbed_hop_dict['first_hop_time'])
                    # delta_Qdot_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_Qdot_avg']-unperturbed_hop_dict['first_hop_Qdot_avg'])
                    # delta_Wdot_hop_perturb_freq_unperturb.append(perturb_hop_dict['first_hop_Wdot_avg']-unperturbed_hop_dict['first_hop_Wdot_avg'])

                    perturb_qdot_indices = [getLeftIndex(perturb_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'],
                                                             qdot_list[i]) for i in range(qdot_list.size)]
                    hop_or_not_perturb_freq_qdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                               > 0.1) for i in perturb_qdot_indices]
                    hop_or_not_perturb_freq_qdot_lists.append(hop_or_not_perturb_freq_qdot_list)

                    perturb_wdot_indices = [ getLeftIndex(perturb_data_dict['<F(t).v(t)_by_(Barrier_by_T)>_T'],
                                                              wdot_list[i]) for i in range(wdot_list.size)]
                    hop_or_not_perturb_freq_wdot_list = [ np.int(np.sum(perturb_data_dict['<# spring transitions(t)>_T'][:i+1])
                                                                >0.1) for i in perturb_wdot_indices ]
                    hop_or_not_perturb_freq_wdot_lists.append(hop_or_not_perturb_freq_wdot_list)

    orig_wdot_end_list = np.array(orig_wdot_end_list)
    ref_y = np.linspace(0.,0.1,11)
    ref_ones = np.ones(ref_y.size)
    orig_wdot_mean = np.mean(orig_wdot_end_list)
    orig_wdot_std = np.std(orig_wdot_end_list)
    orig_qdot_end_list = np.array(orig_qdot_end_list)
    orig_qdot_mean = np.mean(orig_qdot_end_list)
    orig_qdot_std = np.std(orig_qdot_end_list)

    aux_data_dict = dict()
    aux_data_dict['ref_y'] = ref_y
    aux_data_dict['orig_wdot_mean'] = orig_wdot_mean
    aux_data_dict['orig_wdot_std'] = orig_wdot_std
    aux_data_dict['orig_qdot_mean'] = orig_qdot_mean
    aux_data_dict['orig_qdot_std'] = orig_qdot_std

    unperturbed_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_unperturb_qdot_lists),axis=0)
    temp_network_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_temp_network_qdot_lists), axis=0)
    perturb_eig_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_perturb_eig_qdot_lists), axis=0)
    perturb_phi_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_perturb_phi_qdot_lists), axis=0)
    perturb_freq_hop_fraction_qdot_mean = np.mean(1.*np.array(hop_or_not_perturb_freq_qdot_lists), axis=0)

    ########################################################################################################################
    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'aux_data_dict.json')
    with open(dfn, 'w') as f:
        json.dump(aux_data_dict, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'qdot_list.json')
    with open(dfn, 'w') as f:
        json.dump(qdot_list, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'unperturbed_hop_fraction_qdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(unperturbed_hop_fraction_qdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_eig_hop_fraction_qdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_eig_hop_fraction_qdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_phi_hop_fraction_qdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_phi_hop_fraction_qdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_freq_hop_fraction_qdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_freq_hop_fraction_qdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'temp_network_hop_fraction_qdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(temp_network_hop_fraction_qdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    ########################################################################################################################

    unperturbed_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_unperturb_wdot_lists),axis=0)
    perturb_eig_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_eig_wdot_lists), axis=0)
    perturb_phi_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_phi_wdot_lists), axis=0)
    perturb_freq_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_freq_wdot_lists), axis=0)
    temp_network_hop_fraction_wdot_mean = np.mean(1.*np.array(hop_or_not_perturb_eig_wdot_lists), axis=0)

    ########################################################################################################################
    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'wdot_list.json')
    with open(dfn, 'w') as f:
        json.dump(wdot_list, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'unperturbed_hop_fraction_wdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(unperturbed_hop_fraction_wdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_eig_hop_fraction_wdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_eig_hop_fraction_wdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_phi_hop_fraction_wdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_phi_hop_fraction_wdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_freq_hop_fraction_wdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(perturb_freq_hop_fraction_wdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'temp_network_hop_fraction_wdot_mean.json')
    with open(dfn, 'w') as f:
        json.dump(temp_network_hop_fraction_wdot_mean, f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2,
                  separators=(',', ': '))
    ########################################################################################################################

    fig = plt.figure(figsize=(50, 40), dpi=200)
    gs = gridspec.GridSpec(4, 5)
    axarr = [ [ plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3]), plt.subplot(gs[0,4])],
              [ plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4])],
              [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4])],
              [plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2]), plt.subplot(gs[3, 3]), plt.subplot(gs[3, 4])] ]
    data_lists = [[delta_t_hop_perturb_eig_unperturb], #delta_Qdot_hop_perturb_eig_unperturb, delta_Wdot_hop_perturb_eig_unperturb],
                  [delta_t_hop_perturb_phi_unperturb], #delta_Qdot_hop_perturb_phi_unperturb, delta_Wdot_hop_perturb_phi_unperturb],
                  [delta_t_hop_perturb_freq_unperturb], #delta_Qdot_hop_perturb_freq_unperturb, delta_Wdot_hop_perturb_freq_unperturb],
                  [delta_t_hop_temp_network_unperturb], #delta_Qdot_hop_temp_network_unperturb, delta_Wdot_hop_temp_network_unperturb]
                  ]
    xlabels = [ [r"$\Delta t_\mathrm{hop}$ (perturb $\lambda-$unperturb)"], #r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\lambda-$unperturb)",
                  # r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\lambda-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (perturb $\phi-$unperturb)"], #r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\phi-$unperturb)",
                  # r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\phi-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ (perturb $\omega-$unperturb)"], #r"$\Delta\dot{Q}_\mathrm{hop}$ (perturb $\omega-$unperturb)",
                  # r"$\Delta\dot{W}_\mathrm{hop}$ (perturb $\omega-$unperturb)"],
                [r"$\Delta t_\mathrm{hop}$ ($\Delta T-$unperturb)"], #r"$\Delta\dot{Q}_\mathrm{hop}$ ($\Delta T-$unperturb)",
                  # r"$\Delta\dot{W}_\mathrm{hop}$ ($\Delta T-$unperturb)"],
                ]
    ylabel = r"frequency (# simulations)"

    for i in range(len(data_lists)):
        for j in range(len(data_lists[i])):
            data = data_lists[i][j]
            nanlist=[]
            for ii in range(len(data)):
                if np.isnan(data[ii]):
                    nanlist.append(ii)
            print '# nans is %d, nan indices are:'%(len(nanlist))
            print nanlist

            for ii, nanindex in enumerate(nanlist):
                data.pop(nanindex-ii)

            print 'xlabel is %s, data length is %d'%(xlabels[i][j], len(data))
            print 'min data is %.1f, max data is %.1f'%(min(data), max(data))
            data_mean = np.mean(data)
            data_median = np.median(data)

            hist = np.histogram(data, bins='fd')
            hist_coords = (hist[1][:-1] + hist[1][1:]) / 2.
            hist_width = (hist[1][1] - hist[1][0]) if len(hist[1]) > 2 else 0.05

            axarr[i][j].text(0.15, 0.96, r"$\mu=$%.2f, $\tilde{\mu}=$%.2f" % (data_mean, data_median), transform=axarr[i][j].transAxes)

            axarr[i][j].bar(hist_coords, hist[0], color='cadetblue',
                            width=hist_width, edgecolor='k', linewidth=1, alpha=0.7)

            axarr[i][j].set(ylabel=ylabel, xlabel=xlabels[i][j])

    cm = plt.cm.get_cmap('Blues')
    c_orig = cm(0.6)

    axarr[0][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][3].plot(qdot_list, perturb_eig_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb degenerate eigendirections')
    axarr[0][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[0][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][3].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    axarr[0][3].legend()
    axarr[0][3].margins(0.05)

    axarr[1][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][3].plot(qdot_list, perturb_phi_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb forcing direction')
    axarr[1][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[1][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][3].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    axarr[1][3].legend()
    axarr[1][3].margins(0.05)

    axarr[2][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[2][3].plot(qdot_list, perturb_freq_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb frequency')
    axarr[2][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[2][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[2][3].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    axarr[2][3].legend()
    axarr[2][3].margins(0.05)

    axarr[3][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[3][3].plot(qdot_list, temp_network_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='firebrick', marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=7,
                     label='temp network')
    axarr[3][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[3][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[3][3].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    # axarr[3][3].legend()
    axarr[3][3].margins(0.05)

    axarr[0][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][4].plot(wdot_list, perturb_eig_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb degenerate eigendirections')
    axarr[0][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[0][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][4].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    # axarr[0][4].legend()
    axarr[0][4].margins(0.05)

    axarr[1][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][4].plot(wdot_list, perturb_phi_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb forcing direction')
    axarr[1][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[1][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][4].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    # axarr[1][4].legend()
    axarr[1][4].margins(0.05)

    axarr[2][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[2][4].plot(wdot_list, perturb_freq_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='olivedrab', marker='o', markerfacecolor='olivedrab', markeredgecolor='None', markersize=7,
                     label='perturb frequency')
    axarr[2][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[2][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[2][4].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    # axarr[2][4].legend()
    axarr[2][4].margins(0.05)

    axarr[3][4].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[3][4].plot(wdot_list, temp_network_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='firebrick', marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=7,
                     label='temperature network')
    axarr[3][4].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[3][4].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[3][4].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='-.', lw=2)
    # axarr[3][4].legend()
    axarr[3][4].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    suffix_string = suffix_string + str('_rot_%.2f'%(rot_angle)).replace('.','p') + str('_dphi_%.2f'%(delta_phi)).replace('.','p')
    ofn = base_ofn+suffix_string+'_perturb_analyse_qdot_wdot_w_noise.png'
    fig.savefig(ofn)
    plt.close()

def read_plot_perturb_analyse_data(base_dir):

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'aux_data_dict.json')
    with open(dfn, 'r') as f:
        aux_data_dict = json.load(f)

    ref_y = np.array(aux_data_dict['ref_y'])
    ref_ones = np.ones(ref_y.size)
    orig_wdot_mean = aux_data_dict['orig_wdot_mean']
    orig_wdot_std = aux_data_dict['orig_wdot_std']
    orig_qdot_mean = aux_data_dict['orig_qdot_mean']
    orig_qdot_std = aux_data_dict['orig_qdot_std']

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'qdot_list.json')
    with open(dfn, 'r') as f:
        qdot_list = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'unperturbed_hop_fraction_qdot_mean.json')
    with open(dfn, 'r') as f:
        unperturbed_hop_fraction_qdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_eig_hop_fraction_qdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_eig_hop_fraction_qdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_phi_hop_fraction_qdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_phi_hop_fraction_qdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_freq_hop_fraction_qdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_freq_hop_fraction_qdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'temp_network_hop_fraction_qdot_mean.json')
    with open(dfn, 'r') as f:
        temp_network_hop_fraction_qdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/', 'wdot_list.json')
    with open(dfn, 'r') as f:
        wdot_list = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'unperturbed_hop_fraction_wdot_mean.json')
    with open(dfn, 'r') as f:
        unperturbed_hop_fraction_wdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_eig_hop_fraction_wdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_eig_hop_fraction_wdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_phi_hop_fraction_wdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_phi_hop_fraction_wdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'perturb_freq_hop_fraction_wdot_mean.json')
    with open(dfn, 'r') as f:
        perturb_freq_hop_fraction_wdot_mean = np.array(json.load(f))

    dfn = os.path.join('/Volumes/wd/hridesh/force_drive_sims/perturb_analyse_data/',
                       'temp_network_hop_fraction_wdot_mean.json')
    with open(dfn, 'r') as f:
        temp_network_hop_fraction_wdot_mean = np.array(json.load(f))

    fig = plt.figure(figsize=(40, 20), dpi=200)
    gs = gridspec.GridSpec(2, 4)
    axarr = [ [ plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3])],
              [ plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3])] ]
    cm = plt.cm.get_cmap('Blues')
    c_orig = cm(0.6)

    axarr[0][0].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][0].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[0][0].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][0].plot(qdot_list, perturb_eig_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb degenerate eigendirections')
    axarr[0][0].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[0][0].legend()
    axarr[0][0].margins(0.05)

    axarr[0][1].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][1].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[0][1].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][1].plot(qdot_list, perturb_phi_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb forcing direction')
    axarr[0][1].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[0][1].legend()
    axarr[0][1].margins(0.05)

    axarr[0][2].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][2].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[0][2].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][2].plot(qdot_list, perturb_freq_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb frequency')
    axarr[0][2].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    axarr[0][2].legend()
    axarr[0][2].margins(0.05)

    axarr[0][3].fill_betweenx(ref_y, (orig_qdot_mean-orig_qdot_std)*ref_ones, (orig_qdot_mean+orig_qdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[0][3].axvline(x=orig_qdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[0][3].plot(qdot_list, unperturbed_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[0][3].plot(qdot_list, temp_network_hop_fraction_qdot_mean, linestyle=':', lw=2,
                     c='firebrick', marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=7,
                     label='temp network')
    axarr[0][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{Q}$")
    # axarr[0][3].legend()
    axarr[0][3].margins(0.05)

    axarr[1][0].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][0].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    # axarr[1][0].legend()
    axarr[1][0].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][0].plot(wdot_list, perturb_eig_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb degenerate eigendirections')
    axarr[1][0].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")

    axarr[1][0].margins(0.05)

    axarr[1][1].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][1].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[1][1].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][1].plot(wdot_list, perturb_phi_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb forcing direction')
    axarr[1][1].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    # axarr[1][1].legend()
    axarr[1][1].margins(0.05)

    axarr[1][2].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][2].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    # axarr[1][2].legend()
    axarr[1][2].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][2].plot(wdot_list, perturb_freq_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='darkgreen', marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=7,
                     label='perturb frequency')
    axarr[1][2].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    axarr[1][2].margins(0.05)

    axarr[1][3].fill_betweenx(ref_y, (orig_wdot_mean-orig_wdot_std)*ref_ones, (orig_wdot_mean+orig_wdot_std)*ref_ones,
                               color=c_orig, alpha = 0.5)
    axarr[1][3].axvline(x=orig_wdot_mean, ymin=0.05, ymax=0.15, color=cm(0.75), linestyle='--', lw=4)
    axarr[1][3].plot(wdot_list, unperturbed_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c=c_orig, marker='o', markerfacecolor=c_orig, markeredgecolor='None', markersize=7,
                     label='unperturbed')
    axarr[1][3].plot(wdot_list, temp_network_hop_fraction_wdot_mean, linestyle=':', lw=2,
                     c='firebrick', marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=7,
                     label='temperature network')
    axarr[1][3].set(ylabel=r"probability of hopping",
                    xlabel=r"$\dot{W}$")
    # axarr[1][3].legend()
    axarr[1][3].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = os.path.join(base_dir, 'perturb_analyse_plot_qdot_wdot_w_noise_w_more_data.png')
    fig.savefig(ofn)
    plt.close()


def read_sim_data_setup_perturbed_ramp_amp_sims(data_dir, ramped_amp=15., num_cycles=3000,
                                                rot_angle=np.pi, delta_phi=np.pi/3., delta_omega=0.33):
    dfn = os.path.join(data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['PLOT_DIR'] = data_dir
    dfn = os.path.join(data_dir, 'sampled_xilist.json')
    with open(dfn, 'r') as f:
        plot_xilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'sampled_yilist.json')
    with open(dfn, 'r') as f:
        plot_yilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'sampled_vxilist.json')
    with open(dfn, 'r') as f:
        plot_vxilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'sampled_vyilist.json')
    with open(dfn, 'r') as f:
        plot_vyilist_tsteps = np.array(json.load(f))
    dfn = os.path.join(data_dir, 'adjacency_data.json')
    with open(dfn, 'r') as f:
        adj_data = json.load(f)
    ERGRAPH = json_graph.adjacency_graph(adj_data)

    xilist_0 = plot_xilist_tsteps[-1]
    yilist_0 = plot_yilist_tsteps[-1]
    vxilist_0 = plot_vxilist_tsteps[-1]
    vyilist_0 = plot_vyilist_tsteps[-1]

    unperturbed_phi = params_dict['phi_values_list'][0]
    forcing_direction = np.zeros(2*params_dict['NODES'])
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']] = np.cos(unperturbed_phi)
    forcing_direction[2*params_dict['MAX_DEG_VERTEX_I']+1] = np.sin(unperturbed_phi)
    params_dict['FORCING_VEC'] = forcing_direction

    [xilist_0, yilist_0, vxilist_0, vyilist_0, total_spring_flips] = run_undriven_sim(params_dict, xilist_0, yilist_0, vxilist_0, vyilist_0)

    print 'data dir is %s, total spring flips during relaxation is %d' % (data_dir, total_spring_flips)

    if total_spring_flips > 1e-2:
        print 'relaxation of end state leads to change of state, data dir is %s'%(data_dir)
        dfn = os.path.join(params_dict['PLOT_DIR'], 'relaxation_spring_flips.json')
        with open(dfn, 'w') as f:
            json.dump([total_spring_flips], f, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=2, separators=(',', ': '))
        return data_dir
    else:
        unperturbed_string = '_unperturbed_w_noise'
        perturb_string = '_perturb_w_noise'
        Amat = np.array(params_dict['AMAT'],dtype=np.int64)
        params_dict['ANCHORED_NODES'] = np.array(params_dict['ANCHORED_NODES'])
        forcing_period = params_dict['FORCE_PERIOD']

        perturbed_force_periods = forcing_period*np.array([1./(1+delta_omega), 1./(1- delta_omega)])
        params_dict['FORCE_AMP_SLOPE'] = ramped_amp/(params_dict['FORCE_PERIOD']*num_cycles)
        params_dict['T_END'] = (params_dict['FORCE_PERIOD']*num_cycles)
        params_dict['T_RELAX'] = 0.
        params_dict['START_PLOT_INDEX'] = 0
        params_dict['freq_switch_times'] = np.array([params_dict['T_END']])

        for i in range(perturbed_force_periods.size):
            perturbed_force_period = perturbed_force_periods[i]
            perturbed_params_dict = dict(params_dict)
            perturbed_params_dict['FORCE_PERIOD'] = perturbed_force_period
            run_ramped_perturbed_force_sim_save_plots(perturbed_params_dict, ERGRAPH, xilist_0, yilist_0, vxilist_0,
                                                      vyilist_0, num_cycles,
                                                      (perturb_string+'_domega_0p33' + str('_%d_cycles' % (num_cycles))+
                                                      str('_T_%.2f'%(perturbed_force_period)).replace('.','p')))

        return data_dir

def read_sim_data_agg_plot_dwell_data(data_dirs, suffix_string):
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    init_data_dir = data_dirs[0]

    keys_arbitrary_length = ['dwell_times', 'dwell_diss_rate_actual', 'dwell_work_rate_actual',
                        'dwell_diverging_dims', 'dwell_max_diverging_rates', 'dwell_E_avg', 'dwell_E_min',
                        'dwell_E_exit', 'dwell_deltaE' ]

    for k in keys_arbitrary_length:
        dfn = os.path.join(init_data_dir, str(k) + '.json')
        with open(dfn,'r') as f:
            agg_plot_data_dict[k] = []
            agg_plot_data_dict[k].extend(np.array(json.load(f)).tolist())

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_arbitrary_length:
            dfn = os.path.join(data_dir, str(k) + '.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k].extend(np.array(json.load(f)).tolist())

    min_dwell = 3. ## in units of force period
    dwell_keystring = '_long_dwell'
    long_dwell_indices = [ i for i in range(len(agg_plot_data_dict['dwell_times'])) if agg_plot_data_dict['dwell_times'][i] > min_dwell ]

    for k in keys_arbitrary_length:
        new_k = k + dwell_keystring
        agg_plot_data_dict[new_k] = np.take(agg_plot_data_dict[k], long_dwell_indices, axis=0)

    max_div_dim = 1. ## in units of barrier energy
    div_dim_keystring = '_small_div_dim'
    small_div_dim_indices = [ i for i in range(len(agg_plot_data_dict['dwell_diverging_dims'])) if (agg_plot_data_dict['dwell_diverging_dims'][i] < max_div_dim and
                                                                                                    agg_plot_data_dict['dwell_times'][i] > min_dwell)]
    for k in keys_arbitrary_length:
        agg_plot_data_dict[k] = np.array(agg_plot_data_dict[k])
        new_k = k + div_dim_keystring
        agg_plot_data_dict[new_k] = np.take(agg_plot_data_dict[k], small_div_dim_indices, axis=0)


    fig = plt.figure(figsize=(40, 30), dpi=100)
    gs = gridspec.GridSpec(3, 4)
    axarr = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3])],
             [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3])],
             [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(gs[2, 3])], ]
####### choosing work rate as colormap #########
    cm = plt.cm.get_cmap('jet')
    E_min_max = np.amax(agg_plot_data_dict['dwell_E_min'])
    c_Emin = [cm(n/E_min_max) for n in agg_plot_data_dict['dwell_E_min']]

    axarr[0][0].scatter(agg_plot_data_dict['dwell_E_min'], agg_plot_data_dict['dwell_times'],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[0][0].set(xlabel=r"$E_\mathrm{min}/E_b$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$")
    axarr[0][0].set_yscale('log')
    axarr[0][0].margins(0.05)

    axarr[0][1].scatter(agg_plot_data_dict['dwell_E_avg'], agg_plot_data_dict['dwell_times'],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[0][1].set(xlabel=r"$\bar{E}/E_b$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$")
    axarr[0][1].set_yscale('log')
    axarr[0][1].margins(0.05)

    axarr[0][2].scatter(agg_plot_data_dict['dwell_E_exit'] - agg_plot_data_dict['dwell_E_min'],
                        agg_plot_data_dict['dwell_times'], c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[0][2].set(xlabel=r"$(E_\mathrm{exit}-E_\mathrm{min})/E_b$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$")
    axarr[0][2].set_yscale('log')
    axarr[0][2].margins(0.05)

    axarr[0][3].scatter(agg_plot_data_dict['dwell_deltaE'], agg_plot_data_dict['dwell_times'],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[0][3].set(xlabel=r"$(E_f - E_i)/E_b$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$")
    axarr[0][3].set_yscale('log')
    axarr[0][3].margins(0.05)

    cm = plt.cm.get_cmap('jet')
    E_min_max = np.amax(agg_plot_data_dict['dwell_E_min' + div_dim_keystring])
    c_Emin = [cm(n/E_min_max) for n in agg_plot_data_dict['dwell_E_min'+ div_dim_keystring]]

    axarr[1][0].scatter(agg_plot_data_dict['dwell_work_rate_actual' + div_dim_keystring],
                        agg_plot_data_dict['dwell_times'+ div_dim_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[1][0].set(xlabel=r"$\dot{W}/(E_b/\tau)$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (div. dims < %.1f, dwell > %d)"%(max_div_dim, min_dwell))
    axarr[1][0].set_yscale('log')
    axarr[1][0].margins(0.05)

    axarr[1][1].scatter(agg_plot_data_dict['dwell_diss_rate_actual' + div_dim_keystring],
                        agg_plot_data_dict['dwell_times'+ div_dim_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[1][1].set(xlabel=r"$\dot{Q}/(E_b/\tau)$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (div. dims < %.1f, dwell > %d)"%(max_div_dim, min_dwell))
    axarr[1][1].set_yscale('log')
    axarr[1][1].margins(0.05)

    axarr[1][2].scatter(agg_plot_data_dict['dwell_diverging_dims' + div_dim_keystring],
                        agg_plot_data_dict['dwell_times'+ div_dim_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[1][2].set(xlabel=r"# div. dim.s (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (div. dims < %.1f, dwell > %d)"%(max_div_dim, min_dwell))
    axarr[1][2].set_yscale('log')
    axarr[1][2].margins(0.05)

    axarr[1][3].scatter(agg_plot_data_dict['dwell_max_diverging_rates' + div_dim_keystring],
                        agg_plot_data_dict['dwell_times'+ div_dim_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[1][3].set(xlabel=r"$\lambda_\mathrm{max}^{+}\tau$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (div. dims < %.1f, dwell > %d)"%(max_div_dim, min_dwell))
    axarr[1][3].set_yscale('log')
    axarr[1][3].margins(0.05)

    cm = plt.cm.get_cmap('jet')
    # E_min_max = np.amax(agg_plot_data_dict['dwell_E_min' + dwell_keystring])
    c_Emin = [cm(n/E_min_max) for n in agg_plot_data_dict['dwell_E_min'+ dwell_keystring]]

    axarr[2][0].scatter(np.abs(agg_plot_data_dict['dwell_work_rate_actual' + dwell_keystring]),
                        agg_plot_data_dict['dwell_times'+ dwell_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[2][0].set(xlabel=r"$\vert\dot{W}\vert/(E_b/\tau)$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (dwell > %.1f)"%(min_dwell))
    axarr[2][0].set_yscale('log')
    axarr[2][0].margins(0.05)

    axarr[2][1].scatter(agg_plot_data_dict['dwell_diss_rate_actual' + dwell_keystring],
                        agg_plot_data_dict['dwell_times'+ dwell_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[2][1].set(xlabel=r"$\dot{Q}/(E_b/\tau)$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (dwell > %.1f)"%(min_dwell))
    axarr[2][1].set_yscale('log')
    axarr[2][1].margins(0.05)

    axarr[2][2].scatter(agg_plot_data_dict['dwell_diverging_dims' + dwell_keystring],
                        agg_plot_data_dict['dwell_times'+ dwell_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[2][2].set(xlabel=r"# div. dim.s (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (dwell > %.1f)"%(min_dwell))
    axarr[2][2].set_yscale('log')
    axarr[2][2].margins(0.05)

    axarr[2][3].scatter(agg_plot_data_dict['dwell_max_diverging_rates' + dwell_keystring],
                        agg_plot_data_dict['dwell_times'+ dwell_keystring],
                        c=c_Emin, marker='o', s=16, alpha=0.6)
    axarr[2][3].set(xlabel=r"$\lambda_\mathrm{max}^{+}\tau$ (c: $E_\mathrm{min}$)",ylabel=r"dwell time/$\tau$, (dwell > %.1f)"%(min_dwell))
    axarr[2][3].set_yscale('log')
    axarr[2][3].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_dwell_data_bare_max_div_dim_%.1f_minDwell_%d' %(max_div_dim, min_dwell)
    ofn = ofn.replace('.','p')
    ofn = ofn + '.png'
    fig.savefig(ofn)
    plt.close()
##############################################################################################################################################

def check_diss_rate_decay(data_dirs, suffix_string):
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    keys_list = [ '<dissipation_rate(t)/(Barrier/T)>_T' ]

    num_dirs = len(data_dirs)
    init_data_dir = data_dirs[0]

    cm = plt.cm.get_cmap('Blues')

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['COLORS'] = [cm(0.6), cm(0.95)]
    # params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']))
    for k in keys_list:
        dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
            agg_plot_data_dict[k][0] = k_val

    dfn = os.path.join(init_data_dir, str('t/T').replace('/','_by_')+'.json')
    with open(dfn,'r') as f:
        agg_plot_data_dict['t/T'] = np.array(json.load(f))

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_list:
            dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k][i+1] = np.array(json.load(f))

    diss_rate_mean = np.mean(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)
    diss_rate_std = np.std(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)

    fig = plt.figure(figsize=(30,10), dpi=200)
    gs = gridspec.GridSpec(1, 3)
    axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]) ] ]
    start_index = params_dict['START_PLOT_INDEX']
    drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']
    zeros_ref = np.zeros(diss_rate_mean[start_index:].size)
    max_diss_rate = np.amax(diss_rate_mean)
    max_diss_idx = find_nearest_idx(diss_rate_mean, max_diss_rate)
    drive_start_idx = find_nearest_idx(agg_plot_data_dict['t/T'], drive_start_t)
    diss_rate_on = max_diss_rate
    min_diss_rate = np.amin(diss_rate_mean[max_diss_idx:])
    diss_rate_off = min_diss_rate
    print 'drive_start_idx is %d, max_diss_idx is %d'%(drive_start_idx, max_diss_idx)

    off_rate_cdf = (diss_rate_on - diss_rate_mean)/(diss_rate_on - diss_rate_off)
    off_rate_cdf = off_rate_cdf[max_diss_idx:]
    off_rate_cdf_tlist = agg_plot_data_dict['t/T'][max_diss_idx:]
    smoothed_off_rate_cdf = savgol_filter(off_rate_cdf, 51, 3)
    off_rate_pdf = (smoothed_off_rate_cdf[1:] - smoothed_off_rate_cdf[:-1])/(agg_plot_data_dict['t/T'][1] - agg_plot_data_dict['t/T'][0])
    off_rate_pdf_tlist = agg_plot_data_dict['t/T'][max_diss_idx:-1]

    ############ exp fit ##############
    coeff, fit_cov = curve_fit(lambda t,a,b: max_diss_rate - (max_diss_rate-min_diss_rate)*a*(1-np.exp(-b*t)), off_rate_cdf_tlist - off_rate_cdf_tlist[0], diss_rate_mean[max_diss_idx:], p0=(0.2, 0.05))
    a = coeff[0]
    b = coeff[1]
    print 'a*exp(-b (t-t_0)), a is %.4f, b is %.4f'%(a,b)
    exp_fit_off_rate_pdf = (a*b)*np.exp(-b*(off_rate_pdf_tlist-off_rate_pdf_tlist[0]))
    exp_fit_off_rate_cdf = a*(1-np.exp(-b*(off_rate_cdf_tlist-off_rate_cdf_tlist[0])))
    ones_ref = np.ones(off_rate_pdf_tlist.size)
    exp_fit_diss_rate = max_diss_rate - (max_diss_rate-min_diss_rate)*a*(1-np.exp(-b*(off_rate_cdf_tlist-off_rate_cdf_tlist[0])))

    exp_a = a
    exp_b = b

    ############ power law fit ##############
    coeff, fit_cov = curve_fit(lambda t,a,b: max_diss_rate - (max_diss_rate-min_diss_rate)*a*(np.power(off_rate_cdf_tlist[0],-b)-np.power(t,-b)), off_rate_cdf_tlist, diss_rate_mean[max_diss_idx:], p0=(10, 0.5))
    a = coeff[0]
    b = coeff[1]
    print 'a*t^(-b), a is %.4f, b is %.4f'%(a,b)
    pow_fit_off_rate_pdf = (a*b)*np.power(off_rate_pdf_tlist,-b-1)
    pow_fit_off_rate_cdf = a*(-np.power(off_rate_cdf_tlist, -b) + np.power(off_rate_cdf_tlist[0],-b))
    pow_fit_diss_rate = max_diss_rate - (max_diss_rate-min_diss_rate)*a*(np.power(off_rate_cdf_tlist[0],-b)-np.power(off_rate_cdf_tlist,-b))

    axarr[0][0].fill_between(agg_plot_data_dict['t/T'][start_index:],
                            np.maximum(diss_rate_mean[start_index:] - diss_rate_std[start_index:],zeros_ref),
                             diss_rate_mean[start_index:]+ diss_rate_std[start_index:],
                             color=params_dict['COLORS'][0], alpha = 0.5)
    axarr[0][0].plot(agg_plot_data_dict['t/T'][start_index:], diss_rate_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][0].plot(off_rate_cdf_tlist, exp_fit_diss_rate, linestyle=':', lw=3,c='firebrick',
                     marker='None')
    axarr[0][0].plot(off_rate_cdf_tlist, pow_fit_diss_rate, linestyle=':', lw=3,c='darkgreen',
                     marker='None')
    axarr[0][0].text(0.55, 0.6, r"$f(t)=\frac{\dot{Q}^\mathrm{max} - \dot{Q}(t)}{\dot{Q}^\mathrm{max} - \dot{Q}^\mathrm{min}}$", withdash=True,horizontalalignment='center', verticalalignment='center',
                  transform=axarr[0][0].transAxes)
    axarr[0][0].set(ylabel=r"$\dot{Q}(t)$", xlabel=r"$t$")
    axarr[0][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][0].margins(0.05)

    axarr[0][1].plot(off_rate_cdf_tlist, off_rate_cdf, linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][1].plot(off_rate_cdf_tlist, exp_fit_off_rate_cdf, linestyle=':', lw=3,c='firebrick',
                     marker='None')
    axarr[0][1].set(ylabel=r"$f(t)$", xlabel=r"$t$")
    axarr[0][1].text(0.55, 0.6, r"$a(1-e^{-b(t-t_0)})$, a=%.3f, b=%.3f, $t_0$=%d"%(exp_a,exp_b,off_rate_cdf_tlist[0]), withdash=True,horizontalalignment='center', verticalalignment='center',
                     color='firebrick', transform=axarr[0][1].transAxes)
    axarr[0][1].plot(off_rate_cdf_tlist, pow_fit_off_rate_cdf, linestyle=':', lw=3,c='darkgreen',
                     marker='None')
    axarr[0][1].text(0.55, 0.4, r"$a(t^{-b}-t_0^{-b})$, a=%.3f, b=%.3f"%(a,b), withdash=True,horizontalalignment='center', verticalalignment='center',
                     color='darkgreen', transform=axarr[0][1].transAxes)
    axarr[0][1].margins(0.05)

    axarr[0][2].plot(off_rate_pdf_tlist, np.maximum(off_rate_pdf, 1e-4*ones_ref), linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][2].set(ylabel=r"$f'(t)$", xlabel=r"$t$")
    axarr[0][2].text(0.5, 0.6, r"$ab e^{-b(t-t_0)}$, a=%.3f, b=%.3f, $t_0$=%d"%(exp_a,exp_b,off_rate_pdf_tlist[0]), withdash=True,horizontalalignment='center', verticalalignment='center',
                     color='firebrick', transform=axarr[0][2].transAxes)
    axarr[0][2].plot(off_rate_pdf_tlist, np.maximum(exp_fit_off_rate_pdf,1e-4*ones_ref), linestyle=':', lw=3,c='firebrick',
                     marker='None')
    axarr[0][2].text(0.5, 0.8, r"$abt^{-b-1}$, a=%.3f, b=%.3f"%(a,b), withdash=True,horizontalalignment='center', verticalalignment='center',
                     color='darkgreen', transform=axarr[0][2].transAxes)
    axarr[0][2].plot(off_rate_pdf_tlist, np.maximum(pow_fit_off_rate_pdf,1e-4*ones_ref), linestyle=':', lw=3,c='darkgreen',
                     marker='None')
    axarr[0][2].set_yscale('log')
    axarr[0][2].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_cdf_pdf_trapping_rate_fit_cdf.png'
    fig.savefig(ofn)
    plt.close()

##############################################################################################################################################

def get_typicality_future_max_wdot(list_of_data_dirs, suffix_string):

    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    fig = plt.figure(figsize=(11, 10), dpi=200)
    gs = gridspec.GridSpec(1, 1)
    axarr = [[plt.subplot(gs[0, 0])]]

    list_max_wdot = []
    labels = ['drives permuted', 'drive unchanged']
    for list_idx, data_dirs in enumerate(list_of_data_dirs):
        agg_plot_data_dict = dict()
        base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
        keys_list = [ '<F(t).v(t)/(Barrier/T)>_T' ]
        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)
        cm = plt.cm.get_cmap('Blues')
        params_dict['COLORS'][1] = cm(0.6)
        cm = plt.cm.get_cmap('Greens')
        params_dict['COLORS'][0] = 'darkgreen'

        # params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']))
        dfn = os.path.join(init_data_dir, str('t/T').replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            agg_plot_data_dict['t/T'] = np.array(json.load(f))

        drive_switch_t = (params_dict['freq_switch_times'][0]+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD']
        drive_switch_idx = find_nearest_idx(agg_plot_data_dict['t/T'], drive_switch_t)

        for k in keys_list:
            dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k] = np.empty(num_dirs)
                agg_plot_data_dict[k][0] = np.amax(k_val[drive_switch_idx:])

        print 'suffix string is %s'%(suffix_string)
        for i, data_dir in enumerate(data_dirs[1:]):
            for k in keys_list:
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    k_val = np.array(json.load(f))
                    agg_plot_data_dict[k][i+1] = np.amax(k_val[drive_switch_idx:])

        list_max_wdot.append(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'])

    full_max_wdot_vals = np.array(list_max_wdot).flatten()
    full_max_wdot_hist, full_max_wdot_bins = np.histogram(full_max_wdot_vals, bins='fd', density=False)
    for j in range(len(list_of_data_dirs)):
        max_wdot_hist, max_wdot_bins = np.histogram(list_max_wdot[j], bins=full_max_wdot_bins, density=False)
        print 'sum of hist bars is %d, label is %s'%(np.sum(max_wdot_hist), labels[j])
        max_wdot_hist = 1.*max_wdot_hist/(1.*len(list_of_data_dirs[j]))
        axarr[0][0].bar(full_max_wdot_bins[:-1], max_wdot_hist, color=params_dict['COLORS'][j], edgecolor='k',
                        linewidth=0.75, width=full_max_wdot_bins[1] - full_max_wdot_bins[0], alpha=0.5, label=labels[j])

    axarr[0][0].set(ylabel=r"fraction of runs", xlabel=r"max future $\langle\dot{W}\rangle_\tau$")
    axarr[0][0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axarr[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[0][0].legend()
    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn + suffix_string + '_typical_max_future_Wdot.png'
    fig.savefig(ofn)
    plt.close()


def read_sim_data_plot_mode_pr_color_bar(data_dirs, suffix_string):
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    keys_list = ['eigvals_drive_start', 'forcing_overlap_drive_start',
                 'participation_ratios_start', 'participation_ratios_end',
                 'eigvals_drive_end', 'forcing_overlap_drive_end', ]

    num_dirs = len(data_dirs)
    init_data_dir = data_dirs[0]

    cm = plt.cm.get_cmap('Blues')

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['COLORS'] = [cm(0.6), cm(0.95)]
    # params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']))
    for k in keys_list:
        dfn = os.path.join(init_data_dir, str(k).replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
            agg_plot_data_dict[k][0] = k_val

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_list:
            # print 'k is %s, data_dir is %s, previous data_dir was %s'%(k, data_dir, data_dirs[i])
            dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k][i+1] = np.array(json.load(f))

    eig_start_min = np.amin(agg_plot_data_dict['eigvals_drive_start'])
    eig_end_min = np.amin(agg_plot_data_dict['eigvals_drive_end'])
    eig_start_max = np.amax(agg_plot_data_dict['eigvals_drive_start'])
    eig_end_max = np.amax(agg_plot_data_dict['eigvals_drive_end'])

    numbins = 12
    eig_bins_start = np.linspace(eig_start_min - 1e-6, eig_start_max + 1e-6, num=numbins)
    eig_bins_end = np.linspace(eig_end_min - 1e-6, eig_end_max + 1e-6, num=numbins)

    eig_hist_start, eig_bins_start = np.histogram(agg_plot_data_dict['eigvals_drive_start'], bins=eig_bins_start,
                                                  density=True)
    eig_hist_end, eig_bins_end = np.histogram(agg_plot_data_dict['eigvals_drive_end'], bins=eig_bins_end, density=True)

    forcing_overlap_start = np.empty([numbins - 1, num_dirs])
    forcing_overlap_end = np.empty([numbins - 1, num_dirs])

    pr_start = np.zeros([numbins - 1, num_dirs])
    pr_end = np.zeros([numbins - 1, num_dirs])
    num_modes_bin_wise_start = np.zeros(numbins-1)
    num_modes_bin_wise_end = np.zeros(numbins - 1)

    for i in range(num_dirs):
        eig_n_start, bins = np.histogram(agg_plot_data_dict['eigvals_drive_start'][i], bins=eig_bins_start)
        eig_n_end, bins = np.histogram(agg_plot_data_dict['eigvals_drive_end'][i], bins=eig_bins_end)
        start_index = 0
        end_index = 0

        for j in range(numbins - 1):
            forcing_overlap_start[j, i] = np.sum(
                agg_plot_data_dict['forcing_overlap_drive_start'][i, start_index:start_index + eig_n_start[j]])
            pr_start[j, i] += np.sum(
                agg_plot_data_dict['participation_ratios_start'][i, start_index:start_index + eig_n_start[j]])
            num_modes_bin_wise_start[j] += eig_n_start[j]
            start_index += eig_n_start[j]

            forcing_overlap_end[j, i] = np.sum(
                agg_plot_data_dict['forcing_overlap_drive_end'][i, end_index:end_index + eig_n_end[j]])
            pr_end[j, i] += np.sum(agg_plot_data_dict['participation_ratios_end'][i, end_index:end_index + eig_n_end[j]])
            num_modes_bin_wise_end[j] += eig_n_end[j]
            end_index += eig_n_end[j]

    forcing_overlap_start_mean = np.mean(forcing_overlap_start, axis=1)
    forcing_overlap_end_mean = np.mean(forcing_overlap_end, axis=1)

    pr_start_mean = np.sum(pr_start, axis=1)
    pr_start_mean = np.array([pr_start_mean[i]/ num_modes_bin_wise_start[i] if num_modes_bin_wise_start[i] !=0 else pr_start_mean[i] for i in range(numbins-1)])
    pr_end_mean = np.sum(pr_end, axis=1)
    pr_end_mean = np.array([pr_end_mean[i]/ num_modes_bin_wise_end[i] if num_modes_bin_wise_end[i] !=0 else pr_end_mean[i] for i in range(numbins-1)])

    cm = plt.cm.get_cmap('Blues')
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    fig = plt.figure(figsize=(20, 22), dpi=200)
    gs = gridspec.GridSpec(2, 2)
    axarr = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])],
             [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]]

    params_dict['COLORS'][0] = cm(0.4)
    axarr[0][0].bar(eig_bins_start[:-1], eig_hist_start, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1] - eig_bins_start[0])
    for drive_T in params_dict['T_list']:
        axarr[0][0].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[0][0].set(ylabel=r"$\langle$ # of normal modes$\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[0][0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axarr[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axarr[1][0].bar(eig_bins_end[:-1], eig_hist_end, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1] - eig_bins_end[0])
    for drive_T in params_dict['T_list']:
        axarr[1][0].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[1][0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axarr[1][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[1][0].set(ylabel=r"$\langle$ # of normal modes$\rangle$", xlabel=r"normal mode frequency $\omega$")
    xmin0, xmax0 = axarr[0][0].get_xlim()
    xmin1, xmax1 = axarr[1][0].get_xlim()
    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)

    ymin0, ymax0 = axarr[0][0].get_ylim()
    ymin1, ymax1 = axarr[1][0].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)
    axarr[0][0].set_ylim([ymin, ymax])
    axarr[1][0].set_ylim([ymin, ymax])
    axarr[0][0].set_xlim([xmin, xmax])
    axarr[1][0].set_xlim([xmin, xmax])
    axarr[0][0].margins(0.05)
    axarr[1][0].margins(0.05)

    cmap = plt.cm.get_cmap('Blues')
    print 'pr_start_max: %.2f, pr_start_min: %.2f, pr_end_max: %.2f, pr_end_min: %.2f' %(pr_start_mean.max(),
                                                                                         pr_start_mean.min(),
                                                                                         pr_end_mean.max(),
                                                                                         pr_end_mean.min() )
    # pr_max = max( pr_start_mean.max(), pr_end_mean.max())
    pr_max = 0.35
    c_start = [cmap(n.astype(float)/pr_max) for n in pr_start_mean]
    c_end = [cmap(n.astype(float)/pr_max) for n in pr_end_mean]

    cb_ax, kw = matplotlib.colorbar.make_axes_gridspec(axarr[0][1], orientation='horizontal', pad=0.1, fraction=0.1,
                                                       shrink=1.0, aspect=20)
    norm = matplotlib.colors.Normalize(vmin=0., vmax=pr_max+0.001)
    axarr[0][1].bar(eig_bins_start[:-1], forcing_overlap_start_mean, color=c_start, edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1] - eig_bins_start[0])
    cb = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_ticks([round(i * (pr_max )/5, 2) for i in range(6)])
    for drive_T in params_dict['T_list']:
        axarr[0][1].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[0][1].set(
        ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F}_\mathrm{drive}\rangle\vert^2 \right\rangle$",
        xlabel=r"normal mode frequency $\omega$")
    axarr[0][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[0][1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))


    cb_ax, kw = matplotlib.colorbar.make_axes_gridspec(axarr[1][1], orientation='horizontal', pad=0.1, fraction=0.1,
                                                       shrink=1.0, aspect=20)
    norm = matplotlib.colors.Normalize(vmin=0., vmax=pr_max+0.001)
    axarr[1][1].bar(eig_bins_end[:-1], forcing_overlap_end_mean, color=c_end, edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1] - eig_bins_end[0])
    cb = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_ticks([round(i * (pr_max )/5, 2) for i in range(6)])
    for drive_T in params_dict['T_list']:
        axarr[1][1].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[1][1].set(
        ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F}_\mathrm{drive}\rangle\vert^2\right\rangle$",
        xlabel=r"normal mode frequency $\omega$")
    axarr[1][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[1][1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    xmin0, xmax0 = axarr[0][1].get_xlim()
    xmin1, xmax1 = axarr[1][1].get_xlim()
    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)

    ymin0, ymax0 = axarr[0][1].get_ylim()
    ymin1, ymax1 = axarr[1][1].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)

    axarr[0][1].set_ylim([ymin, ymax])
    axarr[1][1].set_ylim([ymin, ymax])
    axarr[0][1].set_xlim([xmin, xmax])
    axarr[1][1].set_xlim([xmin, xmax])
    axarr[0][1].margins(0.05)
    axarr[1][1].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn + suffix_string + '_mode_data_initial_final_color_new.png'
    fig.savefig(ofn)
    plt.close()


def read_sim_data_aggregate_plot_allostery_force_transmission(data_dirs, suffix_string):
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    keys_list = ['allostery_measure_drive_start', 'force_transmission_measure_drive_start',
                 'allostery_measure_drive_end', 'force_transmission_measure_drive_end']

    num_dirs = len(data_dirs)
    init_data_dir = data_dirs[0]

    cm = plt.cm.get_cmap('Blues')

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['COLORS'] = [cm(0.6), cm(0.95)]

    for k in keys_list:
        dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
            agg_plot_data_dict[k][0] = k_val

    omega_list = np.array(params_dict['omega_plot'])

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_list:
            # print 'k is %s, data_dir is %s, previous data_dir was %s'%(k, data_dir, data_dirs[i])
            dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k][i+1] = np.array(json.load(f))

    allostery_drive_start_mean = np.mean(agg_plot_data_dict['allostery_measure_drive_start'], axis=0)
    # allostery_drive_start_std = np.std(agg_plot_data_dict['allostery_measure_drive_start'], axis=0)
    allostery_drive_end_mean = np.mean(agg_plot_data_dict['allostery_measure_drive_end'], axis=0)
    # allostery_drive_end_std = np.std(agg_plot_data_dict['allostery_measure_drive_end'], axis=0)

    force_transmission_drive_start_mean = np.mean(agg_plot_data_dict['force_transmission_measure_drive_start'], axis=0)
    # force_transmission_drive_start_std = np.std(agg_plot_data_dict['force_transmission_measure_drive_start'], axis=0)
    force_transmission_drive_end_mean = np.mean(agg_plot_data_dict['force_transmission_measure_drive_end'], axis=0)
    # force_transmission_drive_end_std = np.std(agg_plot_data_dict['force_transmission_measure_drive_end'], axis=0)

    fig = plt.figure(figsize=(20,10), dpi=200)
    gs = gridspec.GridSpec(1, 2)
    axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1])] ]

    zeros_ref = np.zeros(omega_list.size)+1E-6
    omega_drive = 2*np.pi/params_dict['T_list'][0]

    axarr[0][0].plot(omega_list, allostery_drive_start_mean,linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5,
                     label='at start')
    axarr[0][0].plot(omega_list, allostery_drive_end_mean,linestyle=':', c='firebrick',
                     marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=5,
                     label='at end')
    axarr[0][0].set(ylabel=r"$\frac{1}{N^2}\sum_{i\neq j}\vert \langle j \vert G(\omega)\vert i \rangle \vert^2$ at start",
                    xlabel=r"$\omega$")
    axarr[0][0].set_yscale('log')
    axarr[0][0].axvline(x=omega_drive, c=params_dict['COLORS'][1], linestyle='-.')
    axarr[0][0].legend()
    axarr[0][0].margins(0.05)


    axarr[0][1].plot(omega_list, force_transmission_drive_start_mean,linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5,
                     label='at start')
    axarr[0][1].plot(omega_list, force_transmission_drive_end_mean,linestyle=':', c='firebrick',
                     marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=5,
                     label='at end')
    axarr[0][1].set(ylabel=r"$\frac{1}{N}\sum_{j\neq i}\vert \langle j \vert G(\omega)\vert i_\mathrm{driven} \rangle \vert^2$ at start",
                    xlabel=r"$\omega$")
    axarr[0][1].set_yscale('log')
    axarr[0][1].axvline(x=omega_drive, c=params_dict['COLORS'][1], linestyle='-.')
    axarr[0][1].legend()
    axarr[0][1].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_allostery_force_transmission.png'
    fig.savefig(ofn)
    plt.close()


def read_sim_data_plot_aggregated(data_dirs, suffix_string):
    agg_plot_data_dict = dict()
    base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
    keys_list = ['eigvals_drive_start', 'forcing_overlap_drive_start', 'forcing_delta_2D_overlap_drive_start',
                 'forcing_delta_2N_overlap_drive_start',
                 'eigvals_drive_end', 'forcing_overlap_drive_end', 'forcing_delta_2D_overlap_drive_end',
                 'forcing_delta_2N_overlap_drive_end',
                 '<F(t).v(t)/(Barrier/T)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                  '<# spring transitions(t)>_T', '<U(t)/(Barrier)>_T', '<distance from relaxed state(t)>_T',
                 '<KE(t)/(Barrier)>_T', ]
    num_dirs = len(data_dirs)
    init_data_dir = data_dirs[0]

    cm = plt.cm.get_cmap('Blues')

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    params_dict['COLORS'] = [cm(0.6), cm(0.95)]
#
    # params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']))
    for k in keys_list:
        dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
            agg_plot_data_dict[k][0] = k_val

    keys_matrix_valued = ['forcing_overlap_tsteps'] #, 'spring_flips_tsteps_by_distance']
    for k in keys_matrix_valued:
        dfn = os.path.join(init_data_dir, str(k)+'.json')
        with open(dfn,'r') as f:
            k_val = np.array(json.load(f))
            agg_plot_data_dict[k] = np.empty([num_dirs, k_val.shape[0], k_val.shape[1]])
            agg_plot_data_dict[k][0] = k_val


    dfn = os.path.join(init_data_dir, str('t/T').replace('/','_by_')+'.json')
    with open(dfn,'r') as f:
        agg_plot_data_dict['t/T'] = np.array(json.load(f))

    print 'suffix string is %s'%(suffix_string)
    for i, data_dir in enumerate(data_dirs[1:]):
        for k in keys_list:
            # print 'k is %s, data_dir is %s, previous data_dir was %s'%(k, data_dir, data_dirs[i])
            dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                agg_plot_data_dict[k][i+1] = np.array(json.load(f))
        for k in keys_matrix_valued:
            dfn = os.path.join(data_dir, str(k) + '.json')
            with open(dfn, 'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k][i+1] = k_val

    forcing_overlap_tsteps_mean = np.mean(agg_plot_data_dict['forcing_overlap_tsteps'], axis=0)

    diss_rate_mean = np.mean(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)
    diss_rate_std = np.std(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)

    work_rate_mean = np.mean(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)
    work_rate_std = np.std(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)

    U_mean = np.mean(agg_plot_data_dict['<U(t)/(Barrier)>_T'], axis=0)
    U_std = np.std(agg_plot_data_dict['<U(t)/(Barrier)>_T'], axis=0)

    KE_mean = np.mean(agg_plot_data_dict['<KE(t)/(Barrier)>_T'], axis=0)
    KE_std = np.std(agg_plot_data_dict['<KE(t)/(Barrier)>_T'], axis=0)

    spring_hops_mean = np.mean(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)
    spring_hops_std = np.std(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)

    d_relaxed_mean = np.mean(agg_plot_data_dict['<distance from relaxed state(t)>_T'], axis=0)
    d_relaxed_std = np.std(agg_plot_data_dict['<distance from relaxed state(t)>_T'], axis=0)

    fig = plt.figure(figsize=(40,20), dpi=200)
    gs = gridspec.GridSpec(2, 4)
    axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3])],
              [plt.subplot(gs[1,0]), plt.subplot(gs[1,1]), plt.subplot(gs[1,2]), plt.subplot(gs[1,3])] ]
    start_index = params_dict['START_PLOT_INDEX']#+70 ################# adjusting for start_plot_index set to zero
    drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']
    zeros_ref = np.zeros(diss_rate_mean[start_index:].size)
############################# TEMPORARY, REMOVE 'sampled' IN LINES BELOW, ONCE RUN ############################
    axarr[0][0].fill_between(agg_plot_data_dict['t/T'][start_index:],
                            np.maximum(U_mean[start_index:] - U_std[start_index:], zeros_ref),
                             U_mean[start_index:] + U_std[start_index:],
                             color=params_dict['COLORS'][0], alpha = 0.2)
    axarr[0][0].plot(agg_plot_data_dict['t/T'][start_index:], U_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][0].set(ylabel=r"$\left\langle U(t)/\mathrm{Barrier} \right\rangle$",
                    xlabel=r"$t/T_\mathrm{drive}$")
################################################################################################################
    axarr[0][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][0].margins(0.05)

    axarr[1][0].fill_between(agg_plot_data_dict['t/T'][start_index:],
                            np.maximum(KE_mean[start_index:] - KE_std[start_index:], zeros_ref),
                             KE_mean[start_index:] + KE_std[start_index:],
                             # diss_rate_mean[start_index:] - diss_rate_std[start_index:], diss_rate_mean[start_index:] + diss_rate_std[start_index:],
                             color=params_dict['COLORS'][0], alpha = 0.5)
    axarr[1][0].plot(agg_plot_data_dict['t/T'][start_index:], KE_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
    # axarr[1][0].plot(agg_plot_data_dict['t/T'][start_index:], diss_rate_mean[start_index:], linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][0].set(ylabel=r"$\left\langle KE(t)/\mathrm{Barrier} \right\rangle$",
    # axarr[1][0].set(ylabel=r"$\left\langle \mathrm{Dissipation\; rate}(t)/(\mathrm{Barrier}/T)\right\rangle$",
                    xlabel=r"$t/T_\mathrm{drive}$")
    axarr[1][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][0].margins(0.05)

    axarr[0][1].fill_between(agg_plot_data_dict['t/T'][start_index:],
                         # hot_cold_diff_ke_mean[start_index:] - hot_cold_diff_ke_std[start_index:], hot_cold_diff_ke_mean[start_index:] + hot_cold_diff_ke_std[start_index:],
                            np.maximum(diss_rate_mean[start_index:] - diss_rate_std[start_index:],zeros_ref),
                             diss_rate_mean[start_index:]+ diss_rate_std[start_index:],
                             color=params_dict['COLORS'][0], alpha = 0.2)
    axarr[0][1].plot(agg_plot_data_dict['t/T'][start_index:], diss_rate_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
    # axarr[0][1].plot(agg_plot_data_dict['t/T'][start_index:], hot_cold_diff_ke_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][1].set(ylabel=r"$\left\langle \,\langle\mathrm{Dissipation\,rate}(t)/(\mathrm{Barrier}/T)\rangle_T \, \right\rangle$", xlabel=r"$t/T_\mathrm{drive}$")
    # axarr[0][1].set(ylabel = r"$\left\langle (\mathrm{hot\,node\,KE(t)}-\mathrm{cold\,node\,KE(t)})/\mathrm{Barrier}\right\rangle$", xlabel=r"$t/T_\mathrm{drive}$")
    axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[0][1].margins(0.05)

    axarr[1][1].fill_between(agg_plot_data_dict['t/T'][start_index:],
                             # diss_rate_lr_mean[start_index:] - diss_rate_lr_std[start_index:],
                             # diss_rate_lr_mean[start_index:] + diss_rate_lr_std[start_index:],
                             work_rate_mean[start_index:] - work_rate_std[start_index:],
                             work_rate_mean[start_index:] + work_rate_std[start_index:],
                             color=params_dict['COLORS'][0], alpha = 0.2)
    # axarr[1][1].plot(agg_plot_data_dict['t/T'][start_index:], diss_rate_lr_mean[start_index:],
    axarr[1][1].plot(agg_plot_data_dict['t/T'][start_index:], work_rate_mean[start_index:],
                     linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][1].set(ylabel=r"$\dot{W}/(E_b/\tau)$",
    # axarr[1][1].set(ylabel=r"$\left\langle \,\langle\mathrm{cold\,node\,KE}(t)/(\mathrm{Barrier}/T)\rangle_T \,\right\rangle$",
                    xlabel=r"$t/\tau$")
    axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')

    ymin0, ymax0 = axarr[0][1].get_ylim()
    ymin1, ymax1 = axarr[1][1].get_ylim()

    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)
    axarr[0][1].set_ylim([ymin, ymax])
    axarr[1][1].set_ylim([ymin, ymax])
    axarr[0][1].margins(0.05)
    axarr[1][1].margins(0.05)

    # sum_spring_transitions_last_2k = np.sum(agg_plot_data_dict['<# spring transitions(t)>_T'][:][-2000:], axis=1)
    # max_spring_transitions_last_2k = np.amax(sum_spring_transitions_last_2k)
    # c_spring_transitions = [cm(n/max_spring_transitions_last_2k) for n in sum_spring_transitions_last_2k]
    # for i in range(num_dirs):
    #     axarr[0][2].plot(agg_plot_data_dict['t/T'],
    #                      agg_plot_data_dict['<# spring transitions(t)>_T'][i][:],
    #                      linestyle=':', c=c_spring_transitions[i], marker='o',
    #                      markerfacecolor=c_spring_transitions[i],
    #                      markeredgecolor='None', alpha=0.5, markersize=3)
    axarr[0][2].fill_between(agg_plot_data_dict['t/T'][start_index:],
                            np.maximum(spring_hops_mean[start_index:] - spring_hops_std[start_index:], zeros_ref),
                            np.maximum(spring_hops_mean[start_index:]+ spring_hops_std[start_index:], zeros_ref),
                             color=params_dict['COLORS'][0], alpha = 0.2)
    axarr[0][2].plot(agg_plot_data_dict['t/T'][start_index:], np.maximum(spring_hops_mean[start_index:], zeros_ref),linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0][2].set(ylabel=r"$\langle\Sigma_T\mathrm{\,\#\, spring\, transitions(t)}\rangle$", xlabel=r"$t/T_\mathrm{drive}$")
    axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    # axarr[0][2].set_yscale('log')
    # axarr[0][2].set_yticks([0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10 ])
    # axarr[0][2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    axarr[0][2].margins(0.05)

    axarr[1][2].fill_between(agg_plot_data_dict['t/T'][start_index:],
                        d_relaxed_mean[start_index:] - d_relaxed_std[start_index:],
                        d_relaxed_mean[start_index:]+ d_relaxed_std[start_index:], color=params_dict['COLORS'][0], alpha = 0.5)
    axarr[1][2].plot(agg_plot_data_dict['t/T'][start_index:], d_relaxed_mean[start_index:],linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1][2].set(ylabel=r"$\langle\mathrm{distance\, from\, relaxed\, state(t)}\rangle$", xlabel=r"$t/T_\mathrm{drive}$")
    axarr[1][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[1][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
    axarr[1][2].margins(0.05)

    forcing_overlap_plot = axarr[0][3].imshow(forcing_overlap_tsteps_mean, aspect='auto',
                                           cmap=plt.get_cmap('PuBuGn'),
                                           extent=(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']), agg_plot_data_dict['t/T'][-1],
                                                   params_dict['omega'], 1e-6))
    divider = make_axes_locatable(axarr[0][3])
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(forcing_overlap_plot, orientation='vertical', format='%.2f',
                 label=r"$\vert\langle \hat{\omega}_i\vert \hat{f}\rangle\vert^2$", cax=cax)
    axarr[0][3].set(
        ylabel=r"normal mode frequency $\omega$")
    axarr[0][3].hlines(y=[2*np.pi/float(drive_T) for drive_T in params_dict['T_list']],
       xmin= [drive_start_t] + [(freq_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'] for freq_t in params_dict['freq_switch_times'][:-1]],
       xmax= [(freq_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'] for freq_t in params_dict['freq_switch_times']],
       color=params_dict['COLORS'][1], linewidth=6, linestyle=':')
    # for drive_T in params_dict['T_list']:
    #     axarr[0][3].axhline(y=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=2, linestyle='-.', alpha=0.6)
    for freq_switch_t in params_dict['freq_switch_times']:
        axarr[0][3].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')

    # forcing_perp_overlap_plot = axarr[1][3].imshow(forcing_perp_overlap_tsteps_mean, aspect='auto',
    #                                        cmap=plt.get_cmap('Blues'),
    #                                        extent=(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']), agg_plot_data_dict['t/T'][-1],
    #                                                params_dict['omega'], 1e-6))
    # divider = make_axes_locatable(axarr[1][3])
    # cax = divider.append_axes("right", "5%", pad="3%")
    # plt.colorbar(forcing_perp_overlap_plot, orientation='vertical',
    #              label=r"$\vert\langle \hat{\omega}_i\vert \hat{f}_\perp\rangle\vert^2$", cax=cax)
    # for drive_T in params_dict['T_list']:
    #     axarr[1][3].axhline(y=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=2, linestyle='-.', alpha=0.6)
    # axarr[1][3].set(
    #     ylabel=r"normal mode frequency $\omega$")
    # for freq_switch_t in params_dict['freq_switch_times']:
    #     axarr[1][3].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_KE_U_d_relaxed_overlap_avg.png'
    # ofn = base_ofn+suffix_string+'_hot_cold_diss_U_KE_d_relaxed_avg.png'
    fig.savefig(ofn)
    plt.close()

    eig_start_min = np.amin(agg_plot_data_dict['eigvals_drive_start'])
    eig_end_min = np.amin(agg_plot_data_dict['eigvals_drive_end'])
    eig_start_max = np.amax(agg_plot_data_dict['eigvals_drive_start'])
    eig_end_max = np.amax(agg_plot_data_dict['eigvals_drive_end'])

    numbins = 12
    eig_bins_start = np.linspace(eig_start_min-1e-6, eig_start_max+1e-6,num=numbins)
    eig_bins_end = np.linspace(eig_end_min-1e-6, eig_end_max+1e-6,num=numbins)

    eig_hist_start, eig_bins_start = np.histogram(agg_plot_data_dict['eigvals_drive_start'], bins=eig_bins_start, density=True)
    eig_hist_end, eig_bins_end = np.histogram(agg_plot_data_dict['eigvals_drive_end'], bins=eig_bins_end, density=True)

    forcing_overlap_start = np.empty([numbins-1,num_dirs])
    forcing_overlap_end = np.empty([numbins-1,num_dirs])
    forcing_delta_2D_overlap_start = np.empty([numbins-1,num_dirs])
    forcing_delta_2D_overlap_end = np.empty([numbins-1,num_dirs])

    forcing_delta_2N_overlap_start = np.empty([numbins-1,num_dirs])
    forcing_delta_2N_overlap_end = np.empty([numbins-1,num_dirs])

    # pr_start = np.empty([numbins-1,num_dirs])
    # pr_end = np.empty([numbins-1,num_dirs])

    for i in range(num_dirs):
        eig_n_start, bins = np.histogram(agg_plot_data_dict['eigvals_drive_start'][i], bins=eig_bins_start)
        eig_n_end, bins = np.histogram(agg_plot_data_dict['eigvals_drive_end'][i], bins=eig_bins_end)
        start_index = 0
        end_index = 0

        for j in range(numbins-1):
            forcing_overlap_start[j,i] = np.sum(agg_plot_data_dict['forcing_overlap_drive_start'][i,start_index:start_index+eig_n_start[j]])
            forcing_delta_2D_overlap_start[j,i] = np.sum(agg_plot_data_dict['forcing_delta_2D_overlap_drive_start'][i,start_index:start_index+eig_n_start[j]])
            forcing_delta_2N_overlap_start[j,i] = np.sum(agg_plot_data_dict['forcing_delta_2N_overlap_drive_start'][i,start_index:start_index+eig_n_start[j]])
            # KE_modes_start[j,i] = np.sum(agg_plot_data_dict['KE_mode_drive_start'][i,start_index:start_index+eig_n_start[j]])
            # pr_start[j,i] = np.mean(agg_plot_data_dict['participation_ratios_start'][i,start_index:start_index+eig_n_start[j]])
            start_index += eig_n_start[j]

            forcing_overlap_end[j,i] = np.sum(agg_plot_data_dict['forcing_overlap_drive_end'][i,end_index:end_index+eig_n_end[j]])
            forcing_delta_2D_overlap_end[j,i] = np.sum(agg_plot_data_dict['forcing_delta_2D_overlap_drive_end'][i,end_index:end_index+eig_n_end[j]])
            forcing_delta_2N_overlap_end[j, i] = np.sum(agg_plot_data_dict['forcing_delta_2N_overlap_drive_end'][i, end_index:end_index + eig_n_end[j]])
            # KE_modes_end[j,i] = np.sum(agg_plot_data_dict['KE_mode_drive_end'][i,end_index:end_index+eig_n_end[j]])
            # pr_end[j, i] = np.mean(agg_plot_data_dict['participation_ratios_end'][i, end_index:end_index + eig_n_end[j]])
            end_index += eig_n_end[j]

    forcing_overlap_start_mean = np.mean(forcing_overlap_start, axis=1)
    forcing_overlap_end_mean = np.mean(forcing_overlap_end, axis=1)

    forcing_delta_2D_overlap_start_mean = np.mean(forcing_delta_2D_overlap_start, axis=1)
    forcing_delta_2D_overlap_end_mean = np.mean(forcing_delta_2D_overlap_end, axis=1)

    forcing_delta_2N_overlap_start_mean = np.mean(forcing_delta_2N_overlap_start, axis=1)
    forcing_delta_2N_overlap_end_mean = np.mean(forcing_delta_2N_overlap_end, axis=1)

    cm = plt.cm.get_cmap('Blues')
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    fig = plt.figure(figsize=(40,20), dpi=200)
    gs = gridspec.GridSpec(2,4)
    axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]), plt.subplot(gs[0,3]) ],
                [plt.subplot(gs[1,0]), plt.subplot(gs[1,1]), plt.subplot(gs[1,2]), plt.subplot(gs[1,3])] ]

    eig_hist_max = max(eig_hist_start.max(), eig_hist_end.max())
    c_start = [cm(n.astype(float)/eig_hist_max) for n in eig_hist_start]
    c_end = [cm(n.astype(float)/eig_hist_max) for n in eig_hist_end]

    params_dict['COLORS'][0] = cm(0.4)
    # params_dict['COLORS'][0] = cmprime(0.6)
    axarr[0][0].bar(eig_bins_start[:-1], eig_hist_start, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1]-eig_bins_start[0]) #, yerr=eig_hist_start_err)
    # axarr[0][0].axvline(x=(2*np.pi/params_dict['FORCE_PERIOD']), c='gray', linestyle='-.')
    for drive_T in params_dict['T_list']:
        axarr[0][0].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    # axarr[0][0].text(0.3, 0.9, 'at beginning of drive',transform=axarr[0][0].transAxes)
    axarr[0][0].set(ylabel=r"$\langle$ # of normal modes$\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[0][0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axarr[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # axarr[1][0].bar(eig_bins_start[:-1], eig_hist_start, alpha = 0.5, lw= 3.,
    #                         color=c_start, width=eig_bins_start[1]-eig_bins_start[0])
    axarr[1][0].bar(eig_bins_end[:-1], eig_hist_end, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1]-eig_bins_end[0])
                    # yerr=eig_hist_end_err)
    # axarr[1][0].axvline(x=(2*np.pi/params_dict['FORCE_PERIOD']), c='gray', linestyle='-.')
    for drive_T in params_dict['T_list']:
        axarr[1][0].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[1][0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axarr[1][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axarr[1][0].text(0.3, 0.9, 'at end of drive', transform=axarr[1][0].transAxes)
    axarr[1][0].set(ylabel=r"$\langle$ # of normal modes$\rangle$", xlabel=r"normal mode frequency $\omega$")
    xmin0, xmax0 = axarr[0][0].get_xlim()
    xmin1, xmax1 = axarr[1][0].get_xlim()
    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)

    ymin0, ymax0 = axarr[0][0].get_ylim()
    ymin1, ymax1 = axarr[1][0].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)
    axarr[0][0].set_ylim([ymin, ymax])
    axarr[1][0].set_ylim([ymin, ymax])
    axarr[0][0].set_xlim([xmin, xmax])
    axarr[1][0].set_xlim([xmin, xmax])
    axarr[0][0].margins(0.05)
    axarr[1][0].margins(0.05)

    axarr[0][1].bar(eig_bins_start[:-1], forcing_overlap_start_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1]-eig_bins_start[0])
                    # yerr=forcing_overlap_start_std)
    # axarr[0][1].axvline(x=(2*np.pi/params_dict['FORCE_PERIOD']), c='gray', linestyle='-.')
    for drive_T in params_dict['T_list']:
        axarr[0][1].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    # axarr[0][1].text(0.35, 0.9, 'at beginning of drive',transform=axarr[0][1].transAxes)
    # ax1 = axarr[0][1].add_axes([0.05, 0.05, 0.9, 0.1])
    axarr[0][1].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F}_\mathrm{drive}\rangle\vert^2 \right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[0][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[0][1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    axarr[1][1].bar(eig_bins_end[:-1], forcing_overlap_end_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1]-eig_bins_end[0])
    # yerr=forcing_overlap_end_std)
    # ax1 = axarr[1][1].add_axes([0.05, 0.05, 0.9, 0.1])
    # axarr[1][1].axvline(x=(2*np.pi/params_dict['FORCE_PERIOD']), c='gray', linestyle='-.')
    for drive_T in params_dict['T_list']:
        axarr[1][1].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    # axarr[1][1].text(0.5, 0.9, 'at end of drive', transform=axarr[1][1].transAxes)
    axarr[1][1].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F}_\mathrm{drive}\rangle\vert^2\right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[1][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[1][1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    xmin0, xmax0 = axarr[0][1].get_xlim()
    xmin1, xmax1 = axarr[1][1].get_xlim()
    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)

    ymin0, ymax0 = axarr[0][1].get_ylim()
    ymin1, ymax1 = axarr[1][1].get_ylim()
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)

    axarr[0][2].bar(eig_bins_start[:-1], forcing_delta_2D_overlap_start_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1]-eig_bins_start[0])
    for drive_T in params_dict['T_list']:
        axarr[0][2].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[0][2].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F_\mathrm{drive}+\delta_\mathrm{2D}}\rangle\vert^2\right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[0][2].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    axarr[1][2].bar(eig_bins_end[:-1], forcing_delta_2D_overlap_end_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1]-eig_bins_end[0])
    for drive_T in params_dict['T_list']:
        axarr[1][2].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], linestyle=':', lw=8)
    axarr[1][2].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F_\mathrm{drive}+\delta_\mathrm{2D}}\rangle\vert^2\right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[1][2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[1][2].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    xmin0, xmax0 = axarr[0][2].get_xlim()
    xmin1, xmax1 = axarr[1][2].get_xlim()
    xmin = min(xmin0, xmin1, xmin)
    xmax = max(xmax0, xmax1, xmax)

    ymin0, ymax0 = axarr[0][2].get_ylim()
    ymin1, ymax1 = axarr[1][2].get_ylim()
    ymin = min(ymin0, ymin1, ymin)
    ymax = max(ymax0, ymax1, ymax)

    axarr[0][3].bar(eig_bins_start[:-1], forcing_delta_2N_overlap_start_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_start[1]-eig_bins_start[0])
    for drive_T in params_dict['T_list']:
        axarr[0][3].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], lw=8, linestyle=':')
    axarr[0][3].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F_\mathrm{drive}+\delta_\mathrm{2N}}\rangle\vert^2\right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[0][3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[0][3].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    axarr[1][3].bar(eig_bins_end[:-1], forcing_delta_2N_overlap_end_mean, color=params_dict['COLORS'][0], edgecolor='k', linewidth=0.5,
                    width=eig_bins_end[1]-eig_bins_end[0])
    for drive_T in params_dict['T_list']:
        axarr[1][3].axvline(x=2 * np.pi / float(drive_T), c=params_dict['COLORS'][1], linestyle=':', lw=8)
    axarr[1][3].set(ylabel=r"$\left\langle \vert\langle \hat{\omega}_i\vert \hat{F_\mathrm{drive}+\delta_\mathrm{2N}}\rangle\vert^2\right\rangle$", xlabel=r"normal mode frequency $\omega$")
    axarr[1][3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axarr[1][3].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    xmin0, xmax0 = axarr[0][3].get_xlim()
    xmin1, xmax1 = axarr[1][3].get_xlim()
    xmin = min(xmin0, xmin1, xmin)
    xmax = max(xmax0, xmax1, xmax)

    ymin0, ymax0 = axarr[0][3].get_ylim()
    ymin1, ymax1 = axarr[1][3].get_ylim()
    ymin = min(ymin0, ymin1, ymin)
    ymax = max(ymax0, ymax1, ymax)

    axarr[0][1].set_ylim([ymin, ymax])
    axarr[1][1].set_ylim([ymin, ymax])
    axarr[0][1].set_xlim([xmin, xmax])
    axarr[1][1].set_xlim([xmin, xmax])
    axarr[0][1].margins(0.05)
    axarr[1][1].margins(0.05)

    axarr[0][2].set_ylim([ymin, ymax])
    axarr[1][2].set_ylim([ymin, ymax])
    axarr[0][2].set_xlim([xmin, xmax])
    axarr[1][2].set_xlim([xmin, xmax])
    axarr[0][2].margins(0.05)
    axarr[1][2].margins(0.05)

    axarr[0][3].set_ylim([ymin, ymax])
    axarr[1][3].set_ylim([ymin, ymax])
    axarr[0][3].set_xlim([xmin, xmax])
    axarr[1][3].set_xlim([xmin, xmax])
    axarr[0][3].margins(0.05)
    axarr[1][3].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_mode_data_initial_final_new.png'
    fig.savefig(ofn)
    plt.close()


def plot_rep_distance_amp_variation(list_of_data_dirs, amplist, plot_dir, suffix_string):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    base_ofn = os.path.join(plot_dir, 'agg_rep_distance_amp_vary_plot_')

    # Amin = min(amplist)
    # Amax = max(amplist)
    # contrastA = [0.3, 0.7, 0.9]#[0.4 + i*(0.5/(len(amplist)-1)) for i in range(len(amplist))]

    cm = plt.cm.get_cmap('PuBuGn') #plt.cm.get_cmap('Blues')
    Amin = min(amplist)
    Amax = max(amplist)
    # contrastA = [0.45 + i*(0.35/(len(amplist)-1)) for i in range(len(amplist))]
    contrastA = [ 0.9, 0.4, 0.7]
    clist = [cm(Ai) for Ai in contrastA]
    # clist.reverse()

    for n, data_dirs in enumerate(list_of_data_dirs):
        distance_trajectories_i_j_list = []
        ci = clist[n]

        init_dir = data_dirs[0]

        dfn = os.path.join(init_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            init_params_dict = json.load(f)
        dfn = os.path.join(init_dir, 'sampled_tlist.json')
        with open(dfn, 'r') as f:
            init_tlist = np.array(json.load(f))

        edges = init_params_dict['EDGES']
        num_trajectory_samples = (np.array(init_params_dict['PLOT_STEPS']).size -
                                  np.int(init_params_dict['RELAX_STEPS'] / (2 * init_params_dict['PLOT_DELTA_STEPS'])))
        plot_tlist = init_tlist[-num_trajectory_samples:]

        seed1_list = [dirname[dirname.find('seed1_') + 6:dirname.find('_F_max_deg')] for dirname in data_dirs]
        unique_seeds = list(set(seed1_list))

        for k, unique_seed in enumerate(unique_seeds):
            same_seed_dirs = filter(lambda data_dir: str('seed1_' + unique_seed) in data_dir, data_dirs)

            spring_bit_trajectories = np.empty([len(same_seed_dirs), num_trajectory_samples, edges], dtype=np.int64)
            for i, same_seed_dir in enumerate(same_seed_dirs):
                spring_bit_trajectory = np.empty([init_tlist.size, edges], dtype=np.int64)
                get_spring_bits(same_seed_dir, spring_bit_trajectory)
                spring_bit_trajectories[i] = spring_bit_trajectory[-num_trajectory_samples:]

            for i in range(len(same_seed_dirs)):
                for j in range(i+1,len(same_seed_dirs)):
                    bit_trajectory_distance = np.empty(num_trajectory_samples)
                    get_distance_between_trajectories_tsteps(spring_bit_trajectories[i], spring_bit_trajectories[j],
                                                                                        bit_trajectory_distance)
                    distance_trajectories_i_j_list.append(bit_trajectory_distance)

        distance_trajectories_i_j_arr = np.array(distance_trajectories_i_j_list)

        dist_trajectories_mean = np.mean(distance_trajectories_i_j_arr, axis=0)
        dist_trajectories_std = np.std(distance_trajectories_i_j_arr, axis=0)

        # zeros_ref = np.zeros(dist_trajectories_mean.size)
        # ax.fill_between(plot_tlist,
        #     dist_trajectories_mean-dist_trajectories_std, dist_trajectories_mean+dist_trajectories_std,
        #                          color=ci, alpha=0.15)
        ax.plot(plot_tlist, dist_trajectories_mean,linestyle=':', c=ci,
                 marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[n]))
        ax.set(ylabel=r"$\left\langle \delta d_{ij}(t)\right\rangle$",
                        xlabel=r"$t/T_\mathrm{drive}$")

    # ax.legend(loc='best')
    ax.margins(0.05)
    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_dij_same_init_conditions.png'
    fig.savefig(ofn)
    plt.close()


def plot_perturb_amp_ramp():
    fig = plt.figure(figsize=(10, 12), dpi=200)
    gs = gridspec.GridSpec(3, 1)
    axarr = [ plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0]) ]
    base_ofn = os.path.join('/Volumes/HK_hard_drive/Dropbox (MIT)/jeremy_updates/dynamical_fine_tuning_figs/',
                            'perturb_ramp_amp_plot')
    amp_slope = 15./3000
    tlist = np.linspace(0., 3000., 3001)
    amplist = amp_slope*tlist
    Tlist = np.linspace(0.001, 0.75, 3001)

    cm = plt.cm.get_cmap('Blues')
    c_unperturb = cm(0.6)

    axarr[0].plot(tlist, amplist, linestyle=':', c=c_unperturb,
             marker='o', markerfacecolor=c_unperturb, markeredgecolor='None', markersize=5, label='unperturbed')
    # axarr[0].axhline(y=5.5, c=c_unperturb, linestyle='-.', lw=4)
    axarr[0].set(ylabel=r"A(t)", xlabel=r"$t/\tau$")
    # axarr[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axarr[0].xaxis.major.formatter._useMathText = True
    axarr[0].margins(0.05)

    axarr[1].plot(tlist, amplist, linestyle=':', c='darkgreen',
             marker='o', markerfacecolor='darkgreen', markeredgecolor='None', markersize=5, label='perturbed')
    # axarr[1].axhline(y=5.5, c=c_unperturb, linestyle='-.', lw=4)
    axarr[1].set(ylabel=r"A(t)", xlabel=r"$t/\tau$")
    # axarr[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axarr[1].xaxis.major.formatter._useMathText = True
    axarr[1].margins(0.05)

    axarr[2].plot(tlist, Tlist, linestyle=':', c='firebrick',
             marker='o', markerfacecolor='firebrick', markeredgecolor='None', markersize=5, label='thermal')
    axarr[2].set(ylabel=r"$\beta^{-1}$(t)", xlabel=r"$t/\tau$")
    # axarr[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axarr[2].xaxis.major.formatter._useMathText = True
    axarr[2].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn + '.png'
    fig.savefig(ofn)
    plt.close()


def read_sim_data_plot_amp_variation_fig(list_of_data_dirs, amplist, plot_dir, suffix_string):
    fig = plt.figure(figsize=(30, 20), dpi=200)
    gs = gridspec.GridSpec(2, 3)
    axarr = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])],
             [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]]

    base_ofn = os.path.join(plot_dir, 'agg_amp_plot_')

    cm = plt.cm.get_cmap('PuBuGn') #plt.cm.get_cmap('Blues')
    Amin = min(amplist)
    Amax = max(amplist)
    # contrastA = [0.45 + i*(0.35/(len(amplist)-1)) for i in range(len(amplist))]
    contrastA = [0.9, 0.7, 0.4]
    clist = [cm(Ai) for Ai in contrastA]
    # clist.reverse()
    for i, data_dirs in enumerate(list_of_data_dirs):
        ci = clist[i]
        agg_plot_data_dict = dict()
        keys_list = [ '<F(t).v(t)/(Barrier/T)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                      '<# spring transitions(t)>_T', '<distance from relaxed state(t)>_T',
                     '<dissipation_rate_lr(t)/(Barrier/T)>_T', '<U(t)/(Barrier)>_T']
        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)

        params_dict['START_PLOT_INDEX'] = int(params_dict['T_RELAX']/(2.*params_dict['FORCE_PERIOD']))
        for k in keys_list:
            dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
                agg_plot_data_dict[k][0] = k_val

        dfn = os.path.join(init_data_dir, str('t/T').replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            agg_plot_data_dict['t/T'] = np.array(json.load(f))

        for j, data_dir in enumerate(data_dirs[1:]):
            for k in keys_list:
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    agg_plot_data_dict[k][j+1] = np.array(json.load(f))

        diss_rate_mean = np.mean(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)
        diss_rate_std = np.std(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)

        work_rate_mean = np.mean(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)
        work_rate_std = np.std(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)

        diss_rate_lr_mean = np.mean(agg_plot_data_dict['<dissipation_rate_lr(t)/(Barrier/T)>_T'], axis=0)
        diss_rate_lr_std = np.std(agg_plot_data_dict['<dissipation_rate_lr(t)/(Barrier/T)>_T'], axis=0)

        U_mean = np.mean(agg_plot_data_dict['<U(t)/(Barrier)>_T'], axis=0)#+1e-2
        U_std = np.std(agg_plot_data_dict['<U(t)/(Barrier)>_T'], axis=0)

        spring_hops_mean = np.mean(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)# + 1e-2
        spring_hops_std = np.std(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)
        d_relaxed_mean = np.mean(agg_plot_data_dict['<distance from relaxed state(t)>_T'], axis=0)
        d_relaxed_std = np.std(agg_plot_data_dict['<distance from relaxed state(t)>_T'], axis=0)

        start_index = params_dict['START_PLOT_INDEX']
        drive_start_t = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']
        drive_start_index = find_nearest_idx(agg_plot_data_dict['t/T'], drive_start_t)

        drive_start_index = getLeftIndex(agg_plot_data_dict['t/T'], drive_start_t)
        zeros_ref = np.zeros(diss_rate_mean[start_index:].size)+0.1
        pone_ref = 0.1*np.ones(diss_rate_mean[drive_start_index:].size)
        axarr[0][0].fill_between(agg_plot_data_dict['t/T'][drive_start_index:-1],
                                np.maximum(diss_rate_mean[drive_start_index:-1] - diss_rate_std[drive_start_index:-1], pone_ref[:-1]),
                                 diss_rate_mean[drive_start_index:-1]+ diss_rate_std[drive_start_index:-1],
                                 color=ci, alpha = 0.1)
        axarr[0][0].plot(agg_plot_data_dict['t/T'][drive_start_index:-1], np.maximum(diss_rate_mean[drive_start_index:-1], pone_ref[:-1]),linestyle=':', c=ci,
                         marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[i]))
        axarr[0][0].set(ylabel=r"$\dot{Q}(t)/(E_b/\tau)$", xlabel=r"$t/\tau$")
        axarr[0][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[0][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][0].set_yscale('log')
        axarr[0][0].set_yticks([0.1, 0.2, 0.4, 1, 2, 4, 10 ])
        axarr[0][0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # axarr[0][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # lr_zeros_ref = np.zeros(diss_rate_lr_mean[drive_start_index:].size)
        # axarr[0][1].fill_between(agg_plot_data_dict['t/T'][drive_start_index:],
        #                          np.maximum(diss_rate_lr_mean[drive_start_index:] - diss_rate_lr_std[drive_start_index:], lr_zeros_ref),
        #                          diss_rate_lr_mean[drive_start_index:] + diss_rate_lr_std[drive_start_index:],
        #                          color=ci, alpha = 0.15)
        axarr[0][1].plot(agg_plot_data_dict['t/T'][drive_start_index:-1], diss_rate_lr_mean[drive_start_index:-1],
                         linestyle=':', c=ci, marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5,
                         label='A=%.1f'%(amplist[i]))
        axarr[0][1].set(ylabel=r"$\dot{Q}_{lr}(t)/(E_b/\tau)$", xlabel=r"$t/\tau$")
        axarr[0][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[0][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][1].set_yscale('log')
        axarr[0][1].set_yticks([0.2, 0.4, 1, 2, 4 ])
        axarr[0][1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # axarr[0][1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        axarr[1][0].fill_between(agg_plot_data_dict['t/T'][drive_start_index:-1],
                                np.maximum(spring_hops_mean[drive_start_index:-1] - spring_hops_std[drive_start_index:-1], 0.2*pone_ref[:-1]),
                                 np.maximum(spring_hops_mean[drive_start_index:-1]+ spring_hops_std[drive_start_index:-1], 0.2*pone_ref[:-1]),
                                 color=ci, alpha = 0.1)
        axarr[1][0].plot(agg_plot_data_dict['t/T'][drive_start_index:-1], np.maximum(spring_hops_mean[drive_start_index:-1], 0.2*pone_ref[:-1]),linestyle=':', c=ci,
                         marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[i]))
        axarr[1][0].set(ylabel=r"$\Sigma_\tau\mathrm{\#\,hops(t)}$", xlabel=r"$t/\tau$")
        axarr[1][0].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[1][0].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][0].set_yscale('log')
        axarr[1][0].set_yticks([0.02, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40 ])
        axarr[1][0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # axarr[1][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # axarr[1][1].fill_between(agg_plot_data_dict['t/T'][start_index:],
        #                     np.maximum(d_relaxed_mean[start_index:] - d_relaxed_std[start_index:], zeros_ref),
        #                     d_relaxed_mean[start_index:]+ d_relaxed_std[start_index:], color=ci, alpha = 0.15)
        axarr[1][1].plot(agg_plot_data_dict['t/T'][drive_start_index:], d_relaxed_mean[drive_start_index:],linestyle=':', c=ci,
                         marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[i]))
        axarr[1][1].set(ylabel=r"$\Delta s(t)$", xlabel=r"$t/\tau$")
        # axarr[1][1].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[1][1].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')

        axarr[0][2].fill_between(agg_plot_data_dict['t/T'][drive_start_index:-1],
                                np.maximum(work_rate_mean[drive_start_index:-1] - work_rate_std[drive_start_index:-1],pone_ref[:-1]),
                                work_rate_mean[drive_start_index:-1]+ work_rate_std[drive_start_index:-1],
                                 color=ci, alpha = 0.1)
        axarr[0][2].plot(agg_plot_data_dict['t/T'][drive_start_index:-1], work_rate_mean[drive_start_index:-1],linestyle=':', c=ci,
                         marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[i]))
        axarr[0][2].set(ylabel=r"$\dot{W}(t)/(E_b/\tau)$", xlabel=r"$t/\tau$")
        axarr[0][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')
        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[0][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[0][2].set_yscale('log')
        axarr[0][2].set_yticks([0.1, 0.2, 0.4, 1, 2, 4, 10 ])
        axarr[0][2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # axarr[0][2].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        axarr[1][2].fill_between(agg_plot_data_dict['t/T'][drive_start_index:],
                                np.maximum(U_mean[drive_start_index:] - U_std[drive_start_index:], 2*pone_ref),
                                 U_mean[drive_start_index:]+ U_std[drive_start_index:],
                                 color=ci, alpha = 0.1)
        axarr[1][2].plot(agg_plot_data_dict['t/T'][start_index:], U_mean[start_index:],linestyle=':', c=ci,
                         marker='o', markerfacecolor=ci, markeredgecolor='None', markersize=5, label='A=%.1f'%(amplist[i]))
        axarr[1][2].set(ylabel=r"$U(t)/E_b$", xlabel=r"$t/\tau$")
        axarr[1][2].axvline(x=drive_start_t, c=params_dict['COLORS'][1], linestyle='-.')

        # for freq_switch_t in params_dict['freq_switch_times']:
        #     axarr[1][2].axvline(x=(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'], c=params_dict['COLORS'][1], linestyle=':')
        axarr[1][2].set_yscale('log')

        # axarr[1][2].set_yticks([ 0.2, 0.4, 1, 2, 4, 10, 20 ])
        # axarr[1][2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # axarr[1][2].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # axarr[0][0].legend(loc='best')
    axarr[0][1].margins(0.05)
    axarr[0][1].legend(loc='best')

    # axarr[1][0].set_yscale('log')
    # axarr[1][0].set_yticks([0.01, 0.1, 0.5, 1, 2, 5, 10, 20])
    # axarr[1][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # axarr[1][0].margins(0.05)
    # axarr[1][0].legend(loc='best')
    # axarr[1][1].margins(0.05)
    axarr[1][1].legend(loc='best')
    axarr[0][2].margins(0.05)
    ymin, ymax = axarr[0][2].get_ylim()
    axarr[0][0].set_ylim([ymin, ymax])
    axarr[0][0].margins(0.05)
    # axarr[1][2].set_yscale('log')
    # axarr[1][2].set_yticks([0.01, 0.1, 0.5, 1, 2, 5, 10])
    # axarr[1][2].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axarr[1][2].margins(0.05)
    # axarr[0][2].legend(loc='best')
    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_Qdot_Wdot_hamming_dist_avg_w_std.png'
    fig.savefig(ofn)
    plt.close()


########## CONTINUE WORKING ON THIS FN ##############
def read_sim_data_plot_ss_wdot_vs_drive_period(data_dirs_list, suffix_string):
    drive_T_list = []
    data_dict = dict()

    for data_dirs in data_dirs_list:
        agg_plot_data_dict = dict()
        base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
        keys_list = ['<F(t).v(t)/(Barrier/T)>_T', '<dissipation_rate(t)/(Barrier/T)>_T',
                      '<# spring transitions(t)>_T', ]

        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        cm = plt.cm.get_cmap('Blues')

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)
        params_dict['COLORS'] = [cm(0.6), cm(0.95)]

        for k in keys_list:
            dfn = os.path.join(init_data_dir, str(k).replace('/','_by_')+'.json')
            with open(dfn,'r') as f:
                k_val = np.array(json.load(f))
                agg_plot_data_dict[k] = np.empty([num_dirs, k_val.size])
                agg_plot_data_dict[k][0] = k_val

        dfn = os.path.join(init_data_dir, str('t/T').replace('/','_by_')+'.json')
        with open(dfn,'r') as f:
            scaled_tlist = np.array(json.load(f))

        for i, data_dir in enumerate(data_dirs[1:]):
            for k in keys_list:
                # print 'k is %s, data_dir is %s, previous data_dir was %s'%(k, data_dir, data_dirs[i])
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    agg_plot_data_dict[k][i+1] = np.array(json.load(f))

        diss_rate_mean = np.mean(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)
        diss_rate_std = np.std(agg_plot_data_dict['<dissipation_rate(t)/(Barrier/T)>_T'], axis=0)

        work_rate_mean = np.mean(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)
        work_rate_std = np.std(agg_plot_data_dict['<F(t).v(t)/(Barrier/T)>_T'], axis=0)

        spring_hops_mean = np.mean(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)
        spring_hops_std = np.std(agg_plot_data_dict['<# spring transitions(t)>_T'], axis=0)

        T_list = np.array(params_dict['T_list'])

        for drive_T in T_list:
            print 'drive_T is %.2f'%(drive_T)
            print 'type of drive_T is %s'%(type(drive_T))
            drive_T_list.append(drive_T)

        freq_switch_times = [(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'] for freq_switch_t in
                             params_dict['freq_switch_times'] ]

        T_switch_idx_list = [ find_nearest_idx(scaled_tlist, freq_switch_t) for freq_switch_t in freq_switch_times ]

        print 'T_switch_idx_list:'
        print T_switch_idx_list
        print 'force period is %.1f'%(params_dict['FORCE_PERIOD'])

        for i in range(len(T_switch_idx_list)):
            data_dict[T_list[i]] = dict()
            data_dict[T_list[i]]['wdot_avg'] = (np.mean(
                    work_rate_mean[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])
            data_dict[T_list[i]]['wdot_std'] = (np.std(
                    work_rate_mean[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])

            data_dict[T_list[i]]['qdot_avg'] = (np.mean(
                        diss_rate_mean[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])
            data_dict[T_list[i]]['qdot_std'] = (np.mean(
                        diss_rate_std[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])

            data_dict[T_list[i]]['spring_hops_avg'] = (np.mean(
                        spring_hops_mean[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])
            data_dict[T_list[i]]['spring_hops_std'] = (np.mean(
                        spring_hops_std[T_switch_idx_list[i]-12:T_switch_idx_list[i]-2:])/params_dict['FORCE_PERIOD'])


    drive_T_arr = np.sort(np.array(drive_T_list))
    print drive_T_arr

    omega_arr = np.array([2*np.pi/np.float(T_drive) for T_drive in drive_T_arr])


    wdot_mean_arr = np.array([data_dict[drive_T]['wdot_avg'] for drive_T in drive_T_arr])
    wdot_std_arr = np.array([data_dict[drive_T]['wdot_std'] for drive_T in drive_T_arr])

    qdot_mean_arr = np.array([data_dict[drive_T]['qdot_avg'] for drive_T in drive_T_arr])
    qdot_std_arr = np.array([data_dict[drive_T]['qdot_std'] for drive_T in drive_T_arr])

    spring_hops_mean_arr = np.array([data_dict[drive_T]['spring_hops_avg'] for drive_T in drive_T_arr])
    spring_hops_std_arr = np.array([data_dict[drive_T]['spring_hops_std'] for drive_T in drive_T_arr])

    fig = plt.figure(figsize=(30,10), dpi=200)
    gs = gridspec.GridSpec(1, 3)
    axarr = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])]

    axarr[0].errorbar(omega_arr, wdot_mean_arr, yerr=wdot_std_arr, linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[0].set(ylabel=r"stable $\left\langle \dot{W}\right\rangle_\tau/(E_b)$",
                    xlabel=r"$\omega$, drive frequency")
    axarr[0].margins(0.05)

    axarr[1].errorbar(omega_arr, qdot_mean_arr, yerr=qdot_std_arr, linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[1].set(ylabel=r"stable $\left\langle \dot{Q} \right\rangle_\tau/(E_b)$",
                    xlabel=r"$\omega$, drive frequency")
    axarr[1].margins(0.05)

    axarr[2].errorbar(omega_arr, spring_hops_mean_arr, yerr=spring_hops_std_arr, linestyle=':', c=params_dict['COLORS'][0],
                     marker='o', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=5)
    axarr[2].set(ylabel=r"Spring Hops rate",
                    xlabel=r"$\omega$, drive frequency")
    axarr[2].margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_reverse_order.png'
    print ofn
    fig.savefig(ofn)
    plt.close()


def read_sim_data_print_percent_sims_w_hops(data_dirs, basestr):
    num_sims = len(data_dirs)
    num_sims_w_hops = 0.
    total_sims_hops = 0.
    init_data_dir = data_dirs[0]

    dfn = os.path.join(init_data_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)

    cm = plt.cm.get_cmap('Blues')
    params_dict['COLORS'] = [cm(0.6), cm(0.95)]

    dfn = os.path.join(init_data_dir, str('t/T').replace('/', '_by_') + '.json')
    with open(dfn, 'r') as f:
        scaled_tlist = np.array(json.load(f))

    T_list = np.array(params_dict['T_list'])

    relax_time = params_dict['T_RELAX'] / params_dict['FORCE_PERIOD']
    relax_idx = find_nearest_idx(scaled_tlist, relax_time)

    for data_dir in data_dirs:
        dfn = os.path.join(data_dir, '<# spring transitions(t)>_T.json')
        with open(dfn,'r') as f:
            spring_transitions_array = np.array(json.load(f))

        total_hops = np.sum(spring_transitions_array[relax_idx:])
        if  total_hops >= 1.:
            num_sims_w_hops += 1.
            total_sims_hops += total_hops

    print 'for %s, # sims is %d' %(basestr, num_sims)
    print "fraction of sims w hops is %.2f"%(num_sims_w_hops/num_sims)
    print "total hops is %d, # sims w hops is %d" %(total_sims_hops, num_sims_w_hops)


def read_sim_data_plot_Vsqmax_vs_drive_freq(data_dirs_list, suffix_string):
    drive_T_list = []
    data_dict = dict()

    for data_dirs in data_dirs_list:
        agg_plot_data_dict = dict()
        base_ofn = os.path.join(os.path.split(data_dirs[0])[0], 'agg_plot_')
        keys_list = ['<dissipation_rate(t)/(Barrier/T)>_T',
                      '<# spring transitions(t)>_T']

        num_dirs = len(data_dirs)
        init_data_dir = data_dirs[0]

        dfn = os.path.join(init_data_dir, 'params_dict.json')
        with open(dfn, 'r') as f:
            params_dict = json.load(f)

        cm = plt.cm.get_cmap('Blues')
        params_dict['COLORS'] = [cm(0.6), cm(0.95)]

        dfn = os.path.join(init_data_dir, str('t/T').replace('/', '_by_') + '.json')
        with open(dfn, 'r') as f:
            scaled_tlist = np.array(json.load(f))

        T_list = np.array(params_dict['T_list'])
        avg_Vsq_stable = dict()
        avg_Vsq_unstable = dict()
        for drive_T in T_list:
            print 'drive_T is %.2f'%(drive_T)
            print 'type of drive_T is %s'%(type(drive_T))
            drive_T_list.append(drive_T)
            data_dict[drive_T] = dict()
            avg_Vsq_stable[drive_T] = []
            avg_Vsq_unstable[drive_T] = []

        freq_switch_times = [(freq_switch_t+params_dict['T_RELAX'])/params_dict['FORCE_PERIOD'] for freq_switch_t in
                             params_dict['freq_switch_times'] ]

        T_switch_idx_list = [ find_nearest_idx(scaled_tlist, freq_switch_t) for freq_switch_t in freq_switch_times ]
        relax_time = params_dict['T_RELAX']/params_dict['FORCE_PERIOD']
        relax_idx = find_nearest_idx(scaled_tlist, relax_time)
        T_start_idx_list = [relax_idx ] + T_switch_idx_list[:-1]

        print 'T_switch_idx_list:'
        print T_switch_idx_list
        print 'force period is %.1f'%(params_dict['FORCE_PERIOD'])

        for data_dir in data_dirs:
            data_dir_dict = dict()
            for k in keys_list:
                dfn = os.path.join(data_dir, str(k).replace('/','_by_')+'.json')
                with open(dfn,'r') as f:
                    data_dir_dict[k] = np.array(json.load(f))

            for i in range(len(T_switch_idx_list)):
                if np.sum(data_dir_dict['<# spring transitions(t)>_T'][T_start_idx_list[i]:T_start_idx_list[i]+500])>0:
                    vsq_unstable = (np.amax(
                        data_dir_dict['<dissipation_rate(t)/(Barrier/T)>_T'][T_start_idx_list[i]:T_start_idx_list[i]+500])/
                                    (params_dict['FORCE_PERIOD']*params_dict['DAMPING_RATE']))
                    avg_Vsq_unstable[T_list[i]].append(vsq_unstable)

                if np.sum(data_dir_dict['<# spring transitions(t)>_T'][T_switch_idx_list[i]-22:T_switch_idx_list[i]-2])<1.:
                    vsq_stable = (np.amax(
                        data_dir_dict['<dissipation_rate(t)/(Barrier/T)>_T'][T_switch_idx_list[i]-22:T_switch_idx_list[i]-2])/
                                    (params_dict['FORCE_PERIOD']*params_dict['DAMPING_RATE']))
                    avg_Vsq_stable[T_list[i]].append(vsq_stable)


        for i in range(T_list.size):
            avg_Vsq_unstable[T_list[i]] = np.array(avg_Vsq_unstable[T_list[i]])
            avg_Vsq_stable[T_list[i]] = np.array(avg_Vsq_stable[T_list[i]])
            data_dict[T_list[i]]['Vsq_unstable'] = np.mean(avg_Vsq_unstable[T_list[i]])
            data_dict[T_list[i]]['Vsq_stable'] = np.mean(avg_Vsq_stable[T_list[i]])

    drive_T_arr = np.sort(np.array(drive_T_list))
    print drive_T_arr
    omega_arr = np.array([2*np.pi/np.float(T_drive) for T_drive in drive_T_arr])

    avg_Vsq_unstable = np.array([data_dict[T_i]['Vsq_unstable'] for T_i in drive_T_arr ])
    avg_Vsq_stable = np.array([data_dict[T_i]['Vsq_stable'] for T_i in drive_T_arr])

    fig = plt.figure(figsize=(10,10), dpi=200)
    # fig = plt.figure(figsize=(30,15), dpi=100)
    # gs = gridspec.GridSpec(2,3)
    gs = gridspec.GridSpec(1, 1)
    # axarr = [ [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])],
    #           [plt.subplot(gs[1,0]), plt.subplot(gs[1,1]), plt.subplot(gs[1,2])] ]
    ax = plt.subplot(gs[0,0])

    ax.plot(omega_arr, avg_Vsq_stable, linestyle=':', c=params_dict['COLORS'][0],
                    marker='v', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=7,
                    label='stable')
    ax.plot(omega_arr, avg_Vsq_unstable, linestyle=':', c=params_dict['COLORS'][0],
                    marker='^', markerfacecolor=params_dict['COLORS'][0], markeredgecolor='None', markersize=7,
                    label='unstable')
    ax.axhline(y=2./params_dict['MASS'], linestyle='-.', c='k')
    ax.set(ylabel=r"max $\langle \mathbf{V}^2\rangle$ ",
                    xlabel=r"$\omega$, drive frequency")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fancybox=True, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=16,
                    labelspacing=1., framealpha=0.5)
    ax.margins(0.05)

    gs.update(left=0.1, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    gs.tight_layout(fig)
    ofn = base_ofn+suffix_string+'_moon_criterion.png'
    print ofn
    fig.savefig(ofn)
    plt.close()


def rename_x_to_f_plot_dir(plot_dir):
    dfn = os.path.join(plot_dir, 'params_dict.json')
    with open(dfn, 'r') as f:
        params_dict = json.load(f)
    new_plot_dir = str(plot_dir).replace('_T2p1_', '_T_%.1f_'%(params_dict['FORCE_PERIOD']))
    new_plot_dir = new_plot_dir.replace('.', 'p')
    os.rename(plot_dir, new_plot_dir)

if __name__ == "__main__":
    pathname = '/Volumes/bistable_data/force_drive_sims'
    basestrlist = [ '_erg_n20_m50', '_F_max_deg_A2p00_T_2p1', '_Tend_10700']
    data_dirs = get_data_dirs(pathname, basestrlist)

    read_sim_data_print_percent_sims_w_hops(data_dirs, str('').join(basestrlist))

