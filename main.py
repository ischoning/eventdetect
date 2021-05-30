#import dispersion_setup
#import degrees_of_vision_spyder
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    # plot fixations and saccades
    df = pd.read_csv('fixations_final_data.csv')
    print(df.columns)
    pd.set_option('display.max_columns', None)
    print(df.head())


    t = df['timestamp_milis']
    fx = df['fixation_x']
    fy = df['fixation_y']
    f_start = df['fixation_start_ts']
    f_end = df['fixation_end_ts']
    s_start = df['saccade_start_ts']
    s_end = df['saccade_end_ts']
    s_amp = df['saccade_amplitude']
    s_len = df['saccade_duration']


    plt.scatter(fx, fy)
    plt.title('fx vs fy')
    plt.show()

    plt.scatter(t, fx)
    plt.title('fx')
    plt.show()

    plt.scatter(t, fy)
    plt.title('fy')
    plt.show()

    plt.scatter(t, s_amp)
    plt.title('saccade amplitude')
    plt.show()

<<<<<<< HEAD
def read_unfiltered():
    df = pd.read_csv('/Users/ischoning/PycharmProjects/GitHub/data/participant08_preprocessed172.csv')
    df = df[100:int(len(df) / 500)]

    # assign relevant data
    lx = df['left_forward_x']
    ly = df['left_forward_y']
    lz = df['left_forward_z']
    rx = df['right_forward_x']
    ry = df['right_forward_y']
    rz = df['right_forward_z']
    t = df['raw_timestamp']
    counter = df['Unnamed: 0']

    # compute angular values
    df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
    df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
    df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
    df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))

    # average visual angle between both eyes along each plane
    avg_ang_x = df['Avg_angular_x']
    avg_ang_y = df['Avg_angular_y']

    samples = []
    for i in range(len(df)):
        samples.append(Sample(counter, t, avg_ang_x, avg_ang_y))

    #lstream = ListSampleStream(samples)

    # find intersample velocities and append as df_col 'interp_samplevelocity_right'
    return df

def read_filtered():
    df = pd.read_csv('/Users/ischoning/PycharmProjects/GitHub/data/participant08_preprocessed172.csv')
    df = df[100:int(len(df) / 500)]
    print(df.dtypes)
    return df

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        print(t)
        print(previous)
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


def normal(mu, sigma, x):
    """Return P(x) for a normal distribution with given mu and sigma"""
    try:
        return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-math.pow(float(x) - mu, 2) / (2 * sigma * sigma))
    except: # if x == None
        return 0

# The probability of observing o in state s (fix=0,sac=1)
def emitP(s, o):
    fOPm = 0.001
    fOPv = 0.001
    sOPm = 0.005
    sOPv = 0.5

    if s == 'fix':
        p = normal(fOPm, fOPv, o)  # why do we use gaussian probability? not beta?
        if p == 0:
            p = 0.0001
        return math.log(p)  # = -4 if p = 0.0001 ??why do we work in logs?
    else:
        p = normal(sOPm, sOPv, o)
        if p == 0:
            p = 0.0001
        return math.log(p)  # = -4 if p = 0.0001


def main():
    # read unfiltered data
    #df = read_unfiltered()

    # read filtered data
    df = read_filtered()

    # TODO: Calculate starting probabilities, means, variances

    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Saccade', 'Fixation']
    df['fix1 sac0'] = np.where(v <= 0.02, 1, 0)
    event = df['fix1 sac0']

    # estimate priors (sample means)
    mean_fix = np.mean(df[event == 1]['ang_vel'])
    mean_sac = np.mean(df[event == 0]['ang_vel'])
    std_fix = np.std(df[event == 1]['ang_vel'])
    std_sac = np.std(df[event == 0]['ang_vel'])

    obs = df['interp_samplevelocity_right'][1:len(df)] # NoneType in row 0
    # obs = obs.dropna()
    obs = obs.reset_index(drop = True)
    # print(obs[0])
    states = ("fix", "sac")
    start_p = {"fix": math.log(0.55), "sac": math.log(0.45)}
    trans_p = {
        "fix": {"fix": 0.95, "sac": 0.05},
        "sac": {"fix": 0.05, "sac": 0.95},
    }

    emit_p = {
        "fix": None,
        "sac": None,
    }
    for s in emit_p:
        probs = {}
        for o in obs:
            probs[o] = emitP(s, o)
        emit_p[s] = probs

    #viterbi(obs, states, start_p, trans_p, emit_p)
    stream = ListSampleStream(testPath)
    h2 = HMM(stream, mean_fix, std_fix, mean_sac, std_sac, 0.95, 0.05, 0.95, 0.05)


=======
>>>>>>> parent of e9456f2 (hmm)

if __name__ == "__main__":
    # Testing
    # hello("Isabella")

    # data pipeline
    if not os.path.isfile('fixations_final_data.csv'):
        import degrees_of_vision_spyder

    # run analysis
    main()