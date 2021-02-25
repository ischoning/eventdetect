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


if __name__ == "__main__":
    # Testing
    # hello("Isabella")

    # data pipeline
    if not os.path.isfile('fixations_final_data.csv'):
        import degrees_of_vision_spyder

    # run analysis
    main()