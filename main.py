import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import scipy as scipy
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
# Steps :
#   1: Initializations
#       1a. Import Data into numpy matrix from csv files
#           - Temp vs Time Data for Ice Water
#           - Calibration Data for Each Run
#       1b. Rod dimension data
#   2. Data Clean
#       2a. Temp vs Time Averages (Measured Steady State Data): 5 min average intervals for steady state data
#       2b. Calibration Data: Find Standard Deviation in Each Sensor
#       2c. Calculation of Uncertainty of Steady State Temperature
#           - Note: Uncertainty Parameter is simply Uncertainty_Average + Uncertainty_Calibration
#   3. Fit Function Calc and Analysis
#       3a. Parameter Calculation
#           - Note: Sigma Parameter in .curve_fit() is from 2c
#           - Note2: p0 is based on an Approximate Expected Value for h_0,
#                       Ambient Air Temperature from Calibration,
#                       and an Approximate Expected Value for m
#       3b. Analysis:
#           - R value: scipy.stats.spearmanr()
#
#
#   4. Graphing
#       4a. Run Data (Raw): Note, no error bars are showing on measurement graph,
#                           this is due to the graph being a representation of the instrumental 'read-off' data
#                           and not the analysis of it. Due to the fact that the error bars come from the data itself,
#                            their inclusion is inappropriate.
#       4b. Calibration Data (Raw): Note, --//--
#       4c. Fit Data: Note the error bars


def main():
    # 1. Initializations:
    graph_raw = True
    graph_fit = True
    # ----- 1a. Import Data

    rnd_dat_raw2 = np.transpose(np.genfromtxt('Round_2_1_30.txt', delimiter=','))   # Temp vs Time Data
    rnd_dat_cal2 = np.genfromtxt('Calibration_Round_2.txt', delimiter=',')          # Calibration Data

    rnd_dat_raw3 = np.transpose(np.genfromtxt('Round_3_1_30.txt', delimiter=','))   # Temp vs Time Data
    rnd_dat_cal3 = np.genfromtxt('Calibration_Round_3.txt', delimiter=',')          # Calibration Data

    sq_dat_raw2 = np.transpose(np.genfromtxt('Square_2_1_30.txt', delimiter=','))   # Temp vs Time Data
    sq_dat_cal2 = np.genfromtxt('Calibration_Square_2.txt', delimiter=',')          # Calibration Data

    # ----- 1b. Rod Dimension Data

    x_square = [.018, .052, .088, .121, .159]  # Water location, Sensor Coord. from Water (m)
    l_sq = 0.215    # Length of Square Bar

    x_round = [.022, .038, .072, .107, .146]   # Water location, Sensor Coord. from Water (m)
    l_rnd = 0.306   # Length of Round Bar

    dx = 0.001      # Uncertainty of x_measurement (m) from the size of the sensor.

    # ----- 1c. Graph Raw Data vs Time

    #   -----> Runs Raw

    # Round 2 run
    if graph_raw:
        '''
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x_size = np.ones(360)
        for i in range(0,5):
            x_data = x_size*x_round[i]
            # ** print(x_data)
            y_data = rnd_dat_raw2[0]
            # ** print(y_data)
            z_data = rnd_dat_raw2[i+1]
            # ** print(z_data)
            ax.scatter3D(y_data, x_data, z_data,  label='a_'+str(i))  # c=z_data, cmap='Greens')

        ax.legend(loc='best')
        ax.set_ylabel('x position (m)')
        ax.set_xlabel('Time (sec)')
        ax.set_zlabel('Temperature (deg C)')

        ax.set_title('Round Rod Sample Data 2')
        fig.savefig("rnd2_raw.png", format="png", dpi=1000)
        '''
        # -------- Round 3-------------

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x_size = np.ones(360)
        for i in range(0, 5):
            x_data = x_size * x_round[i]
            # ** print(x_data)
            y_data = rnd_dat_raw3[0]
            # ** print(y_data)
            z_data = rnd_dat_raw3[i + 1]
            # ** print(z_data)
            ax.scatter3D(y_data, x_data, z_data, label='a_' + str(i))  # c=z_data, cmap='Greens')

        ax.set_ylabel('x position (m)')
        ax.set_xlabel('Time (sec)')
        ax.set_zlabel('Temperature (deg C)')
        ax.legend(loc='best')
        ax.set_title('Round Rod Sample Data 3')
        fig.savefig("rnd_raw.png", format="png", dpi=1000)

        # -------- Square 2 ---------------


        fig = plt.figure(figsize=(6, 5), layout='constrained')
        ax = plt.axes(projection='3d')
        x_size = np.ones(360)
        for i in range(0, 5):
            x_data = x_size * x_square[i]
            # ** print(x_data)
            y_data = sq_dat_raw2[0]
            # ** print(y_data)
            z_data = sq_dat_raw2[i + 1]
            # ** print(z_data)
            ax.scatter3D(y_data, x_data, z_data, label='a_' + str(i))  # c=z_data, cmap='Greens')

        ax.set_ylabel('x position (m)')
        ax.set_xlabel('Time (sec)')
        ax.set_zlabel('Temperature (deg C)')
        ax.legend(loc='best')
        ax.set_title('Square Rod Temperature vs Position vs Time')
        fig.savefig("sq_raw.png", format="png", dpi=1000)
    
    # 2. Clean Data:

    # ----- 2a. Equilibrium Pts. Calculation: (See function equi_vals() for more documentation)
    #   -----> Note (Naming Convension):    rnd_eq2 -> Equilibrium Values for 2nd run of Round Rod
    #                                       sig_rnd2 -> Uncertainty values for 2nd run of Round Rod
    rnd_eq2, sig_rnd2 = equi_vals(rnd_dat_raw2)
    print("Here!: ", rnd_eq2, sig_rnd2)
    rnd_eq3, sig_rnd3 = equi_vals(rnd_dat_raw3)
    sq_eq2, sig_sq2 = equi_vals(sq_dat_raw2)

    # ----- 2b. Uncertainty Values in Sensor Calibration Data:
    #   -----> Note (Derivation):   Uncertainty_Calibration = Rounded 2 Sig-Fig (Standard Deviation
    #                                                                               /   Sqrt(N)     )
    sig_rnd_cal2 = np.around(np.divide(np.std(rnd_dat_cal2[:, 1:], axis=0),
                                       np.sqrt(np.shape(rnd_dat_cal2)[0])), decimals=2)
    sig_rnd_cal3 = np.around(np.divide(np.std(rnd_dat_cal3[:, 1:], axis=0),
                                       np.sqrt(np.shape(rnd_dat_cal3)[0])), decimals=2)
    sig_sq_cal2 = np.around(np.divide(np.std(sq_dat_cal2[:, 1:], axis=0),
                                       np.sqrt(np.shape(sq_dat_cal2)[0])), decimals=2)

    print("Sensor Cal(Round2): ", sig_rnd_cal2)
    print("Sensor Cal(Round3): ", sig_rnd_cal3)
    print("Sensor Cal(Square2): ", sig_sq_cal2)

    # 3. Fit Function Calculation and Analysis

    # ----- 3a. Fit Function Coefficient Calculation

    print("-------Round 2-----------")
    t_rnd2 = 3 # Choice of 5 minute Interval (here 20-25 minutes)

    rnd2_dat = rnd_eq2[t_rnd2, 1:]
    rnd2_sig = sig_rnd2[t_rnd2, 1:]+sig_rnd_cal2
    '''
    # print('rnd_eq2\n', rnd_eq2)
    # print('sig_rnd2\n', sig_rnd2)
    # print('rnd2_dat', rnd2_dat)
    # print(rnd2_sig)
    '''
    popt_rnd2, pcov_rnd2 = scipy.optimize.curve_fit(_inter_rnd, np.append(x_round[0:1],x_round[3:4]),
                                                    np.append(rnd2_dat[0:1],rnd2_dat[3:4]),
                                                    sigma=np.append(rnd2_sig[0:1], rnd2_sig[3:4]),
                                                    absolute_sigma=True, p0=[150, 24.7, 30], bounds=(0,1000))
    perr_rnd2 = np.sqrt(np.diag(pcov_rnd2))

    print("popt", popt_rnd2)
    print("perr", perr_rnd2)

    print("-------Round 3-----------")
    t_rnd3 = 4
    print(rnd_eq3)
    print(sig_rnd3)

    rnd3_dat = rnd_eq3[t_rnd3, 1:]
    rnd3_sig = sig_rnd3[t_rnd3, 1:]
    popt_rnd3, pcov_rnd3 = scipy.optimize.curve_fit(_inter_rnd, x_round, rnd3_dat, sigma=rnd3_sig+sig_rnd_cal3,
                                                    absolute_sigma=True, p0=[150, 25.9, 30])
    perr_rnd3 = np.sqrt(np.diag(pcov_rnd3))
    r_val_rnd3 = np.corrcoef(rnd3_dat, _inter_rnd(x_round, *popt_rnd3))
    print("R value: ", r_val_rnd3)
    print('rnd3_dat: ', rnd3_dat, '\nFit Data: ', _inter_rnd(x_round, *popt_rnd3))
    print("popt", popt_rnd3)
    print("perr", perr_rnd3)

    print("-------Square 2-----------")
    t_sq2 = 2
    print(sq_eq2)
    print(sig_sq2)

    sq2_dat = sq_eq2[t_sq2, 1:]

    # The overall uncertainty/sigma is the fluxuation of the 5 minute average (sig_sq2[t_sq2, 1:])
    # plus the uncertainty of the calibration temperature (sig_sq_cal2)
    sq2_sig = sig_sq2[t_sq2, 1:] + sig_sq_cal2

    popt_sq2, pcov_sq2 = scipy.optimize.curve_fit(_inter_sq, x_square, sq2_dat, sigma=sq2_sig+sig_sq_cal2,
                                                  absolute_sigma=True, p0=[150, 25.2, 30])
    perr_sq2 = np.sqrt(np.diag(pcov_sq2))
    r_val_sq2 = np.corrcoef(sq2_dat, _inter_rnd(x_square, *popt_sq2))
    print("R value: ", r_val_sq2)
    print("popt", popt_sq2)
    print("perr", perr_sq2)
# 4. Graphs
# ----- 4a. Raw Plot Data



# ---------- Plot Best Fit ----------
    if graph_fit:
        # ----- Round 2 -----

        fig2, ax2 = plt.subplots(figsize=(6, 4), layout='constrained')
        plt.grid(True)
        x_int2 = np.linspace(0, l_rnd, 50)
        ax2.errorbar(x_round, rnd2_dat, xerr=dx, yerr=sig_rnd2[t_rnd2, 1:]+sig_rnd_cal2, marker='.', linestyle='', label='rounded data')
        ax2.plot(x_int2, _inter_rnd(x_int2, *popt_rnd2), 'g--',
                 label='fit: h_0=%5.2f +- %5.2f (W/m^2K),\nT_a=%5.1f +- %5.1f (deg C),\nm=%5.2f +- %5.2f (1/m)'
                       % tuple(np.ravel(np.transpose([popt_rnd2, perr_rnd2]))))
        ax2.set_xlabel('x position (m)')
        ax2.set_ylabel('Steady-State Temperature (deg C)')
        ax2.set_title('Round Rod Sample Data 2(edited2)')
        ax2.legend()
        fig2.savefig("rnd2.png", format="png", dpi=1000)

        # ----- Round 3 -----

        fig1, ax1 = plt.subplots(figsize=(6, 4), layout='constrained')
        plt.grid(True)
        x_int1 = np.linspace(0, l_rnd, 50)
        ax1.errorbar(x_round, rnd3_dat, xerr=dx, yerr=sig_rnd3[t_rnd3, 1:], marker='.', linestyle='', label='rounded data')
        ax1.plot(x_int1, _inter_rnd(x_int1, *popt_rnd3), 'g--',
                 label='fit: R^2=%5.5f\nh_0=%5.2f +- %5.2f (W/m^2K),\nT_a=%5.1f +- %5.1f (deg C),\nm=%5.2f +- %5.2f (1/m)'
                       % tuple(np.append([r_val_rnd3[0][1]**2],np.ravel(np.transpose([popt_rnd3, perr_rnd3])))))
        ax1.set_xlabel('x position (m)')
        ax1.set_ylabel('Steady-State Temperature (deg C)')
        ax1.set_title('Round Rod Sample Data 3')
        ax1.legend()
        fig1.savefig("rnd3.png", format="png", dpi=1000)

        # ----- Square 2 -----

        fig3, ax3 = plt.subplots(figsize=(6, 4), layout='constrained')
        plt.grid(True)
        x_int = np.linspace(0, l_sq, 50)
        ax3.errorbar(x_square, sq2_dat, xerr=dx, yerr=sig_sq2[t_sq2, 1:],
                     marker='.', linestyle='', label='5 Minute Averages')
        ax3.plot(x_int, _inter_sq(x_int, *popt_sq2), 'g-',
                 label='fit: R^2=%5.5f\nh_0=%5.0f +- %5.2f (W/m^2K),\nT_a=%5.1f +- %5.1f (deg C),\nm=%5.2f +- %5.2f (1/m)'
                       % tuple(np.append([r_val_rnd3[0][1]**2],np.ravel(np.transpose([popt_sq2, perr_sq2])))))
        ax3.set_xlabel('x position (m)')
        ax3.set_ylabel('Steady-State Temperature (deg C)')
        ax3.set_title('Square Rod Sample Data 2')
        ax3.legend()
        fig3.savefig("sq2.png", format="png", dpi=1000)
    return 0


def equi_vals(mat):
    # Purpose of Function:
    #   Find Equilibrium Values for Temperature Data
    #   Assumed, if at Equilibrium, Temperature is Relatively Stable, thus Average gives good approx. equilib
    # Function Practice:
    #   Takes average over 5 minute intervals,
    #   Should return dat in 5 min average intervals in form: np.array([]), np.array([])
    #       np.array([[5,avg(a_0(from 0-5min)), avg(a_1(0-5)),..., avg(a_4)(...)],
    #       [10,a_0(5-10),...],...,[30,...]])
    #       ,
    #       np.array([[std(a_0(from 0-5min)), std(a_1(0-5)),...,std(a_4(...))],
    #       [std(a_0(5-10),...][std(a_0(25-30)),...]])
    if np.shape(np.ravel(mat)) != (2160,):
        # Because the runs are coded for 5 second interval recordings for 30 minutes,
        # the number of expected data points in the input matrix is known and should be the same for each run.
        # That is 6 sets of 30*60/5 data points (one time column and 5 sensor columns) or 2160 total numbers in the .csv
        print(np.shape(np.ravel(mat)))  # Prints number of data points in matrix
        print("Error: Improper Size Data Matrix")
        return 0
    else:
        mat2 = np.reshape(np.ravel(mat), (36, 60))  # makes each row a 5 minute intervals of data

        mat_avg = np.around(np.average(mat2, axis=1), decimals=2)
        # mat_avg is rounded average of 5 min interval of data pts to maintain sig figs.

        mat_std = np.around(np.divide(np.std(mat2, axis=1), np.sqrt(np.shape(mat2)[1])), decimals=2)
        # mat_std is uncertainty in calibration offset:
        #   at its core, the uncertainty is just the rounded quantity of, standard deviation divided by sqrt(N).

        mat_avg_shape = np.reshape(mat_avg, (6, 6))     # Re-organizes data into more workable form
        mat_std_shape = np.reshape(mat_std, (6, 6))

        mat_avg_shape[0] = [5, 10, 15, 20, 25, 30]      # Makes Time Row Readable
        mat_std_shape[0] = [5, 10, 15, 20, 25, 30]

        mat_avg_final = np.transpose(mat_avg_shape)     # Takes the transpose to properly have (Time, Data) Rows
        mat_std_final = np.transpose(mat_std_shape)

        return mat_avg_final, mat_std_final     # Returns Matrix of Interval Averages and Matrix of Uncertainty


def _inter_rnd(x, h_0, t_a, m):
    k = 14.6    # Thermal Conductivity of 304 Stainless Steel (W/K m)
    length = .306    # Length of Bar (m)
    del_x = m*np.subtract(length, x)
    eq_temp = t_a - (h_0*t_a*np.cosh(del_x))/(h_0*math.cosh(m*length) + m*k*math.sinh(m*length))
    return eq_temp


def _inter_sq(x, h_0, t_a, m):
    k = 14.6    # Thermal Conductivity of 304 Stainless Steel (W/K m)
    length = .215    # Length of Bar (m)
    del_x = m*np.subtract(length, x)
    eq_temp = t_a - (h_0*t_a*np.cosh(del_x))/(h_0*math.cosh(m*length) + m*k*math.sinh(m*length))
    return eq_temp


'''
def graph_plot(name_str, x_array, y_array_gauss, title, axis_labels, label_vect, plot_dat_arr, **kwargs):
    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(x_array, y_array_gauss, "ro")

    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-0.5, 5)

    ax1.set_xlabel("x_array", family="serif", fontsize=12)
    ax1.set_ylabel("y_array", family="serif", fontsize=12)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.tick_params(axis='both', which='major', direction="out", top="off", right="on", bottom="on", length=8,
                    labelsize=8)
    ax1.tick_params(axis='both', which='minor', direction="out", top="off", right="on", bottom="on", length=5,
                    labelsize=8)

    fig.tight_layout()
    fig.savefig(name_str, format="png", dpi=1000)
'''

main()
