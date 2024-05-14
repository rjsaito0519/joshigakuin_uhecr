import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import plotly.graph_objs as go
from datetime import date, timedelta, datetime
import statistics
import csv
import os
import sys
import lmfit as lf
import lmfit.models as lfm
if 'ipykernel' in sys.modules:
    # Jupyter Notebook
    from tqdm import tqdm_notebook as tqdm
else:
    # ipython, python script, ...
    from tqdm import tqdm

# +---------------------------+
# | Cosmic Watchのデータクラス |
# +---------------------------+
class CWData():
    def __init__(self, path):
        self.data ={
            'event':[],
            'date':[],
            'adc':[],
            'temp':[]
        }
        self.path = path
        self.LoadData(path)
    
    def LoadData(self, path):
        df_tsv = pd.read_table(path, header=None)
        tmp_data = df_tsv.values
        self.data["event"] = tmp_data[:, 0]
        self.data["date"]  = pd.to_datetime(tmp_data[:, 1], format='%Y-%m-%d-%H-%M-%S.%f')
        self.data["adc"]   = tmp_data[:, 3]
        self.data["temp"]  = tmp_data[:, 6]

# +---------------------------+
# | Cosmic Watchの解析用クラス |
# +---------------------------+
class AnalysisHelper():
    def __init__(self):
        self.data = np.zeros((0, 3))
        self.time = np.nan
        self.tot_num = np.nan
        self.filename1 = None
        self.filename2 = None
        self.window = None
        self.const = None
        self.ADCth = None

    def plt_setting(self):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.size"] = 20
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
        plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 5                 #x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 5                 #y軸補助目盛り線の長さ

    def coincidence(self, cwdata1, cwdata2, window = 0.1, const = 0., isRate = False, ADCthreshold = -1):
        """
        コインシデンスの解析。尺取り法で時間差を計算している
        """
        # prepare w/ or w/o ADC cut data
        if ADCthreshold == -1:
            time1 = cwdata1.data["date"]
            time2 = cwdata2.data["date"]
            adc1  = cwdata1.data["adc"]
            adc2  = cwdata2.data["adc"]
        else:
            indices1 = [ i for i in range(len(cwdata1.data["adc"])) if cwdata1.data["adc"][i] > ADCthreshold ]
            indices2 = [ i for i in range(len(cwdata2.data["adc"])) if cwdata2.data["adc"][i] > ADCthreshold ]
            time1 = [ cwdata1.data["date"][i] for i in indices1 ]
            adc1  = [ cwdata1.data["adc"][i]  for i in indices1 ]
            time2 = [ cwdata2.data["date"][i] for i in indices2 ]
            adc2  = [ cwdata2.data["adc"][i]  for i in indices2 ]

        # progress bar setting
        max_loop = len(time1) + len(time2)
        pbar = tqdm(total=max_loop)

        # coincidence analysis
        i = 0
        j = 0
        data = []
        while ( i < len(time1) and j < len(time2) ):
            time_diff = (time1[i] - time2[j] ).total_seconds() + const
            if time_diff > window:
                j += 1
            elif time_diff < -1 * window:
                i += 1
            else:
                data.append([ time_diff, adc1[i], adc2[j] ])
                i += 1
                j += 1
            pbar.update(1)
        pbar.close()

        self.data = np.array(data)
        self.filename1 = os.path.basename(cwdata1.path)
        self.filename2 = os.path.basename(cwdata2.path)
        self.window = window
        self.const = const
        self.ADCth = ADCthreshold
        self.data = np.array(data)
        self.time = (np.min([ cwdata1.data["date"][-1], cwdata2.data["date"][-1] ]) - np.max([ cwdata1.data["date"][0], cwdata2.data["date"][0] ])).total_seconds()
        self.tot_num = len(data)

        if isRate:
            return [len(data)/self.time*3600, np.sqrt(len(data))/self.time*3600]
        else:
            return np.array(data)

    # def acci_coin(self, cwdata1, cwdata2, window):
    #     t = window
    #     r1 = len( cwdata1.data["date"] )/ ( cwdata1.data["date"][-1] - cwdata1.data["date"][0] ).total_seconds()
    #     r2 = len( cwdata2.data["date"] )/ ( cwdata2.data["date"][-1] - cwdata2.data["date"][0] ).total_seconds()
    #     sigma_r1 = np.sqrt( len( cwdata1.data["date"] ) ) / ( cwdata1.data["date"][-1] - cwdata1.data["date"][0] ).total_seconds()
    #     sigma_r2 = np.sqrt( len( cwdata2.data["date"] ) ) / ( cwdata2.data["date"][-1] - cwdata2.data["date"][0] ).total_seconds()
    #     R = r1 * ( 1 - np.exp(-r2*t) ) + r2 * ( 1 - np.exp(-r1*t) )
    #     dRdr1 = 1 - np.exp( -r2*t ) + r1 * r2 * np.exp(-r1*t)
    #     dRdr2 = 1 - np.exp( -r1*t ) + r1 * r2 * np.exp(-r2*t)
    #     sigma_R = np.sqrt( dRdr1**2 * sigma_r1**2 + dRdr2**2 * sigma_r2**2 )
    #     return R

    def check_order(self, data):
        """
        描画の際の軸の数値を調整するために次数を計算
        """
        max_value = np.max(np.abs(np.where(~np.isnan(data), data, 0)))
        s = "{:e}".format(max_value)
        # order = 10**(int(s[-3:]))
        return int(s[-3:])

    def rate_plot(self, cwdata1, cwdata2, const = 0., windows = np.linspace(0, 0.02, 10)):
        """
        コインシデンスレートを計算して描画
        """
        # data calculation
        t = np.linspace( np.min( windows ), np.max(windows), 1000 )
        r1 = len( cwdata1.data["date"] )/ ( cwdata1.data["date"][-1] - cwdata1.data["date"][0] ).total_seconds()
        r2 = len( cwdata2.data["date"] )/ ( cwdata2.data["date"][-1] - cwdata2.data["date"][0] ).total_seconds()
        r1_err = np.sqrt( len( cwdata1.data["date"] ) ) / ( cwdata1.data["date"][-1] - cwdata1.data["date"][0] ).total_seconds()
        r2_err = np.sqrt( len( cwdata2.data["date"] ) ) / ( cwdata2.data["date"][-1] - cwdata2.data["date"][0] ).total_seconds()
        dRdr1 = 1 - np.exp( -r2*t ) + r1 * r2 * np.exp(-r1*t)
        dRdr2 = 1 - np.exp( -r1*t ) + r1 * r2 * np.exp(-r2*t)
        acci_R     = ( r1*(1-np.exp(-r2*t)) + r2*(1-np.exp(-r1*t)) ) * 3600
        acci_R_err = np.sqrt( dRdr1**2 * r1_err**2 + dRdr2**2 * r2_err**2 ) * 3600
        coin_R = np.array([ self.coincidence( cwdata1, cwdata2, tmp_window/2, const, True ) for tmp_window in tqdm(windows) ])

        # set xaxis unit
        xorder = self.check_order(windows) // 3
        if xorder <= -1:
            t *= 10**3
            windows *= 10**3
            xlabel = "Time Window [ms]"
        else:
            xlabel = "Time Window [s]"

        # plot
        self.plt_setting()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(t, acci_R, "-", zorder = 2, label = r"$R_{\rm acci}$")
        ax.fill_between(t, acci_R - acci_R_err, acci_R + acci_R_err, color='C0', alpha = 0.3, zorder = 1)
        ax.plot(windows, coin_R[:, 0], "--o", zorder = 3, label = r"$R_{\rm data}$")
        ax.fill_between(windows, coin_R[:, 0] - coin_R[:, 1], coin_R[:, 0] + coin_R[:, 1], color='C1', alpha = 0.3, zorder = 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rate of Coincidences [/hour]")
        ax.yaxis.set_major_formatter(ptick.EngFormatter())
        plt.subplots_adjust(right=0.98, top=0.98)
        plt.legend()
        plt.show()

    def num_of_coincidence(self, cwdata1, cwdata2, const = 0., windows = np.linspace(0, 0.02, 10), cwdata3 = None, cwdata4 = None, label1 = "data1 & data2", label2 = "data3 & data4"):
        """
        コインシデンスしたイベント数を描画
        """
        # data calculation
        flag = False
        num1 = np.array([ len(self.coincidence( cwdata1, cwdata2, tmp_window/2, const, False )) for tmp_window in tqdm(windows) ])
        if (cwdata3 is None) or (cwdata4 is None):
            num2 = np.nan
            flag = True
        else:
            num2 = np.array([ len(self.coincidence( cwdata3, cwdata4, tmp_window/2, const, False )) for tmp_window in tqdm(windows) ])

        # set xaxis unit
        xorder = self.check_order(windows) // 3
        if xorder <= -1:
            windows *= 10**3
            xlabel = "Time Window [ms]"
        else:
            xlabel = "Time Window [s]"

        # plot
        self.plt_setting()
        fig, ax = plt.subplots(figsize=(8, 8))
        if flag: # 1 conbination
            ax.errorbar( windows, num1, yerr = np.sqrt(num1), fmt='o', capsize = 5, markersize=8, ecolor='black', markeredgecolor = "black", color='w')
        else: # 2 conbinations
            ax.errorbar( windows, num1, yerr = np.sqrt(num1), fmt='o', capsize = 5, markersize=8, ecolor='black', markeredgecolor = "black", color='C0', markeredgewidth=0.25, alpha = 0.8, label = label1)
            ax.errorbar( windows, num2, yerr = np.sqrt(num2), fmt='s', capsize = 5, markersize=8, ecolor='black', markeredgecolor = "black", color='C1', markeredgewidth=0.25, alpha = 0.8, label = label2)
            plt.legend()
        ax.yaxis.set_major_formatter(ptick.EngFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Coincidences [counts]")
        plt.subplots_adjust(right=0.98, top=0.98)
        plt.show()

    def plot(self, keyword = "default", bin_num = 100, bin_width_show = False):
        """
        基本的な結果の描画を管理。coincidence()を実行した後に実行しないといけないのに注意
        """
        self.plt_setting()
        if keyword == "default":
            # set xaxis unit
            xorder = self.check_order(self.data[:, 0]) // 3
            if xorder <= -1:
                data = self.data[:, 0]*10**3
                xlabel = "Time Difference [ms]"
                draw_range = self.window*1000
            else:
                xlabel = "Time Difference [s]"
                draw_range = self.window

            # plot
            fig1, ax = plt.subplots(figsize=(8, 8))
            hist_info = ax.hist(data, bins = bin_num)
            if bin_width_show: # for debug etc...
                hist_x = hist_info[1]
                roll_x = np.roll(hist_x, 1)
                bin_width_array = (hist_x - roll_x)[1:]
                bin_width = statistics.mean( bin_width_array )
                ax.plot([], [], " ", label = "bin width = {:.2E} [s]".format(bin_width))
                ax.legend(handletextpad = -2.0)
            ax.set_xlim(-1*draw_range, draw_range)
            ax.yaxis.set_major_formatter(ptick.EngFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Number of Events")
            plt.subplots_adjust(right=0.98, top=0.98)
            
            # plot w/ summary info
            fig2, (ax1, ax2) = plt.subplots(1, 2, sharey = True, gridspec_kw={'width_ratios': [4, 3]}, figsize=(12,8))
            ax1.hist(data, bins = bin_num)
            ax1.set_xlim(-1*draw_range, draw_range)
            ax1.yaxis.set_major_formatter(ptick.EngFormatter())
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("Number of Events")
            table_data = [
                ["data1", self.filename1],
                ["data2", self.filename2],
                ["window [s]", self.window],
                ["const [s]", self.const],
                ["threshold", self.ADCth],
                ["event", self.tot_num],
                ["time [s]",self.time],
                ["rate [/s]", self.tot_num/self.time],
            ]
            table = ax2.table(cellText=table_data, cellLoc='left', rowLoc='center', bbox = [0, 0, 1, 1], colWidths=[0.35, 0.65])
            table.auto_set_font_size(False)
            table.set_fontsize(18)
            ax2.tick_params(labelbottom=True, bottom=False)
            ax2.tick_params(labelleft=False, left=False)
            ax2.axis('off')
            plt.subplots_adjust(right=0.98, top=0.98, wspace=0)
        elif keyword == "adc": # plot ADC hist
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(self.data[:, 1], bins = bin_num, range = [0, 1023], alpha = 0.5, label = "data1")
            ax.hist(self.data[:, 2], bins = bin_num, range = [0, 1023], alpha = 0.5, label = "data2")
            ax.yaxis.set_major_formatter(ptick.EngFormatter())
            ax.set_xlabel("ADC [arb. unit]")
            ax.set_ylabel("Number of Events")
            plt.subplots_adjust(right=0.98, top=0.98)
            plt.legend()
        elif keyword == "adc_subplot": # plot ADC hist separately
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.hist(self.data[:, 1], bins = bin_num, range = [0, 1023])
            ax2.hist(self.data[:, 2], bins = bin_num, range = [0, 1023])
            ax1.yaxis.set_major_formatter(ptick.EngFormatter())
            ax2.yaxis.set_major_formatter(ptick.EngFormatter())
            ax2.set_xlabel("ADC [arb. unit]")
            fig.supylabel("Number of Events")
            plt.subplots_adjust(hspace=0, right=0.98, top=0.98)
        elif keyword == "adc_2d": # plot ADC 2d hist
            fig, ax = plt.subplots(figsize=(10, 8))
            hist_data = ax.hist2d( self.data[:, 1], self.data[:, 2], bins = [ 32, 32 ], range = [ [0, 1023], [0, 1023] ] )
            fig.colorbar(hist_data[3], ax=ax)
            ax.set_xlabel("data1 ADC [arb. unit]")
            ax.set_ylabel("data2 ADC [arb. unit]")
            ax.set_aspect('equal', adjustable='box')
            plt.subplots_adjust(right=0.98, top=0.98)
        plt.show()

    def plot_adc_cut_histo(self, cwdata1, cwdata2, window = 0.1, const = 0., ADCthreshold = -1, bin_num = 50):
        """
        ADC cut ありなしのヒストグラムの比較
        """
        # data calculation
        data1 = self.coincidence( cwdata1, cwdata2, window = window, const = const, ADCthreshold = 0 )
        data2 = self.coincidence( cwdata1, cwdata2, window = window, const = const, ADCthreshold = ADCthreshold )

        # set xaxis unit
        xorder = self.check_order(data1[:, 0]) // 3
        if xorder <= -1:
            data1[:, 0] *= 10**3
            data2[:, 0] *= 10**3
            xlabel = "Time Difference [ms]"
            draw_range = self.window*1000
        else:
            xlabel = "Time Difference [s]"
            draw_range = self.window

        self.plt_setting()
        fig, ax = plt.subplots(figsize=(8, 8))
        histo_info = ax.hist(data1[:, 0], bins = bin_num, alpha = 0.5, label = r"ADC $\geq$ 0")
        ax.hist(data2[:, 0], bins = bin_num, range = [ histo_info[1][0], histo_info[1][-1] ], alpha = 0.5, label = "ADC > {}".format(ADCthreshold))
        ax.set_xlim(-1*draw_range, draw_range)
        ax.yaxis.set_major_formatter(ptick.EngFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Events")
        plt.legend(borderaxespad=1, handletextpad = 0.5, handlelength=1., fontsize = 16)
        plt.subplots_adjust(right=0.98, top=0.98)
        plt.show()

    def fit(self, bin_num = 100, range_min = np.nan, range_max = np.nan, isLorentzian = False):
        """
        peak位置をfittingで求める。coincidence()を実行した後に実行しないといけないのに注意
        """
        self.plt_setting()
        fig, ax = plt.subplots(figsize=(10, 8))
        if np.isnan(range_min):
            range_min = np.min( self.data[:, 0] )
        if np.isnan(range_max):
            range_max = np.max( self.data[:, 0] )

        # set xaxis unit
        xorder = self.check_order(self.data[:, 0]) // 3
        if xorder <= -1:
            data = self.data[:, 0] * 10**3
            range_min *= 10**3
            range_max *= 10**3
            xlabel = "Time Difference [ms]"
        else:
            xlabel = "Time Difference [s]"

        # plot hist
        hist_info = ax.hist(data[ (data >= range_min) & ( data <= range_max ) ], bins = bin_num, histtype='step', color='k')
        y = hist_info[0]
        tmp_x = hist_info[1]

        # prepare fitting
        roll_x = np.roll(tmp_x, 1)
        x = (tmp_x/2 + roll_x/2)[1:]
        bin_width = statistics.mean( (tmp_x-roll_x)[1:] )
        # init value
        c = statistics.mean(y)
        amplitude = np.max(y) - c
        center = x[np.argmax(y)]
        sigma = 0.1
        # fit model setting
        if isLorentzian:
            model = lfm.LorentzianModel() + lfm.ConstantModel()
        else:
            model = lfm.GaussianModel() + lfm.ConstantModel()
        params = model.make_params(amplitude=amplitude, center=center, sigma=sigma, c=c)
        result = model.fit(x=x, data=y, params=params, method='leastsq')
        print(result.fit_report())
        # count events
        tmp_y = y - np.full_like(y, result.result.params["c"].value)
        left = result.result.params["center"].value - 2*result.result.params["sigma"].value
        right = result.result.params["center"].value + 2*result.result.params["sigma"].value
        indices1 = np.where( (x >= left) & (x <= right) )[0]
        print("\n\n number of event(N) = ", sum( tmp_y[indices1] ) )
        print(" measurement time(T) [s] = ", self.time )
        print("\n N/T [/s] = ", sum( tmp_y[indices1] )/self.time )

        # plot
        fit_x = np.linspace(np.min(x), np.max(x), 10000)
        if isLorentzian:
            fit_y = result.eval_components(x=fit_x)["lorentzian"] +  result.eval_components(x=fit_x)["constant"]
        else:
            fit_y = result.eval_components(x=fit_x)["gaussian"] +  result.eval_components(x=fit_x)["constant"]
        ax.plot(fit_x, fit_y, color = "C1", ls = "dashed")
        fill_x = np.array([ tmp_x[i//2] for i in range(len(tmp_x)*2) ])[1:-1]
        fill_y = np.array([ y[i//2] for i in range(len(fill_x)) ])
        indices2 = np.where( (fill_x+bin_width/2 >= left) & (fill_x-bin_width/2 <= right) )[0]
        ax.fill_between(fill_x[indices2], np.full_like(fill_x[indices2], result.result.params["c"].value), fill_y[indices2], facecolor='C0', alpha=0.5)
        ax.yaxis.set_major_formatter(ptick.EngFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Events")
        plt.subplots_adjust(right=0.98, top=0.98)
        plt.show()

    # def compare(self, cwdata1.data, cwdata2.data, window = 0.1, const = 0., isRate = False, bin_num = 100):
    #     plt.rcParams['mathtext.fontset'] = 'stix'
    #     plt.rcParams['font.size'] = 18
    #     plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
    #     plt.rcParams['axes.grid'] = True

    #     self.coincidence( cwdata1.data, cwdata2.data, window, const )
    #     fig1, ax1 = plt.subplots(figsize=(10, 8))
    #     ax1.hist(self.data[:, 0], bins = bin_num, histtype='step', color='k', zorder = 2)
    #     ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax1.ticklabel_format(style="sci",  axis="both",scilimits=(0,0))
    #     ax1.set_xlabel("Time Difference [s]")
    #     ax1.set_ylabel("Number of Events")

    #     counter = 1
    #     for raw_data in [cwdata1.data, cwdata2.data]:
    #         diff = []
    #         for i in tqdm(range(1, len(raw_data["date"]))):
    #             diff.append( (raw_data["date"][i] - raw_data["date"][i-1]).total_seconds() )

    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         hist_info = ax.hist(diff, bins = 1000, histtype='step', color='k', range = (0., 15))
    #         y = hist_info[0]
    #         tmp_x = hist_info[1]
    #         roll_x = np.roll(tmp_x, 1)
    #         x = (tmp_x/2 + roll_x/2)[1:]

    #         model = lfm.ExponentialModel()
    #         params = model.guess(x=x, data=y)
    #         result = model.fit(x=x, data=y, params=params, method='leastsq')
    #         print(result.fit_report())

    #         fit_x = np.linspace(np.min(x), np.max(x), 10000)
    #         fit_y = result.eval_components(x=fit_x)["exponential"]
    #         ax.plot(fit_x, fit_y, color = "C1", ls = "dashed")
    #         ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #         ax.ticklabel_format(style="sci",  axis="both",scilimits=(0,0))
    #         ax.set_xlabel("Time Difference [s]")
    #         ax.set_ylabel("Number of Events")
    #         ax.set_yscale("log")
    #         fig.show()

    #         random_diff = np.random.exponential(  result.result.params["decay"].value, 10**6)

    #         tmp_data = {"date":[], "adc":[]}
    #         tmp_time = datetime(2022, 12, 8, 0, 0, 0)

    #         for tmp_diff in random_diff:
    #             tmp_time += timedelta(seconds=tmp_diff)
    #             tmp_data["date"].append(tmp_time)
    #             tmp_data["adc"].append(0)
    #             if (tmp_time - datetime(2022, 12, 8, 0, 0, 0)).total_seconds() > self.time:
    #                 print("berak")
    #                 break

    #         if counter == 1:
    #             new_data1 = tmp_data
    #             counter += 1
    #         else:
    #             new_data2 = tmp_data

    #     self.coincidence( new_data1, new_data2, window, 0. )
    #     ax1.hist(self.data[:, 0], bins = bin_num, zorder = 1, alpha = 0.5, color = "C0")
    #     plt.show()

def cut_data(pathlist, range_min, range_max, save_name = "test"):
    data = []
    for path in tqdm(pathlist):   # ファイルを読み込む
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                for row in reader:
                    date = pd.to_datetime(row[1], format='%Y-%m-%d-%H-%M-%S.%f')
                    if date >= range_min and date <= range_max:
                        data.append(row)
            except:
                # print ('failed to load data: ',row)
                pass
    data.sort(key=lambda x: x[1])
    if save_name == "":
        save_name = "{}".format(pd.to_datetime(data[0][1], format='%Y-%m-%d-%H-%M-%S.%f').date())
    with open("/content/drive/Shareddrives/宇宙線探究/女子学院/data/cut_data/{}.dat".format(save_name), "w") as f: # ファイルの保存
        for i, row in enumerate(data):
            s = '\t'.join([str(i)] + row[1:]) + "\n"
            f.write(s)

def check(path):
    data = [[], []]
    if type(path) == list:
        for tmp_path in path:
           tmp_data = CWData(tmp_path)
           data[0].extend( tmp_data.data["date"] )
           data[1].extend( tmp_data.data["event"] )
    else:
        tmp_data = CWData(path)
        data[0].extend( tmp_data.data["date"] )
        data[1].extend( tmp_data.data["event"] )

    plt.figure(figsize=(8, 6))
    plt.plot( data[0], data[1], ".")
    plt.xticks(rotation =45)
    plt.show()

    trace = go.Scatter(
        x = data[0],
        y = data[1],
        mode = 'lines',
        marker= dict(
            color= "blue",
            opacity= 1.0
        ),
    )
    plotdata =[trace]
    layout = go.Layout(
        font = dict(       # グローバルのフォント設定
            size = 18,
        ),
        xaxis=dict(
            title='日付',
        ),
        yaxis1=dict(
            title='イベント番号'
        )
    )
    fig = go.Figure(data=plotdata, layout=layout)
    fig.update_layout(margin=dict(t=15, b=20, l=25, r=25))
    fig.show()

if __name__ == '__main__':
    print()

    data1 = CWData("data/2024-04-21-23_1.dat")
    data2 = CWData("data/2024-04-21-23_2.dat")
    # data3 = CWData("data/2024-04-21-23_3.dat")
    # data4 = CWData("data/2024-04-21-23_4.dat")
    # data1 = CWData("data/2024-05-02-09_1.dat")
    # data2 = CWData("data/2024-05-02-09_2.dat")

    coin = AnalysisHelper()
    # coin.num_of_coincidence(data1, data2, 0, np.linspace(0, 5, 10), cwdata3 = data3, cwdata4 = data4)
    # coin.num_of_coincidence(data1, data2, 0, np.linspace(0, 5, 5))
    # coin.rate_plot(data1, data2, 0, np.linspace(0, 5, 5))

    # coin.coincidence(data2, data1, 0.03, 0)
    # coin.fit()
    # print(coin.data)
    # coin.plot(keyword="default", bin_num= 30)
    # coin.plot(keyword="adc", bin_num= 30)
    # coin.plot(keyword="adc_subplot", bin_num= 30)
    # coin.plot(keyword="adc_2d", bin_num= 30)
    # coin.plot_adc_cut_histo(data2, data1, 0.1, 0, 150)