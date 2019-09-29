import matplotlib.pyplot as plt
import ruptures as rpt
import json
import numpy as np
import time
from sys import argv
import demjson


# No pre-set true change point
def predict_display(signal, computed_chg_pts=None):
    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)    # Reshape the signal to n rows, 1 column
    n_samples, n_features = signal.shape

    # Set all options
    figsize = (10, 2 * n_features)  # figure size
    color = "k"  # color of the lines indicating the computed_chg_pts
    linewidth = 3  # linewidth of the lines indicating the computed_chg_pts
    linestyle = "--"  # linestyle of the lines indicating the computed_chg_pts

    fig, axarr = plt.subplots(n_features, figsize=figsize, sharex=True)
    if n_features == 1:
        axarr = [axarr]

    for axe, sig in zip(axarr, signal.T):
        # plot s
        axe.plot(range(n_samples), sig)

        # vertical lines to mark the computed_chg_pts
        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(x=bkp - 0.5,
                                color=color,
                                linewidth=linewidth,
                                linestyle=linestyle)

    fig.tight_layout()

    return fig, axarr


# Process json string to np.ndarray signal and record the time
def signal_process(json_str: str) -> (np.ndarray, list):
    signal_dic = json.loads(json_str)
    sort_dic = {}

    for key in sorted(signal_dic.keys()):
        sort_dic[key] = signal_dic[key]
    s_v = list(sort_dic.values())   # signal value
    signal = np.array(s_v)

    t_v = [int(t) for t in list(sort_dic.keys())]   # time value
    time_list = []
    for timestamp in t_v:
        time_local = time.localtime(timestamp)
        time_list.append(time.strftime("%Y-%m-%d %H:%M:%S", time_local))

    return signal, time_list


# Choose an algorithm to predict the result of change point list
# n_bkps: number of change point
def bkp_predict(signal: np.ndarray, n_bkps: int) -> list:
    algo = rpt.Dynp(model='l2').fit(signal)
    result = algo.predict(n_bkps=n_bkps)

    # algo = rpt.Window(width=10, model='l2').fit(signal)
    # result = algo.predict(n_bkps=n_bkps)

    # algo = rpt.Binseg(model='l2').fit(signal)
    # result = algo.predict(n_bkps=n_bkps)

    # algo = rpt.BottomUp(model='l2').fit(signal)
    # result = algo.predict(n_bkps=n_bkps)

    # algo = rpt.Omp(min_size=2).fit(signal)
    # result = algo.predict(n_bkps=n_bkps)

    return result


#  The time of the change points
def bkp_time(result: list, time_list: list) -> list:
    bkp_time_list = []
    for i in result[:-1]:
        bkp_time_list.append(time_list[i])

    return bkp_time_list


# Find change point in one signal data flow
# [time_start, time_end] = getAbruptChangeRange(int timepoint, data[] )
def getAbruptChangeRange(data: str) -> list:
    signal, time_list = signal_process(data)
    result = bkp_predict(signal, 2)
    # print(bkp_time(result, time_list))
    predict_display(signal, result)
    plt.show()
    return bkp_time(result, time_list)


def computeDelta(signal: np.ndarray) -> float:
    upper = 0.9 * signal.max()
    below = 1.1 * signal.min()
    delta = upper - below
    return delta


def main(filename: str):
    with open(filename) as file:
        for line in file:
            signal, time_list = signal_process(line)
            result = bkp_predict(signal, 2)
            print('异常点为：', bkp_time(result, time_list))
            print('时间序列的 delta 为：', computeDelta(signal))
            print('=' * 100)
            predict_display(signal, result)
            plt.show()


if __name__ == '__main__':
    # # -------------- 若直接读取本地的json文件，用这段代码 --------------------------
    # log_file = r"C:\Users\w50005335\Desktop\data.json"
    # main(log_file)
    # # ------------------------------------------------------------------------------

    # ---------- 若从 java 程序里传递 json 文件路径参数，用这段代码 ----------------
    log_file = argv[1]
    main(log_file)
    # ------------------------------------------------------------------------------

    # # ------------ 若从 java 程序里传递单独的一段 json 字符串，用这段代码 ----------
    # json_data = demjson.decode(argv[1])
    # json_str = json.dumps(json_data)
    #
    # time_start, time_end = getAbruptChangeRange(json_str)
    # print(time_start)
    # print(time_end)
    # # ------------------------------------------------------------------------------

    # # ------------ 若直接分析一段单独的 json 字符串，用这段代码 --------------------
    # json_str = '{"1561913400":4.4196485E9,"1561881600":4.6552146E9,"1561942800":7.1144233E9,"1561957740":8.0242033E9,"1561885800":5.4144666E9,"1561875600":4.3789404E9,"1561879800":4.5012844E9,"1561953540":1.20566057E10,"1561932600":5.0253384E9,"1561917600":4.4634163E9,"1561936800":3.82557338E9,"1561926600":4.14572186E9,"1561947540":1.50036183E10,"1561891800":4.6486825E9,"1561888800":4.6535875E9,"1561945800":1.50554573E10,"1561876800":5.11072E9,"1561922400":4.5116058E9,"1561941600":6.2651423E9,"1561903200":5.6121114E9,"1561907400":5.2482509E9,"1561956540":8.8836639E9,"1561880400":4.8173763E9,"1561884600":4.5765663E9,"1561952340":1.56685435E10,"1561933800":3.43927859E9,"1561912200":4.08181632E9,"1561916400":4.7291612E9,"1561927800":3.84530688E9,"1561894800":4.8031032E9,"1561890600":4.6913101E9,"1561930800":3.98551373E9,"1561887600":4.742913E9,"1561873800":4.6286981E9,"1561923600":4.6096256E9,"1561940400":4.5303542E9,"1561902000":4.9305027E9,"1561906200":4.4903306E9,"1561944600":1.33641196E10,"1561915800":4.7595223E9,"1561911600":4.7395077E9,"1561938000":4.14502912E9,"1561883400":4.5694152E9,"1561951140":1.41220925E10,"1561955340":9.1284716E9,"1561901400":4.4303498E9,"1561886400":5.1259244E9,"1561893600":4.3023104E9,"1561949340":1.47891354E10,"1561897800":4.3074744E9,"1561909800":4.7807718E9,"1561943400":9.4593096E9,"1561882200":5.43742E9,"1561920600":5.0325064E9,"1561924800":4.28210432E9,"1561905600":5.332137E9,"1561914600":6.3439882E9,"1561958340":7.2600468E9,"1561935000":3.8026921E9,"1561954140":1.14995845E10,"1561929000":4.5839877E9,"1561895400":4.7916211E9,"1561899600":4.4913316E9,"1561939200":3.74853146E9,"1561948140":1.50451845E10,"1561910400":4.05289779E9,"1561918800":4.8468285E9,"1561878000":4.7890181E9,"1561900200":4.8259589E9,"1561889400":5.0664131E9,"1561892400":4.3474191E9,"1561896600":6.0725443E9,"1561942200":6.0869704E9,"1561908600":4.03533338E9,"1561921800":5.9515044E9,"1561881000":4.8309176E9,"1561885200":5.4829824E9,"1561946400":1.48851507E10,"1561904400":4.6488059E9,"1561875000":4.6899021E9,"1561957140":8.3945047E9,"1561932000":3.85433523E9,"1561936200":4.3083776E9,"1561917000":5.080833E9,"1561894200":4.9764168E9,"1561898400":4.5028285E9,"1561879200":5.8956759E9,"1561903800":6.2376724E9,"1561888200":4.5363456E9,"1561945200":1.39520707E10,"1561891200":5.6361405E9,"1561926000":5.4089774E9,"1561941000":4.5894738E9,"1561884000":4.5772462E9,"1561876200":4.5364209E9,"1561937400":3.85036211E9,"1561912800":5.20975E9,"1561893000":4.5637059E9,"1561897200":5.7151227E9,"1561933200":4.3959757E9,"1561952940":1.46748672E10,"1561902600":4.525825E9,"1561887000":4.6441011E9,"1561923000":5.4042327E9,"1561890000":4.6868777E9,"1561906800":4.11002138E9,"1561944000":1.20529613E10,"1561927200":4.6994734E9,"1561946940":1.54137979E10,"1561938600":3.92263757E9}'
    # time_start, time_end = getAbruptChangeRange(json_str)
    # print(time_start)
    # print(time_end)
    # # -------------------------------------------------------------------------------