import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from model import *
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d


def main(model):

    root = tk.Tk()
    root.title("GHG Emission Predictor")
    root.geometry('800x700')
    root.resizable(False, False)
    nb = ttk.Notebook(root)

    style = ttk.Style()
    style.theme_create('Cloud', settings={
        ".": {
            "configure": {
                "background": root.cget('bg'),
            }
        },
        "TNotebook": {
            "configure": {
                "tabposition": 'n'
             }
        },
        "TNotebook.Tab": {
            "configure": {
                "background": 'light blue',
                "padding": [20, 20],
                "font": "white"
            }
        }})
    style.theme_use('Cloud')

    page1 = ttk.Frame(nb)
    layout1(page1, model)
    page2 = ttk.Frame(nb)
    layout2(page2, model)

    nb.add(page1, text='Gaussian Process Regression')
    nb.add(page2, text='    Quantile Regression    ')
    # nb.add(page3, text='page3')

    nb.pack(fill=BOTH, expand=1)
    root.mainloop()


def layout1(root, model):

    # column 0
    tk.Label(root, text='Revenue (mm) : ').grid(row=0, column=0)
    tk.Label(root, text='Market Cap (mm) : ').grid(row=1, column=0)
    tk.Label(root, text='Total Asset (mm) : ').grid(row=2, column=0)
    tk.Label(root, text='BICS Level 4 : ').grid(row=3, column=0)
    tk.Label(root, text='Prior Year\'s GHG Emission (mt) : ').grid(row=4, column=0)
    tk.Label(root, text='# of employees (Optional) : ').grid(row=5, column=0)
    tk.Label(root, text='Confidence Interval : ').grid(row=6, column=0)

    # column 1
    revenue_ = tk.Entry(root, width=20)
    revenue_.grid(row=0, column=1)
    market_cap_ = tk.Entry(root, width=20)
    market_cap_.grid(row=1, column=1)
    total_asset_ = tk.Entry(root, width=20)
    total_asset_.grid(row=2, column=1)
    prev_ghg_ = tk.Entry(root, width=20)
    prev_ghg_.grid(row=4, column=1)
    bics_list_ = [_tmp[0] for _tmp in model.bics_list_.values]
    variable = tk.StringVar(root)
    variable.set(bics_list_[0])
    bics_4_ = tk.OptionMenu(root, variable, *bics_list_)
    bics_4_.grid(row=3, column=1)
    n_employees_ = tk.Entry(root, width=20)
    n_employees_.grid(row=5, column=1)
    ci_ = tk.Entry(root, width=20)
    ci_.insert(0, "95")
    ci_.grid(row=6, column=1)

    # column 2
    tk.Label(root, text='          OR          ').grid(row=0, column=2, rowspan=2)

    # column 3
    tk.Label(root, text='  Ticker :   ').grid(row=0, column=3)
    tk.Label(root, text='  Year :   ').grid(row=1, column=3)

    # column 4
    ticker_ = tk.Entry(root, width=20)
    ticker_.grid(row=0, column=4)
    year_ = tk.Entry(root, width=20)
    year_.grid(row=1, column=4)

    # plot
    def error_popup():
        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        win_x = root_x + 300
        win_y = root_y + 200
        top = tk.Toplevel(root)
        top.geometry("300x50")
        top.geometry(f'+{win_x}+{win_y}')
        top.title("Error")
        tk.Label(top, text="No Data Found for the Given TICKER + YEAR").pack(fill="none", expand=True)

    rv = tk.IntVar()
    r1 = tk.Radiobutton(root, text='Chart', variable=rv, value=0)
    r1.grid(row=3, column=3)
    r2 = tk.Radiobutton(root, text='Table', variable=rv, value=1)
    r2.grid(row=4, column=3)

    def click():

        if (revenue_.get() == "" or market_cap_.get() == "") and (ticker_.get() == "" or year_.get() == ""):
            error_popup()
        elif rv.get() == 0:
            fig = Figure(figsize=(6, 4), dpi=100)
            plot1 = fig.add_subplot(111)

            if ticker_.get():
                tmp = model.df_[(model.df_.index == ticker_.get() + " US Equity") & (model.df_['Year'] == int(year_.get()))]
                if len(tmp) == 0:
                    error_popup()
                else:
                    model.predict(tmp.iloc[:, :4].values[0], y_true=tmp.iloc[:, -3][0], ci=float(ci_.get())/100, ax=plot1)
                    canvas = FigureCanvasTkAgg(fig, master=root)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=7, column=0, columnspan=5)
            else:
                feed = [float(revenue_.get()), float(market_cap_.get()), variable.get(), float(prev_ghg_.get())]
                model.predict(feed, y_true=None, ci=float(ci_.get())/100, ax=plot1)
                canvas = FigureCanvasTkAgg(fig, master=root)
                canvas.draw()
                canvas.get_tk_widget().grid(row=7, column=0, columnspan=5)
        else:

            txt = Text(root, background="light yellow", height=30, width=86)
            txt.tag_configure("center", justify='center')
            txt.insert("1.0", "  \n")
            txt.tag_add("center", "1.0", "end")
            txt.grid(row=7, column=0, columnspan=5)

            class PrintToTXT(object):
                def write(self, s):
                    txt.insert(END, s)

            sys.stdout = PrintToTXT()

            def _print(array, f):
                for a in array:
                    print(f.format(a), end=" ")
                print("")

            if ticker_.get():
                tmp = model.df_[(model.df_.index == ticker_.get() + " US Equity") & (model.df_['Year'] == int(year_.get()))]
                if len(tmp) == 0:
                    error_popup()
                else:
                    res = model.predict(tmp.iloc[:, :4].values[0], plot=False)
                    _print(np.arange(0.1, 1, 0.1), f="{:8.0%}")
                    _print(res, f="{:8.2f}")

            else:
                feed = [float(revenue_.get()), float(market_cap_.get()), variable.get(), float(prev_ghg_.get())]
                res = model.predict(feed, y_true=None, plot=False)
                _print(np.arange(0.1, 1, 0.1), f="{:8.0%}")
                _print(res, f="{:8.2f}")

    tk.Button(root, text='PLOT', command=click, height=2, width=6, highlightbackground="black", fg='black').grid(
        row=3, column=4, rowspan=2, columnspan=1)


def layout2(root, model):
    # column 0
    tk.Label(root, text='Revenue (mm) : ').grid(row=0, column=0)
    tk.Label(root, text='BICS Level 4 : ').grid(row=1, column=0)
    tk.Label(root, text='Prior Year\'s GHG Emission (mt) : ').grid(row=2, column=0)
    tk.Label(root, text='Bandwidth : ').grid(row=3, column=0)

    # column 1
    revenue_ = tk.Entry(root, width=20)
    revenue_.grid(row=0, column=1)
    prev_ghg_ = tk.Entry(root, width=20)
    prev_ghg_.grid(row=2, column=1)
    bics_list_ = [_tmp[0] for _tmp in model.bics_list_.values]
    variable = tk.StringVar(root)
    variable.set(bics_list_[0])
    bics_4_ = tk.OptionMenu(root, variable, *bics_list_)
    bics_4_.grid(row=1, column=1)
    bandwidth_ = tk.Entry(root, width=20)
    bandwidth_.grid(row=3, column=1)

    # column 2
    tk.Label(root, text='          OR          ').grid(row=0, column=2, rowspan=2)

    # column 3
    tk.Label(root, text='  Ticker :   ').grid(row=0, column=3)
    tk.Label(root, text='  Year :   ').grid(row=1, column=3)

    # column 4
    ticker_ = tk.Entry(root, width=20)
    ticker_.grid(row=0, column=4)
    year_ = tk.Entry(root, width=20)
    year_.grid(row=1, column=4)

    # plot
    def error_popup():
        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        win_x = root_x + 300
        win_y = root_y + 200
        top = tk.Toplevel(root)
        top.geometry("300x50")
        top.geometry(f'+{win_x}+{win_y}')
        top.title("Error")
        tk.Label(top, text="No Data Found for the Given TICKER + YEAR").pack(fill="none", expand=True)

    rv = tk.IntVar()
    r1 = tk.Radiobutton(root, text='Chart', variable=rv, value=0)
    r1.grid(row=3, column=3)
    # r2 = tk.Radiobutton(root, text='Table', variable=rv, value=1)
    # r2.grid(row=4, column=3)

    def click():

        if revenue_.get() == "" and (ticker_.get() == "" or year_.get() == ""):
            error_popup()
        elif rv.get() == 0:
            fig = Figure(figsize=(6, 4), dpi=100)
            plot1 = fig.add_subplot(111)
            bw = float(bandwidth_.get()) if bandwidth_.get() else None

            if ticker_.get():
                tmp = model.df_[(model.df_.index == ticker_.get() + " US Equity") & (model.df_['Year'] == int(year_.get()))]
                if len(tmp) == 0:
                    error_popup()
                else:
                    model.predict_2(tmp.iloc[:, :4].values[0], y_true=tmp.iloc[:, -3][0], ax=plot1, bw=bw)
                    canvas = FigureCanvasTkAgg(fig, master=root)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=7, column=0, columnspan=5)

            else:
                feed = [revenue_.get(), 0, variable.get(), prev_ghg_.get()]
                model.predict_2(feed, y_true=None, ax=plot1, bw=bw)
                canvas = FigureCanvasTkAgg(fig, master=root)
                canvas.draw()
                canvas.get_tk_widget().grid(row=7, column=0, columnspan=5)

        else:
            txt = Text(root, background="light yellow", height=30, width=86)
            txt.tag_configure("center", justify='center')
            txt.insert("1.0", "  \n")
            txt.tag_add("center", "1.0", "end")
            txt.grid(row=7, column=0, columnspan=5)

            class PrintToTXT(object):
                def write(self, s):
                    txt.insert(END, s)

            sys.stdout = PrintToTXT()

            def _print(array, f):
                for a in array:
                    print(f.format(a), end=" ")
                print("")

            if ticker_.get():
                tmp = model.df_[(model.df_.index == ticker_.get() + " US Equity") & (model.df_['Year'] == int(year_.get()))].values.flatten()[:4]
                if len(tmp) == 0:
                    error_popup()
                else:
                    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    _X = np.array(tmp)
                    _sector = _X[2]
                    _res = []
                    for q in quantiles:
                        _res.append(model.models_2_[_sector, q].predict([[_X[3]]])[0])
                    _res = np.array(_res) / _X[0] * 1000
                    _print(np.arange(0.1, 1, 0.1), f="{:8.0%}")
                    _print(_res, f="{:8.2f}")
            else:
                feed = [revenue_.get(), 0, variable.get(), prev_ghg_.get()]
                quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                _X = np.array(feed)
                _sector = _X[2]
                _res = []
                for q in quantiles:
                    _res.append(model.models_2_[_sector, q].predict([[float(_X[3])]])[0])
                _res = np.array(_res) / float(_X[0]) * 1000
                _print(np.arange(0.1, 1, 0.1), f="{:8.0%}")
                _print(_res, f="{:8.2f}")

    tk.Button(root, text='PLOT', command=click, height=2, width=6, highlightbackground="black", fg='black').grid(
        row=3, column=4, rowspan=2, columnspan=1)


# def layout3(page):
#     pass
