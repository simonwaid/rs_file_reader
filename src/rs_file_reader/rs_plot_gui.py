'''
    File name: gui_plot.py
    Author: Simon Waid
    Date created: 2021
    Python Version: 3.8, 3.9
'''


from scipy.fft import rfft, fft, fftfreq, next_fast_len
import sys
import numpy as np
import time
from PySide2 import QtWidgets
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PySide2.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow, QWidget, QApplication, QFileDialog, QGroupBox, QLabel, QLineEdit, QMessageBox, QComboBox, QSpacerItem, QSizePolicy
from PySide2 import QtCore
from rs_file import RS_File
from rs_analysis import RS_Analysis
matplotlib.use("Qt5Agg")  # Declare the use of QT5
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#from config import CACHE_DIR
import os 
from tool_box import pqt5_exception_workaround, Persival


class PlotWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(PlotWindow, self).__init__(*args, **kwargs)

        #self.toolbar=NavigationToolbar(self.canvas, self)
        
        self.layout = QVBoxLayout()
        #self.layout.addWidget(self.toolbar)
        #self.layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.figure=Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.canvas.figure.subplots()
        self.plot=None
        self.layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas, self))
    
        self.show()


    def update_histo(self, rs_analysis, source):

        print('Generating histogram...')

        histogram, extent=rs_analysis.get_2d_histo(source)#, no_bins_x=100000)
        
        print('Plotting... ')        
        
        dx=extent[1]-extent[0]
        dy=extent[3]-extent[2]
        
        #shape=histogram.shape
        aspect=dx/dy*0.5
        print(aspect)
        #aspect=0.5
        histo_t=np.flip(histogram, 1).transpose()
        histo_t_log=np.log10(histo_t)
        histo_t_log[histo_t==0]=0
        
        t=time.time()
        if self.plot is None:
            self.plot=self.ax.imshow(histo_t_log, aspect=aspect, cmap='gist_stern',  extent=extent)
        else:
            self.plot.set_data(histo_t)
            self.plot.set_extent(extent)
            self.ax.set_aspect(aspect)
            
        print('Plotting column: ', time.time() - t)
        t = time.time()
        self.figure.tight_layout()
        self.canvas.draw()
        self.show()
        print('Draw canvas: ', time.time() - t)
        print('Plotting complete.\n\n' )

    def update_td_plot(self, rs_file, source, start, stop):
        
        start_idx=rs_file.timeToIndex(start)
        stop_idx=rs_file.timeToIndex(stop)
        if start_idx is None or stop_idx is None:
            print('Time invalid')
            return
        print(f'Loading into memory {start_idx}, {stop_idx} ... ')
        
        plt_data = rs_file.getAsDf(start=start_idx, stop=stop_idx, source=source)
        
        columns=list(plt_data.columns)
        columns.remove('Time')
        print(columns)
        xdata=plt_data['Time']
        print('Length', len(xdata))
        print('Updating plot ... ')
        if self.plot is None:
            self.plot = {}
            for column in columns:   
                ydata=plt_data[column]
                plot=self.ax.plot(xdata, ydata)
                self.plot[column] =plot
        else:
            for column in columns:
                ydata=plt_data[column]
                self.plot[column][0].set_xdata(xdata)
                self.plot[column][0].set_ydata(ydata)

        # Trigger the canvas to update and redraw.
        self.figure.tight_layout()
        self.canvas.draw()
        self.show()
        
    def update_fourier_plot(self, rs_file, source, start, stop):
        '''
        '''


        start_idx=rs_file.timeToIndex(start)
        stop_idx=rs_file.timeToIndex(stop)
        if start_idx is None or stop_idx is None:
            print('Time invalid')
            return
                
        length=stop_idx-start_idx
        t_sample=rs_file.sample_time
        # Find the appropriate signal length for executing a fast fft.
        fft_len=next_fast_len(length)
        
        print(f'Will use {fft_len} samples for fft.')
        # Get the data
        print(f'Loading data into memory. Source: {source}')
        data=rs_file.getAsDf(start=start_idx, stop=start_idx+fft_len, source=source, time=False) 
        data=data[source].values
        
        # Compute fft.
        # Note: We keep fft_len as 
        print('Computing fft')
        N=fft_len
        yf=rfft(data, fft_len)
        print('Computing frequencies')
        xf = fftfreq(fft_len, t_sample)[:N//2]
        print('Plotting')
        
        x=xf[1:N//2]
        y=2.0/N * np.abs(yf[1:N//2])
        
        if self.plot is None:
            self.plot=self.ax.loglog(x, y)
        else:
            self.plot[0].set_xdata(x)
            self.plot[0].set_ydata(y)
        self.figure.tight_layout()            
        self.canvas.draw()
        self.show()
        
#        plt.loglog(xf[1:N//2], 2, '-b')
        
 #       plt.legend(['FFT', 'FFT w. window'])
        
  #      plt.grid()
        
   #     plt.show()

class ApplicationWindow(QMainWindow):
    def __init__(self, cacheDir=None):
        super().__init__()
        self._main = QWidget()
        self.cacheDir=cacheDir
        self.setCentralWidget(self._main)
        top_layout = QVBoxLayout(self._main) 
        
        main_layout = QHBoxLayout()
        box_selected_file=self._add_selected_file()
        top_layout.addWidget(box_selected_file)
        
        top_layout.addLayout(main_layout)

        group_data_selection=self._add_file_channel()
        group_histogram_plot=self._add_histogram_plot()
        group_td_plot=self._add_detail_plot()
        group_fourier_plot=self._add_fourier_plot()
        
        main_layout.addWidget(group_data_selection)
        main_layout.addWidget(group_histogram_plot)
        main_layout.addWidget(group_td_plot)
        main_layout.addWidget(group_fourier_plot)

        self.rs_file=None

        # Remembers settings between sessions
        self.persival=Persival('gui_plot.json')
        pwd = os.getcwd()
        self.persival.setDefault('directory', pwd)
        self.histogram_plot_window=None
        self.td_plot_window=None
        self.fourier_plot_window=None

    def _add_selected_file(self):
        
        box_selected_file=QGroupBox("Selected file") 
        layout_selected_file=QVBoxLayout()
        box_selected_file.setLayout(layout_selected_file)
        self.label_selected_file=QLabel('No file selected')
        layout_selected_file.addWidget(self.label_selected_file)

        return box_selected_file

    def _add_file_channel(self):
        '''
        '''
        box_data_selection=QGroupBox("Data selection") 
        layout_data_selection=QVBoxLayout()
        box_data_selection.setLayout(layout_data_selection)
        
        self.open_file_button=QPushButton("Open file")
        #self.open_file_button.setFixedSize(150, 40)
        layout_data_selection.addWidget(self.open_file_button)
        self.open_file_button.clicked.connect(self.open_file)

        label_selected_channel=QLabel("Channel:") 
        layout_data_selection.addWidget(label_selected_channel)
        self.cbox_trace=QComboBox()
        layout_data_selection.addWidget(self.cbox_trace)

        self.cbox_trace.currentIndexChanged.connect(self.update_source)

        vertical_spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout_data_selection.addItem(vertical_spacer)

        return box_data_selection

    def _add_histogram_plot(self):
        # Detail plot        
        box_fourier_plot=QGroupBox("Histogram plot") 
        layout_fourier_plot=QVBoxLayout()
        box_fourier_plot.setLayout(layout_fourier_plot)
        
        vertical_spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout_fourier_plot.addItem(vertical_spacer)
        
        self.button_historgram_plot=QPushButton("Update Histogram plot")
        layout_fourier_plot.addWidget(self.button_historgram_plot)
        self.button_historgram_plot.clicked.connect(self.open_histogram)
        
       
        return box_fourier_plot

    def _add_fourier_plot(self):
        # Detail plot        
        box_fourier_plot=QGroupBox("Fourier plot") 
        layout_fourier_plot=QVBoxLayout()
        box_fourier_plot.setLayout(layout_fourier_plot)
        
        label_fourier_plot_start=QLabel('Start time / s')
        layout_fourier_plot.addWidget(label_fourier_plot_start)
        
        self.edit_fourier_plot_start=QLineEdit('1')
        layout_fourier_plot.addWidget(self.edit_fourier_plot_start)
        
        label_fourier_plot_end=QLabel('End time / s')
        layout_fourier_plot.addWidget(label_fourier_plot_end)
        
        self.edit_fourier_plot_end=QLineEdit('1.1')
        layout_fourier_plot.addWidget(self.edit_fourier_plot_end)
        
        self.button_rd_plot=QPushButton("Update Fourier plot")
        layout_fourier_plot.addWidget(self.button_rd_plot)
        self.button_rd_plot.clicked.connect(self.open_fourier_plot)
        
        return box_fourier_plot

    def _add_detail_plot(self):
        # Detail plot        
        box_td_plot=QGroupBox("Time domain plot") 
        layout_td_plot=QVBoxLayout()
        box_td_plot.setLayout(layout_td_plot)
        
        label_td_plot_start=QLabel('Start time / s')
        layout_td_plot.addWidget(label_td_plot_start)
        
        self.edit_td_plot_start=QLineEdit('1')
        layout_td_plot.addWidget(self.edit_td_plot_start)
        
        label_td_plot_end=QLabel('End time / s')
        layout_td_plot.addWidget(label_td_plot_end)
        
        self.edit_td_plot_end=QLineEdit('1.1')
        layout_td_plot.addWidget(self.edit_td_plot_end)
        
        self.button_rd_plot=QPushButton("Update TD plot")
        layout_td_plot.addWidget(self.button_rd_plot)
        self.button_rd_plot.clicked.connect(self.open_td_plot)
        
        return box_td_plot
    
    def open_fourier_plot(self):
        '''
        '''
        print('Fourier plot')
    
        try: 
            start = float(self.edit_fourier_plot_start.text())
        except ValueError:
            #fsAcquisitions=None
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Not a number")
            dlg.setText("The start time is invalid.")
            button = dlg.exec()
            return None
        
        try: 
            end = float(self.edit_fourier_plot_end.text())
        except ValueError:
            #fsAcquisitions=None
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Not a number")
            dlg.setText("The end time is invalid.")
            button = dlg.exec()
            return None
        
        print('TD plot')
        if self.fourier_plot_window is None:
            self.fourier_plot_window=PlotWindow()
        self.fourier_plot_window.update_fourier_plot(self.rs_file, self.source,  start, end)
    
    def open_td_plot(self):
        '''
        '''
        
        try: 
            start = float(self.edit_td_plot_start.text())
        except ValueError:
            #fsAcquisitions=None
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Not a number")
            dlg.setText("The start time is invalid.")
            button = dlg.exec()
            return None
        
        try: 
            end = float(self.edit_td_plot_end.text())
        except ValueError:
            #fsAcquisitions=None
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Not a number")
            dlg.setText("The end time is invalid.")
            button = dlg.exec()
            return None
        
        print('TD plot')
        if self.td_plot_window is None:
            self.td_plot_window=PlotWindow()
        self.td_plot_window.update_td_plot(self.rs_file, self.source, start, end)
        
    def open_histogram(self):
        '''
        '''
        if self.histogram_plot_window is None:
            self.histogram_plot_window=PlotWindow()
        self.histogram_plot_window.update_histo(self.rs_analysis, self.source)
        
    def open_file(self):
        '''
        Opens an oscilloscope file.
        '''
        
        directory=self.persival.get('directory')
        #options= QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open RS file", directory,"RS RTP file (*.bin *.csv);; All Files (*)" )
        #Ignore empty selections
        if fileName== '':
            return
        # Save the directory for future use.
        directory=os.path.dirname(fileName)
        self.persival.set('directory', directory)
        # Show the file name in the GUI so the user is aware of what he did.
        self.label_selected_file.setText(fileName)
        
        print(f'File: {fileName}')
        self.rs_file=RS_File(fileName)
        self.rs_analysis=RS_Analysis(self.rs_file, self.cacheDir)
        self.cbox_trace.clear()
        self.cbox_trace.addItems(self.rs_file.signal_sources)
        self.update_source()
        
    def update_source(self):
        
        self.source=self.cbox_trace.currentText()
    
    
    def analyze(self):
        '''
        Analyzes a loaded file
        ''' 
        #Does nothing if no file is loaded
        if self.rs_file is None:
            return

        start, stop = self.rs_file._getLimitedStartStop()
        t = time.time()
        statFloat = self.rs_file.getStatistics(start, stop)
        print('Statistics calculation took {:.2f}s '.format(time.time()-t))
        print("Statistics:", statFloat)
        basedir, filename = os.path.split(self.rs_file.meta['metadata_file'])
        file_base, _ = os.path.splitext(filename)
        target_dir = os.path.join(basedir, file_base + '_analysis')
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        target_dir = os.path.join(target_dir, 'events')
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for ch, v in statFloat.items():
            
            # Analyze the events and get the Tot
            t = time.time()
            events, meta = self.rs_file.getTotAuto()
            print('Meta: ', meta)
            print('Searching for events took {:.2f}s '.format(time.time()-t))
            if len(events) == 0:
                print('No events found')
            else:
                t = time.time()
                print('We have {} events'.format(len(events)))
            
                # Plot some of them for analysis
                self.rs_file.plotEvents(events, os.path.join(target_dir, ch), ch)
                print('Plotting events took {:.2f}s\n'.format(time.time()-t))
    
    def closeEvent(self, event):
        '''
        Before closing the application we need to perform some clean up activities.
        ''' 
        self.persival.save()
        print('Persival saving done!')
        event.accept()   

def plot_gui():
    pqt5_exception_workaround()
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)
    cacheDir=None
    app = ApplicationWindow(cacheDir=cacheDir)
    app.show()
    #app.activateWindow()
    #app.raise_()
    qapp.exec_()
    
# Opens the gui for plotting and analyzing
if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    plot_gui()