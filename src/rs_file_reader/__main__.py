import sys, os
sys.path.append( os.path.join(os.getcwd(), '../'))


from rs_file_reader.rs_plot_gui import plot_gui

if __name__ == '__main__':
    plot_gui()