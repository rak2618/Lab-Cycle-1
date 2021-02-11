# Lab-Cycle-1
E3 Experiment Numerical Analysis Code


Three Files have been included, One for each stage of the experiment. 

DataPlotting2:
    This file imports the gradient from linear_interpolation and imports the csv file with all the raw data.
    It has 6 Callable Functions: PlotData() PlotAll() PlotGaussiansOnly() PlotAngles1() PlotAngles2() PlotCompton()
    The smoothing paramter h can be changed to visualise the data better 
    To plot the the Compton Equation on top of the scattering energy data use PlotAngles2 and PlotCompton at the same time
 
 
 MCSimulation:
    This file imports DataPlotting2.
    Run each cell separately to run the simulation. Cell 1 will plot a simulation. Cell 2 will run the simulation over and over 500 times, Cell 3 plots these points     and fits a function
 
