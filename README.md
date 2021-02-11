# Lab-Cycle-1
E3 Experiment Numerical Analysis Code


Four Files have been included, One for each stage of the experiment. 


1)Calibration:
        Plots Gaussians over raw data from csv file. Make sure to have the csv file in the same folder so it can find the data.
        This program is difficult to use as it requires modifying for everytime a new Gaussian is plotted, however the results have been directly typed into the             the next file.


2) linear_interpolation:
        This is used to calculate the energy per bin or gradient. 
        A separate calibration file has been included that plots Gaussians over each raw set of data and these means are typed into linear_interpolation directly.

3) DataPlotting2:
        This file imports the gradient from linear_interpolation and imports the csv file with all the raw data.
        It has 6 Callable Functions: PlotData() PlotAll() PlotGaussiansOnly() PlotAngles1() PlotAngles2() PlotCompton()
        The smoothing paramter h can be changed to visualise the data better 
        To plot the the Compton Equation on top of the scattering energy data use PlotAngles2 and PlotCompton at the same time
 
 
 4) MCSimulation:
        This file imports DataPlotting2.
        Run each cell separately to run the simulation. Cell 1 will plot a simulation. Cell 2 will run the simulation over and over 500 times, Cell 3 plots these           points and fits a function
 
