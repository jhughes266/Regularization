import matplotlib.pyplot as plt

#reference: https://www.youtube.com/watch?v=7RgoHTMbp4A provided the neccessary information to build this
#also take away the color for a good experience
class DynamicGrapher:
    def __init__(self, xTitle, yTitle):
        #set up the axis titles
        self._xTitle = xTitle
        self._yTitle = yTitle
        #set up lists to store data to be graphed
        self._timeAxis = []
        self._cost = []
        #keeps track of the time step
        self._t = 0

    def UpdatePlot(self, rawData):

        #append new data into data arrays
        self._timeAxis.append(self._t)
        self._cost.append(rawData)
        #setting the plt axis limits
        if self._t % 50 == 0:
            plt.xlim(0, self._t + 1)
            plt.ylim(0, max(self._cost) + 1)
            # setting titles
            plt.xlabel(self._xTitle)
            plt.ylabel(self._yTitle)
            # plotting the plot
            plt.plot(self._timeAxis, self._cost, color='black')
            # pause the plot
            plt.pause(0.000001)
        #increment the t
        self._t += 1

    #reuired to keep plot on screen once execution of program has finished
    def FinishPlot(self):
        plt.show()