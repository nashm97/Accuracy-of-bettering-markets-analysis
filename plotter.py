import prediction as p
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy


class Plotter:
    """Plots buckets of predictions.

    Note:
        This is NOT a general purpose plotting class. It specifically plots a list
        of the Bucket instances (from prediction.py) and nothing else.

    Attributes:
        buckets (List[Bucket]): The buckets of predictions to plot.
        num_samples (int): The number of samples to use for frequency distributions.
        sample_size (int): The number of predictions per sample.
        fig (Figure): The main plot figure (matplotlib object).
        ax1 (Axis): The axis for the overall graph of all buckets.
        ax2 (Axis): The axis for the residual plot of all buckets.
        standard_deviations (List[float]): The standard deviations of each bucket's distribution.

    """
    def __init__(self, buckets, num_samples, sample_size):
        self.buckets = buckets
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=80)
        self.frequency_distributions = None
        self.standard_deviations = None

    def plot_predictions(self):
        """Plots the predicted vs actual outcome frequencies."""
        predicted, actual = p.calculate_outcome_frequencies(self.buckets)

        # Set titles and axis labels
        self.ax1.set_title("Actual vs predicted true outcome frequency")
        self.ax1.set_xlabel("Predicted frequency")
        self.ax1.set_ylabel("Actual frequency")

        # Set axis properties
        self.ax1.axis([0, 1, 0, 1])
        #self.ax1.set_xticks(predicted)
        #self.ax1.set_yticks(predicted)
        self.ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        self.ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        self.ax1.yaxis.grid(True)

        # Plot predicted vs actual outcome frequencies
        self.ax1.plot(predicted, actual, 'bo')

        # Plot diagonal line
        self.ax1.plot(range(0, 101), range(0, 101), linestyle='dashed', color='black')

        # Plot error bars
        predicted, actual = p.calculate_outcome_frequencies(self.buckets)
        self.ax1.errorbar(predicted, actual, xerr=0, yerr=p.calc_error_for_buckets(self.buckets), fmt='bo')

        # Show the plot
        plt.show(block=False)

        self.__load_error_bars_and_residual_plot()

    def __load_error_bars_and_residual_plot(self):
        """plots residual vs presidcted value"""
       #from prediction import calc_residual_value

        residual = p.calc_residual_values(self.buckets)
        predicted = p.get_mid_points(self.buckets)




        self.ax2.set_title("residual value vs predicted frequency")
        self.ax2.set_xlabel("predicted frequency")
        self.ax2.set_ylabel("residual value")

        #set axis properties
        self.ax2.axis([0, 1, -0.2, 0.2])
        #self.ax2.set_xticks(predicted)
        #self.ax2.set_yticks(predicted)
        self.ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        self.ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.02f'))
        self.ax2.yaxis.grid(True)

        # Plot predicted vs residual value
        self.ax2.plot(predicted, residual, 'bo')


        # Plot error bars
        predicted, actual = p.calculate_outcome_frequencies(self.buckets)
        self.ax2.errorbar(predicted, residual, xerr=0, yerr=p.calc_error_for_buckets(self.buckets), fmt='bo')

        # Show the plot
        plt.show()


