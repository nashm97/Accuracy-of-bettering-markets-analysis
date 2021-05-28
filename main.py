
import sys
import plotter as pred_plotter
import csv
import prediction as pred
import prediction as p

def main():
    """
    Main function called to run prediction analysis. Requires a number of parameters
    in the following format:
        filename start_row odds_col outcomes_col outcomes_str num_buckets num_samples sample_size

    Description of parameters:
        filename (str): The name/path of the csv file to get data from.
        start_row (int): The first row of the csv file to start getting data from.
        odds_col (int): The column in the csv file with the odds/probabilities.
            Everything in this column should be a float between 0 and 1.
        outcomes_col (int): The column in the csv file with the outcome of the event.
        outcomes_str (str): The string to look for in the outcomes column that means that
            the outcome was true, or that the event occurred, etc. Usually "TRUE".
        num_buckets (int): The number of buckets to sort predictions into. Can alternatively be
            "auto", in which case a number will be automatically determined.
        num_samples (int): The number of samples per bucket to be used for frequency distributions.
        sample_size (int): The size of each sample used for frequency distributions.

    """

    # Initial arguments
    filename = sys.argv[1]
    start_row = int(sys.argv[2])
    odds_col = int(sys.argv[3])
    outcomes_col = int(sys.argv[4])
    outcomes_str = sys.argv[5]
    num_buckets = sys.argv[6]
    num_samples = int(sys.argv[7])
    sample_size = int(sys.argv[8])

    # Open the csv file
    csvfile = open(filename)

    # Get the predictions
    predictions = get_predictions_from_csv(csvfile, start_row, odds_col, outcomes_col, outcomes_str)

    # If auto parameter, automatically get the number of buckets
    if num_buckets == 'auto':
        num_buckets = pred.find_largest_acceptable_num_buckets(predictions)
    else:
        num_buckets = int(num_buckets)

    # Sort the predictions into the given number of buckets
    buckets = pred.sort_predictions_into_buckets(predictions, num_buckets)

    # Plot the predictions
    plotter = pred_plotter.Plotter(buckets, num_samples, sample_size)
    plotter.plot_predictions()
    #print p.count_num_in_std_dev(buckets)

def get_predictions_from_csv(csvfile, start_row, odds_col, outcomes_col, outcome_str):
    """Gets a list of predictions from a csv file.

    Note:
        Row/column numbers start at 1.

    Args:
        csvf\ ile (str): The name/path of the csv file.
        start_row (int): The row number of first row to start getting predictions from.
        odds_col (int): The column number of the column containing the prediction odds.
        outcomes_col (int): The column number of the column containing the prediction outcomes.
        outcome_str (str): The string in the outcomes column that should be parsed as True.
            Any other string will be parsed as False.

    Returns:
        List[Predictions]: The predictions obtained from the csv file.

    """
    reader = csv.reader(csvfile)

    while start_row > 1:
        reader.next()
        start_row -= 1

    return [
        pred.Prediction(float(row[odds_col - 1]), row[outcomes_col - 1] == outcome_str)
        for row in reader
        if is_number(row[odds_col - 1])
    ]


def is_number(s):
    """Checks if a string can be cast to a float.

    Args;
        s (str): The string to check.

    Returns:
        bool: True if cast was successful, False if exception was caught.

    """
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
