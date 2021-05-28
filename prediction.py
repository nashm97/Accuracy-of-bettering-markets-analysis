import numpy
import random
import math


class Prediction:
    """Some event with a binary outcome that we have a predicted odds for.

    Attributes:
        odds (float): Number between 0-1 indicating predicted odds as a frequency or probability.
        outcome (bool): True if event occurred, false if it did not.

    """

    def __init__(self, odds, outcome):
        self.odds = odds
        self.outcome = outcome


class Bucket:
    """Holds a number of predictions that fit in the given start and end values.

    Note:
        We use the term "bucket" instead of "bin", but they are analogous. This is partly
        due to the fact that bin() is already a built-in function in python.

    Attributes:
        start_val (float): Minimum odds value.
        end_val (float): Maximum odds value.
        predictions (List[Prediction]): Predictions in this bucket.

    """

    def __init__(self, start_val, end_val):
        self.start_val = start_val
        self.end_val = end_val
        self.predictions = []

    def safe_add_predictions(self, predictions, start_inclusive=True, end_inclusive=False):
        """Adds only predictions with odds between the start_val and end_val.

        Note:
            Predictions with odds outside of the range of the bucket will simply not be added.

        Args:
            predictions (List[Prediction]): The predictions to attempt to add to the bucket.
            start_inclusive (Optional[bool]): Whether to allow predictions with the same odds as start_val.
                Defaults to True.
            end_inclusive (Optional[bool]); Whether to allow predictions with the same odds as end_val.
                Defaults to False.

        """
        self.predictions.extend(
            [p for p in predictions if self.start_val < p.odds < self.end_val]
        )

        if start_inclusive:
            self.predictions.extend(
                [p for p in predictions if p.odds == self.start_val]
            )

        if end_inclusive:
            self.predictions.extend(
                [p for p in predictions if p.odds == self.end_val]
            )


    def get_actual_outcome_frequency(self):
        """Calculates what portion of the predictions in this bucket had True outcomes.

        Returns:
            float: Frequency of True outcomes. (between 0 and 1)

            Will return None if there are no predictions in this bucket.

        """
        if len(self.predictions) == 0:
            return None

       # outcomes = [p.outcome for p in self.predictions]
        return float(self.get_actual_outcome_number())/len(self.predictions)


    def get_actual_outcome_number(self):
        """calculates the NUMBER of predictions in a bucket that had true outcomes.

           Returns:
              float: the number of actual true outcomes

              will return None if there are no predictions in this nucket.
        """
        if len(self.predictions) == 0:
            return None

        outcomes = [p.outcome for p in self.predictions]
        return float(outcomes.count(True))


    def get_mid_point(self):
        """calculates the midpoint value of the start value and end value of a bucket.

           Args:
              start_val: start value of bucket.
              end _val: end value of the bucket.
           Returns:
               mid_point_val: the value of the midpoint.
        """

        return float((self.start_val + self.end_val)/2)


    def calc_error_for_bucket(self):
        """claculates the error on each bucket using E = Za/2(sqr((p'hat*q'hat)/n) where Za/2 =1, p'hat = self.get_actual_outcome_frequency()/(len(self.predictions)),
           q'hat = 1- p'hat

           Returns: Error value.

           Args: self.get_actual_outcome_frequency : frequency of a true outcome for this bucket.
                 len(self.predictions) : number of predictions actually made in a bucket.
        """
        x = self.get_actual_outcome_frequency()

        y = (1-x)

        return float(2*(math.sqrt((x*y)/(len(self.predictions)))))

       # return float( math.sqrt(((self.get_actual_outcome_frequency/(len(self.predictions)))*(1 - (self.get_actual_outcome_frequency/(len(self.predictions)))))/(len(self.predictions)))

    def get_upper_error_val(self):
        """calculates the upper error bar for each predicted value.

           Returns: upper_error_val: upper value for the error bars.

        """
        return float((self.get_actual_outcome_frequency() + self.calc_error_for_bucket()))


    def get_lower_error_val(self):
        """calculates the lower error bar for each predicted value.

           Returns: lower_error_val: lower value for the error bars.

        """
        return float((self.get_actual_outcome_frequency() - self.calc_error_for_bucket()))

    def __str__(self):

        #return 'Bucket(%f, %f, %d, %d, %f, %f)' % (self.start_val, self.end_val, (len(self.predictions)), self.get_actual_outcome_number(), self.get_actual_outcome_frequency(), self.calc_error_for_bucket())
        #return '%f  \n' % (self.get_actual_outcome_frequency())
        #b return '%.9f  \n' % (self.get_lower_error_val())
        #return '%.9f  \n' % (self.get_upper_error_val())
        return '%.11s' % (self.is_num_in_std_dev())


    def __repr__(self):
        return self.__str__()


    def get_prediction_samples(self, num_samples, sample_size):
        """Gets a list of a number of random samples of the predictions in this bucket.

        Args:
            num_samples (int); The number of samples to use.
            sample_size (int): The size of each samples.

        Returns:
            List[List[Prediction]]: The mean probabilities of each of the samples.

        """
        return [random.sample(self.predictions, sample_size) for _ in range(num_samples)]

    def calc_residual_value(self):
        """calculates the residual values for the predicted values

        args: get_mid_points()  get_actual_outcome_frequency()
        return: residual value for each bucket
        """
        y = self.get_actual_outcome_frequency()

        x = self.get_mid_point()

        r = y - x
        return r

    def is_val_in_num_std_dev(self):
        """gives true if a residual is within the number 
        of standaerd deviations specified in the error barsand false if not"""
        if self.calc_residual_value() < 0:
            r = self.calc_residual_value() + self.calc_error_for_bucket()
            return r > 0
        else:
            z = self.calc_residual_value() - self.calc_error_for_bucket()
            return z < 0

    def get_frequency_distribution(self, num_samples, sample_size):
        """Determines a frequency distribution for the predictions in this bucket.

        First we get a number of random samples of the predictions. Then, for each of
        those samples, we find their outcome frequency. We can then find the distribution
        and standard deviation of these outcome frequencies.

        Args:
            num_samples (int); The number of samples to use.
            sample_size (int): The size of each sample.

        Returns:
            List[int]: The frequency distribution for odds of 0-1.
            float: The standard deviation of the distribution.

            """
        samples = self.get_prediction_samples(num_samples, sample_size)

        samples_num_trues = [
            [prediction.outcome for prediction in sample].count(True)
            for sample in samples
        ]

        frequencies = [samples_num_trues.count(i) for i in range(sample_size)]
        standard_deviation = numpy.std([[float(i)/sample_size for i in samples_num_trues]])

        return frequencies, standard_deviation


    def range_contains(self, frequency):
        """Test if the given frequency is within start_val and end_val for this bucket.

        Args:
            frequency (float): The frequency to test. Between 0 and 1.

        Returns:
            bool: True if the given frequency is within the bucket's range, and False otherwise.

        """
        return self.start_val <= frequency <= self.end_val


def calculate_outcome_frequencies(buckets):
    """ Calculates the predicted and actual True outcome frequencies for all buckets.

    Args:
        buckets (List[Bucket]): All buckets to perform calculations on.

    Returns:
        List[float]: The predicted outcome frequency.
        List[float]: The actual outcome frequency.

    """
    predicted = [(bucket.start_val + bucket.end_val) / 2.0 for bucket in buckets]
    actual = [bucket.get_actual_outcome_frequency() for bucket in buckets]

    # Remove empty buckets from results
    predicted, actual = (
        zip(*filter(lambda x: x[1] is not None, zip(predicted, actual)))
    )

    return predicted, actual


def sort_predictions_into_buckets(predictions, num_buckets):
    """Sorts a list of predictions by odds into their appropriate buckets.

    Note:
        The start_val of the first bucket is 0 and the end_val of the last bucket is 1.
        The bucket size is determined by the number of buckets. All buckets are the same size.

    Args:
        predictions (List[Prediction]): The predictions to be sorted into buckets.
        num_buckets (int); The number of buckets.

    Returns:
        List[Bucket]: All buckets, with the first bucket having a start_val of 0 and the last
            bucket having an end_val of 1.

    """
    buckets = []
    bucket_vals = numpy.linspace(0, 1, num_buckets + 1)

    for i in range(0, num_buckets - 1):
        bucket = Bucket(bucket_vals[i], bucket_vals[i + 1])
        bucket.safe_add_predictions(predictions)
        buckets.append(bucket)

    # Include end_val for last bucket
    bucket = Bucket(bucket_vals[num_buckets - 1], bucket_vals[num_buckets])
    bucket.safe_add_predictions(predictions, end_inclusive=True)
    buckets.append(bucket)

    return buckets

def count_num_in_std_dev(buckets):
    """counts how many residuals are within the specified 
    number of standard deviations"""
    s = []
    for n in buckets:
        w = n.is_val_in_num_std_dev()
        s.append(w)
    return s.count(True)

def get_mid_points(buckets):
   """calculates the midpoints of each bucket from get_mid_pointand make it
    accessible to plotter.py"""
   m = []
   for x in buckets:
       y = x.get_mid_point()
       m.append(y)
   return m

def calc_residual_values(buckets):
    """ takes the resdiaul values and makes then accessible by plotter.py"""
    r = []
    for x in buckets:
        c = x.calc_residual_value()
        r.append(c)
    return r

def calc_error_for_buckets(buckets):
    """takes the error values and makes then accessible by plotter.py"""
    z = []
    for x in buckets:
        y = x.calc_error_for_bucket()
        z.append(y)
    return z

def find_largest_acceptable_num_buckets(predictions):
    """Finds the largest acceptable number of buckets.

    This indirectly determines the smallest acceptable bucket size.

    The number of buckets is acceptable if no actual outcome frequency of the predictions
    in that bucket fall outside of the range of that bucket.

    Args:
        predictions (List[Prediction]): All of the predictions.

    Returns:
        int: The largest acceptable number of buckets.

    """
    num_buckets = 1
    buckets = sort_predictions_into_buckets(predictions, num_buckets)

    while all([b.range_contains(b.get_actual_outcome_frequency()) for b in buckets]):
        num_buckets += 1
        buckets = sort_predictions_into_buckets(predictions, num_buckets)

    return num_buckets - 1
