from abc import ABC
from math import pi

import numpy as np
import statsmodels.stats.weightstats as ws
import statsmodels.tsa.stattools as ts
from scipy.stats import ttest_ind


class HypothesisTest(ABC):
    """
    Abstract base class for hypothesis tests.
    """
    alpha = 0.05

    def run(self, name: str, **kwargs):
        """Run test

        Args:
            name (str): Name of test to be run

        Returns:
            (None): 
        """
        self.name = name

        test = getattr(self, name)

        self.stat, self.pval = test(**kwargs)
        self.accept_h0 = (self.pval > self.alpha)


class TwoSampleTest(HypothesisTest):
    def __init__(self, control: np.array, treatment: np.array):
        self.control = control
        self.treatment = treatment

    def ind_t(self, **kwargs):
        """Calculate the T-test for the means of two independent samples of scores.
        The null-hypothesis (H0) is: mean(control) <= mean(treatment)
        The alternative hypothesis (H1) is: mean(control) > mean(treatment)
        This test provides the probability to see the sample's test statistic or 
        something that is even less in line with the null hypothesis (H0) 
        under the H0: P(T<=0|H0), where T is the Welch's_t-test statistic [4, 2].

        Necessary Assumptions [1]:
            * 1. The means of the two populations being compared should follow 
                normal distributions. Under weak assumptions, this follows in large 
                samples from the central limit theorem, even when the distribution 
                of observations in each group is non-normal.[18]

            * 2. If using Student's original definition of the t-test, the two 
                populations being compared should have the same variance 
                (testable using F-test, Levene's test, Bartlett's test, or the 
                Brown–Forsythe test; or assessable graphically using a Q–Q plot). 
                If the sample sizes in the two groups being compared are equal, 
                Student's original t-test is highly robust to the presence of unequal 
                variances. Welch's t-test is insensitive to equality of the 
                variances regardless of whether the sample sizes are similar.

            * 3. The data used to carry out the test should either be sampled 
                independently from the two populations being compared or be fully 
                paired. This is in general not testable from the data, but if the 
                data are known to be dependent (e.g. paired by test design), a 
                dependent test has to be applied. For partially paired data, the 
                classical independent t-tests may give invalid results as the test 
                statistic might not follow a t distribution, while the dependent 
                t-test is sub-optimal as it discards the unpaired data

            Most two-sample t-tests are robust to all but large deviations from the assumptions

        Args:
            control (Iterable): Iterable containing average to be tested
            treatment (Iterable): Iterable containing average to be tested
            pval_threshold (float, optional): p-value threshold. Defaults to 0.05.

        Returns:
            (float, float): Test statistic and p-value

        Example:
            >>> np.random.seed(1) # seed the random number generator
            >>> control = 5 * np.random.randn(100) + 50
            >>> treatment = 5 * np.random.randn(100) + 51
            >>> test = TwoSampleTest(control, treatment)
            >>> test.run("ind_t")
            >>> test.name
            'ind_t'
            >>> test.stat
            -2.2620139704259556
            >>> test.pval
            0.024782819014639627
            >>> test.accept_h0
            False

        References:
            [1] https://en.wikipedia.org/wiki/Student%27s_t-test
            [2] https://stats.stackexchange.com/questions/491294/can-you-multiply-p-values-if-you-perform-the-same-test-multiple-times
            [3] https://stackoverflow.com/questions/15984221/how-to-perform-two-sample-one-tailed-t-test-with-numpy-scipy
            [4] https://en.wikipedia.org/wiki/Welch%27s_t-test
            [5] https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
        """
        return ttest_ind(self.control, self.treatment, **kwargs)

    def ind_z(self, **kwargs):
        """Do a independent z-test to verify if the difference of a mean between treatment and control
        is statistically significant. The need for independent test is because the variance of control
        and treatment are not necessarily equal.

        Necessary Assumptions [3]:
            * Nuisance parameters should be known, or estimated with high accuracy 
            (an example of a nuisance parameter would be the standard deviation in a one-sample location test). 
            Z-tests focus on a single parameter, and treat all other unknown parameters as being fixed at their 
            true values. In practice, due to Slutsky's theorem, "plugging in" consistent estimates of 
            nuisance parameters can be justified. However if the sample size is not large enough for these 
            estimates to be reasonably accurate, the Z-test may not perform well.


            * The test statistic should follow a normal distribution. Generally, one appeals to the central limit 
            theorem to justify assuming that a test statistic varies normally. There is a great deal of 
            statistical research on the question of when a test statistic varies approximately normally. 
            If the variation of the test statistic is strongly non-normal, a Z-test should not be used.

        Args:
            control (Iterable): Iterable containing average to be tested
            treatment (Iterable): Iterable containing average to be tested
            pval_threshold (float, optional): p-value threshold. Defaults to 0.05.

        Returns:
            (float, float): Test statistic and p-value

        Example:
            >>> np.random.seed(1) # seed the random number generator
            >>> control = 5 * np.random.randn(100) + 50
            >>> treatment = 5 * np.random.randn(100) + 51
            >>> test = TwoSampleTest(control, treatment)
            >>> test.run("ind_z")
            >>> test.name
            'ind_z'
            >>> test.stat
            -2.2620139704259556
            >>> test.pval
            0.02369654016419
            >>> test.accept_h0
            False

        References:
            [1] https://stats.stackexchange.com/questions/124096/two-samples-z-test-in-python
            [2] https://stackoverflow.com/questions/65235896/statsmodel-z-test-not-working-as-intended-statsmodels-stats-weightstats-compare
            [3] https://en.wikipedia.org/wiki/Z-test
            [4] https://stats.stackexchange.com/questions/491294/can-you-multiply-p-values-if-you-perform-the-same-test-multiple-times
        """
        col1 = ws.DescrStatsW(self.control)
        col2 = ws.DescrStatsW(self.treatment)

        cm_obj = ws.CompareMeans(col1, col2)

        return cm_obj.ztest_ind(**kwargs)


class SingleSampleTest(HypothesisTest):
    def __init__(self, data: np.array):
        self.data = data

    def stationarity(self, **kwargs):
        """[summary]

        Returns:
            [type]: [description]

        Example:
            >>> x = np.arange(365)
            >>> data = np.sin(pi*x/60)
            >>> test = SingleSampleTest(data)
            >>> test.run("stationarity")
            >>> test.name
            'stationarity'
            >>> test.stat
            -7612364038365.818
            >>> test.pval
            0.0
            >>> test.accept_h0
            False
        """

        stat, pval, _, _, critical_vals, _ = ts.adfuller(self.data, **kwargs)

        return stat, pval
