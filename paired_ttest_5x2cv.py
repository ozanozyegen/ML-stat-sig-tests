import numpy as np
from scipy import stats


def paired_ttest_5x2cv(scores1, scores2):
    """ Implements the 5x2cv paired t test proposed
    by Dieterrich (1998) to compare the performance of two models.

    Assume R1CV1, R1CV2, R2CV1 ... order
    where R1CV1 is cross-validation repeat 1 and split 1
    Args:
        scores1 (list): Regressor 1 scores on the splits
        scores2 (list): Regressor 2 scores on the splits
    Returns:
        t_stat: The T-statistic
        pvalue (float):  If the chosen significance level is larger than
        the p-value, we reject the null hypothesis and accept that
        there are significant differences in the two compared models.
    """
    variance_sum = 0
    first_diff = None
    for i in range(5):
        scores_diff1 = scores1[i * 2] - scores2[i * 2]
        scores_diff2 = scores1[i * 2 + 1] - scores2[i * 2 + 1]
        score_mean = (scores_diff1 + scores_diff2) / 2.0
        score_var = (scores_diff1 - score_mean) ** 2 + \
            (scores_diff2 - score_mean) ** 2
        variance_sum += score_var
        if first_diff is None:
            first_diff = scores_diff1

    numerator = first_diff
    denominator = np.sqrt(1 / 5.0 * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(pvalue)
