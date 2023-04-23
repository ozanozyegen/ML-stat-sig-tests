from scipy import stats


def combined_ftest_5x2cv(scores1, scores2):
    """Perform the Combined 5x2CV F-test on two sets of
    model scores to compare their performance.

    Assume R1CV1, R1CV2, R2CV1 ... order
    where R1CV1 is cross-validation repeat 1 and split 1
    Args:
        scores1 (list): Regressor 1 scores on the splits
        scores2 (list): Regressor 2 scores on the splits
    Returns:
        f_stat (float): The F-statistic
        pvalue (float):  If the chosen significance level is larger than
        the p-value, we reject the null hypothesis and accept that
        there are significant differences in the two compared models.
    """
    variances = []
    differences = []
    for i in range(5):
        scores_diff1 = scores1[i * 2] - scores2[i * 2]
        scores_diff2 = scores1[i * 2 + 1] - scores2[i * 2 + 1]
        score_mean = (scores_diff1 + scores_diff2) / 2.0
        score_var = (scores_diff1 - score_mean) ** 2 + \
            (scores_diff2 - score_mean) ** 2

        differences.extend([scores_diff1**2, scores_diff2**2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    pvalue = stats.f.sf(f_stat, 10, 5)
    return float(f_stat), float(pvalue)
