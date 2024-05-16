"""
This type stub file was generated by pyright.
"""

from ._axis_nan_policy import _axis_nan_policy_factory

class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''
    def __init__(self) -> None:
        '''Minimal initializer'''
        ...
    
    def pmf(self, k, m, n): # -> Any:
        ...
    
    def pmf_recursive(self, k, m, n): # -> Any:
        '''Probability mass function, recursive version'''
        ...
    
    def pmf_iterative(self, k, m, n): # -> Any:
        '''Probability mass function, iterative version'''
        ...
    
    def cdf(self, k, m, n): # -> Any:
        '''Cumulative distribution function'''
        ...
    
    def sf(self, k, m, n): # -> ndarray[Any, dtype[Any]]:
        '''Survival function'''
        ...
    


_mwu_state = ...
MannwhitneyuResult = ...
@_axis_nan_policy_factory(MannwhitneyuResult, n_samples=2)
def mannwhitneyu(x, y, use_continuity=..., alternative=..., axis=..., method=...): # -> MannwhitneyuResult:
    r'''Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    difference in location between distributions.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    use_continuity : bool, optional
            Whether a continuity correction (1/2) should be applied.
            Default is True when `method` is ``'asymptotic'``; has no effect
            otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        Let *F(u)* and *G(u)* be the cumulative distribution functions of the
        distributions underlying `x` and `y`, respectively. Then the following
        alternative hypotheses are available:

        * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for
          at least one *u*.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`, i.e. *F(u) > G(u)* for all *u*.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`, i.e. *F(u) < G(u)* for all *u*.

        Note that the mathematical expressions in the alternative hypotheses
        above describe the CDFs of the underlying distributions. The directions
        of the inequalities appear inconsistent with the natural language
        description at first glance, but they are not. For example, suppose
        *X* and *Y* are random variables that follow distributions with CDFs
        *F* and *G*, respectively. If *F(u) > G(u)* for all *u*, samples drawn
        from *X* tend to be less than those drawn from *Y*.

        Under a more restrictive set of assumptions, the alternative hypotheses
        can be expressed in terms of the locations of the distributions;
        see [5] section 5.1.
    axis : int, optional
        Axis along which to perform the test. Default is 0.
    method : {'auto', 'asymptotic', 'exact'} or `PermutationMethod` instance, optional
        Selects the method used to calculate the *p*-value.
        Default is 'auto'. The following options are available.

        * ``'asymptotic'``: compares the standardized test statistic
          against the normal distribution, correcting for ties.
        * ``'exact'``: computes the exact *p*-value by comparing the observed
          :math:`U` statistic against the exact distribution of the :math:`U`
          statistic under the null hypothesis. No correction is made for ties.
        * ``'auto'``: chooses ``'exact'`` when the size of one of the samples
          is less than or equal to 8 and there are no ties;
          chooses ``'asymptotic'`` otherwise.
        * `PermutationMethod` instance. In this case, the p-value
          is computed using `permutation_test` with the provided
          configuration options and other appropriate settings.

    Returns
    -------
    res : MannwhitneyuResult
        An object containing attributes:

        statistic : float
            The Mann-Whitney U statistic corresponding with sample `x`. See
            Notes for the test statistic corresponding with sample `y`.
        pvalue : float
            The associated *p*-value for the chosen `alternative`.

    Notes
    -----
    If ``U1`` is the statistic corresponding with sample `x`, then the
    statistic corresponding with sample `y` is
    ``U2 = x.shape[axis] * y.shape[axis] - U1``.

    `mannwhitneyu` is for independent samples. For related / paired samples,
    consider `scipy.stats.wilcoxon`.

    `method` ``'exact'`` is recommended when there are no ties and when either
    sample size is less than 8 [1]_. The implementation follows the recurrence
    relation originally proposed in [1]_ as it is described in [3]_.
    Note that the exact method is *not* corrected for ties, but
    `mannwhitneyu` will not raise errors or warnings if there are ties in the
    data. If there are ties and either samples is small (fewer than ~10
    observations), consider passing an instance of `PermutationMethod`
    as the `method` to perform a permutation test.

    The Mann-Whitney U test is a non-parametric version of the t-test for
    independent samples. When the means of samples from the populations
    are normally distributed, consider `scipy.stats.ttest_ind`.

    See Also
    --------
    scipy.stats.wilcoxon, scipy.stats.ranksums, scipy.stats.ttest_ind

    References
    ----------
    .. [1] H.B. Mann and D.R. Whitney, "On a test of whether one of two random
           variables is stochastically larger than the other", The Annals of
           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.
    .. [2] Mann-Whitney U Test, Wikipedia,
           http://en.wikipedia.org/wiki/Mann-Whitney_U_test
    .. [3] A. Di Bucchianico, "Combinatorics, computer algebra, and the
           Wilcoxon-Mann-Whitney test", Journal of Statistical Planning and
           Inference, Vol. 79, pp. 349-364, 1999.
    .. [4] Rosie Shier, "Statistics: 2.3 The Mann-Whitney U Test", Mathematics
           Learning Support Centre, 2004.
    .. [5] Michael P. Fay and Michael A. Proschan. "Wilcoxon-Mann-Whitney
           or t-test? On assumptions for hypothesis tests and multiple \
           interpretations of decision rules." Statistics surveys, Vol. 4, pp.
           1-39, 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857732/

    Examples
    --------
    We follow the example from [4]_: nine randomly sampled young adults were
    diagnosed with type II diabetes at the ages below.

    >>> males = [19, 22, 16, 29, 24]
    >>> females = [20, 11, 17, 12]

    We use the Mann-Whitney U test to assess whether there is a statistically
    significant difference in the diagnosis age of males and females.
    The null hypothesis is that the distribution of male diagnosis ages is
    the same as the distribution of female diagnosis ages. We decide
    that a confidence level of 95% is required to reject the null hypothesis
    in favor of the alternative that the distributions are different.
    Since the number of samples is very small and there are no ties in the
    data, we can compare the observed test statistic against the *exact*
    distribution of the test statistic under the null hypothesis.

    >>> from scipy.stats import mannwhitneyu
    >>> U1, p = mannwhitneyu(males, females, method="exact")
    >>> print(U1)
    17.0

    `mannwhitneyu` always reports the statistic associated with the first
    sample, which, in this case, is males. This agrees with :math:`U_M = 17`
    reported in [4]_. The statistic associated with the second statistic
    can be calculated:

    >>> nx, ny = len(males), len(females)
    >>> U2 = nx*ny - U1
    >>> print(U2)
    3.0

    This agrees with :math:`U_F = 3` reported in [4]_. The two-sided
    *p*-value can be calculated from either statistic, and the value produced
    by `mannwhitneyu` agrees with :math:`p = 0.11` reported in [4]_.

    >>> print(p)
    0.1111111111111111

    The exact distribution of the test statistic is asymptotically normal, so
    the example continues by comparing the exact *p*-value against the
    *p*-value produced using the normal approximation.

    >>> _, pnorm = mannwhitneyu(males, females, method="asymptotic")
    >>> print(pnorm)
    0.11134688653314041

    Here `mannwhitneyu`'s reported *p*-value appears to conflict with the
    value :math:`p = 0.09` given in [4]_. The reason is that [4]_
    does not apply the continuity correction performed by `mannwhitneyu`;
    `mannwhitneyu` reduces the distance between the test statistic and the
    mean :math:`\mu = n_x n_y / 2` by 0.5 to correct for the fact that the
    discrete statistic is being compared against a continuous distribution.
    Here, the :math:`U` statistic used is less than the mean, so we reduce
    the distance by adding 0.5 in the numerator.

    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> U = min(U1, U2)
    >>> N = nx + ny
    >>> z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)
    >>> p = 2 * norm.cdf(z)  # use CDF to get p-value from smaller statistic
    >>> print(p)
    0.11134688653314041

    If desired, we can disable the continuity correction to get a result
    that agrees with that reported in [4]_.

    >>> _, pnorm = mannwhitneyu(males, females, use_continuity=False,
    ...                         method="asymptotic")
    >>> print(pnorm)
    0.0864107329737

    Regardless of whether we perform an exact or asymptotic test, the
    probability of the test statistic being as extreme or more extreme by
    chance exceeds 5%, so we do not consider the results statistically
    significant.

    Suppose that, before seeing the data, we had hypothesized that females
    would tend to be diagnosed at a younger age than males.
    In that case, it would be natural to provide the female ages as the
    first input, and we would have performed a one-sided test using
    ``alternative = 'less'``: females are diagnosed at an age that is
    stochastically less than that of males.

    >>> res = mannwhitneyu(females, males, alternative="less", method="exact")
    >>> print(res)
    MannwhitneyuResult(statistic=3.0, pvalue=0.05555555555555555)

    Again, the probability of getting a sufficiently low value of the
    test statistic by chance under the null hypothesis is greater than 5%,
    so we do not reject the null hypothesis in favor of our alternative.

    If it is reasonable to assume that the means of samples from the
    populations are normally distributed, we could have used a t-test to
    perform the analysis.

    >>> from scipy.stats import ttest_ind
    >>> res = ttest_ind(females, males, alternative="less")
    >>> print(res)
    Ttest_indResult(statistic=-2.239334696520584, pvalue=0.030068441095757924)

    Under this assumption, the *p*-value would be low enough to reject the
    null hypothesis in favor of the alternative.

    '''
    ...

