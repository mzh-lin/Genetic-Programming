# Initialization

## Variables

In this problem, the selection of variables is an important part for this algorithm. We can first use some simple daily stock data to build a naive set.

- Open
- High
- Low
- Close
- Volume

In the future we can use some intra-day feature to build some complex variables.

## Primitives Set

One of the most crucial aspect of a GP program is the choice of the primitives set. They should make good building blocks for the individuals so the evolution can succeed. In this problem, we use a classical set of primitives, which are basic arithmetic functions in traditional technical analysis:

### Function

$X,Y$ are two $m*n$ matrix, while $a,d$ are scalars.

| Function              | Return        | Description                                                  |
| --------------------- | ------------- | ------------------------------------------------------------ |
| $add(X,Y)$            | numpy.ndarray | addition                                                     |
| $sub(X,Y)$            | numpy.ndarray | subtraction                                                  |
| $mul(X,Y)$            | numpy.ndarray | multiplication                                               |
| $div(X,Y)$            | numpy.ndarray | division                                                     |
| $abs(X,Y)$            | numpy.ndarray | absolute value                                               |
| $sqrt(X,Y)$           | numpy.ndarray | square root                                                  |
| $log(X,Y)$            | numpy.ndarray | log                                                          |
| $inv(X,Y)$            | numpy.ndarray | inverse                                                      |
| $rank(X)$             | numpy.ndarray | cross section rank for each time t                           |
| $delay(X,d)$          | numpy.ndarray | time series delay d day                                      |
| $corr(X,Y,d)$         | numpy.ndarray | correlation between $X_n$ and $Y_n$ rolling t day            |
| $cov(X,Y,d)$          | numpy.ndarray | covariance between $X_n$ and $Y_n$ rolling t day             |
| $scale(X, a)$         | numpy.ndarray | $a*X/sum(abs(X))$ for each row $m$                           |
| $delta(X, d)$         | numpy.ndarray | $X-delay(X,d)$                                               |
| $signedpower(X, a)$   | numpy.ndarray | $sign(X)*(abs(X)^a)$                                         |
| $decay\_linear(X, d)$ | numpy.ndarray | weighted average of X for past d days, weights are $d,d-1,\dots,1$ (weights must be scaled to one) |
| $ts\_min(X, d)$       | numpy.ndarray | the minimum value for X in past d days                       |
| $ts\_max(X, d)$       | numpy.ndarray | the maximum value for X in past d days                       |
| $ts\_argmin(X, d)$    | numpy.ndarray | the index of the minimum value for X in past d days          |
| $ts\_argmax(X, d)$    | numpy.ndarray | the index of the maximum value for X in past d days          |
| $ts\_rank(X, d)$      | numpy.ndarray | the rank for X in past d days                                |
| $ts\_sum(X, d)$       | numpy.ndarray | the sum for X in past d days                                 |
| $ts\_product(X, d)$   | numpy.ndarray | the production for X in past d days                          |
| $ts\_stddev(X, d)$    | numpy.ndarray | the standard deviation for X in past d days                  |

### Ephemeral Constant

We need some Ephemeral Constant as the params in above function. By calling the function  ```random.randint(1,n)``` , we randomly choose the integrate from 1 to n. In the other words, we only use the information in past n days. 

## Fitness

Now that we can represent a formula as an executable program, we need to determine how well it performs. In factor investing field, people prefer the factor which has a high and stable correlation with the return of underlying assets. We hope to maximized the score we called Ranked ICIR:

$Rank\ IC_t=$ cross section Spearman’s rank-order correlation between the factor and the return of stock at time t
$$
Rank\ ICIR=\frac{mean(Rank\ IC)}{std(Rank\ IC)}
$$

In deap(a gp programming library), we can use Creator to create the fitness object (this is a maximization problem, so the weight is positive). The weights argument must be an iterable of weights, even if there is only one fitness measure. 

```python
creator.create("FitnessMax",base.Fitness,weights=(1,0))
```