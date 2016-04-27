# Widrow-Hoff learning rule

## data

3-class data(d=2)

border 2 to 3 is **-0.4** and

border 1 to 2 is **0.7**.

ex)

> -1.5 => class 3   
> -0.2 => class 2   
>  0.8 => class 1   


## note

You can change `ROW` as well when you want to change parameters.

`self.weight` is initialized random 0~1 in float.

So, perhaps result is not the same perfectly when you run before.

If you want to get the same results, add `np.random.seed(100)` in main.
