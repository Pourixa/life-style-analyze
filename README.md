# life-style-analyze

This is a personal project I coded after finishing Andrew Ng’s Machine Learning Specialization to test my understanding.
I implemented each algorithm from scratch and compared the results to `scikit-learn`. I used a dataset I found on Kaggle to train models that investigate the relationship between BMI and calories, weight, and fat percentage.

## Procedure

### Linear regression

First I plotted the data — see the figure below:

![graphs](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%201.png)

The relationship between BMI and weight, calories, and fat percentage looks roughly linear. So I started with linear regression, the simplest algorithm.

Before training, I wrote a scaling function to normalize features. Here is the function I used:

```py
def scale(x:np.ndarray,scaling:int=0):

    """

    x (ndarray) : feature values

    scaling (int) : Scaling method

    0 (feature scaling) : min/max < x/max < max/max

    1 (mean normalization) :  x = (x-mean)/max-min

    2 (z-score normalization) : x = (x-mean)/standard-deviation

    """

    match scaling:

        case 0:

            return x/x.max()

        case 1:

            mean=x.mean()

            return (x-mean)/(x.max()-x.min())

        case 2: 

            mean=x.mean()

            std=x.std()

            return (x-mean)/std

        case _:

            raise ValueError("Invalid input!")
```

The user can choose among three methods:

* `0` — feature scaling: `x / x.max()`
* `1` — mean normalization: `(x - mean) / (x.max() - x.min())`
* `2` — z-score: `(x - mean) / std`

> **Note:** later I discovered that when making predictions I must scale features the same way as during training. This function fails on a single sample because the single-sample standard deviation can be zero.

Here is my linear regression using gradient descent:

```py
def linear_regression_gradient_descent(x:np.ndarray,y:np.ndarray,w_i:float=0,b_i:float=0,alpha:float=0.001,iters:int=1000,cost:bool=False):

    """

    x (ndarray) : feature values shape(n feature,m samples)

    y (ndarray) : target values

    w_i (float) : initial regression coefficient

    b_i (float) : initial interception value

    alpha (float) : value of learning rate

    iters (int) : number of gradient descent iterations

    cost (bool) : return cost function and number of iteration for learning curve

    """

    w=w_i

    b=b_i

    a=alpha

    m = len(x)      #total values

    cost_history=[]    #saving data for learning curve

    x_iters=np.arange(0,iters)

    for _ in range(iters):

        #Model function

        f_wb=w*x+b

        #cost function

        j_wb=(1/(2*m)) * np.sum((f_wb - y)**2)

        #derivatives

        dj_dw = (1/m) * np.sum((f_wb - y) * x)

        dj_db = (1/m) * np.sum((f_wb - y))

        cost_history.append(j_wb)

        #gradient descent

        w-= a*dj_dw

        b-= a*dj_db

    match cost:

        case False:

           return w,b

        case True:

            return w,b,cost_history,x_iters

        case _ :

            raise ValueError("Invalid input!")
```

Here is the cost function plotted over iterations using my model:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%202.png)

Below are the regression lines for BMI versus fat percentage, calories, and weight:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%203.png)

I also trained a `scikit-learn` model — the results match closely:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%204.png)

---

### Multiple Linear Regression

Next I implemented multiple linear regression:

```py
#multiple linear regression.

def multiple_linear_regression_gradient_descent(*x:np.ndarray,y:np.ndarray,w_i:float=0,b_i:float=0,alpha:float=0.001,iters:int=1000,cost:bool=False):

    """

    *x (ndarray) : features values

    y (ndarray) : target values

    w_i (float) : initial regression coefficient

    b_i (float) : initial interception value

    alpha (float) : value of learning rate

    iters (int) : number of gradient descent iterations

    cost (bool) : return cost function and number of iteration for learning curve

    """

    x=np.array(x)

    y=np.array(y)

    w=np.full((x.shape[0],1),w_i,dtype="float64")

    b=b_i

    m=x.shape[1]

    a=alpha

    cost_history=[]    #saving data for learning curve

    x_iters=np.arange(0,iters)

    for _ in range(iters):

        f_wb = np.dot(w.T,x) + b

        j_wb=(1/(2*m)) * np.sum((f_wb - y)**2)

        dj_dw = (1/m) * np.dot((f_wb-y),x.T) #1*2

        dj_db = (1/m) * np.sum((f_wb-y))

        cost_history.append(j_wb)

        w-= a*dj_dw.T

        b-= a*dj_db

    match cost:

        case False:

           return w,b

        case True:

            return w,b,cost_history,x_iters

        case _ :

            raise ValueError("Invalid input!")
```

> **Note:** I used `*x` to accept features individually and then assembled them into an array; later I learned it’s simpler to pass `X` as an `(n_samples, n_features)` array.

Cost curve:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%205.png)

This worked well for two features. For visualization I removed one feature so I could plot the surface:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%206.png)

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%207.png)

The fit looks solid. I compared it to `scikit-learn` — to show both surfaces I reduced the opacity of the `sklearn` surface so the two overlap visibly (black = my model, blue = sklearn):

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%208.png)

---

### Polynomial Regression

I implemented polynomial regression next:

```py
#Polynomial regression

def polynomial_regression_gradient_descent(*x:tuple,y:np.ndarray,w_i:float=0,b_i:float=0,alpha:float=0.001,iters:int=1000,cost:bool=False):

    """

    *x (tuple) : features values and their exponent | (x:ndarray,e:float)

    y (ndarray) : target values

    w_i (float) : initial regression coefficient

    b_i (float) : initial interception value

    alpha (float) : value of learning rate

    iters (int) : number of gradient descent iterations

    cost (bool) : return cost function and number of iteration for learning curve

    """

    w=np.full(len(x),w_i,dtype="float64").reshape((1,-1))

    b=b_i

    a=alpha

    xa=[]

    e=[]

    for i in range(len(x)):

        xa.append(x[i][0])

    for i in range(len(x)):

        e.append(x[i][1])

    x=np.array(xa)

    e=np.array(e)

    m=len(x[0])

    x=np.power(x.T,e)

    cost_history=[]    #saving data for learning curve

    x_iters=np.arange(0,iters)

    for _ in range(iters):

        f_wb=np.dot(w,x.T) + b

        j_wb=(1/(2*m)) * np.sum((f_wb - y)**2)

        dj_dw = (1/m) * np.dot((f_wb-y),x)

        dj_db = (1/m) * np.sum((f_wb-y))

        cost_history.append(j_wb)

        w-=a*dj_dw

        b-=a*dj_db

    match cost:

        case False:

           return w,b

        case True:

            return w,b,cost_history,x_iters

        case _ :

            raise ValueError("Invalid input!")
```

**Note:** `*x` is a tuple where each element is `(samples, exponent)`, so `(x1, 3)` means that the feature `x1` will be raised to the third power.

Cost curve:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%209.png )

For visualization I removed one feature:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2010.png)

Then I compared to `scikit-learn`:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2011.png)

My polynomial implementation didn’t line up as closely as the other algorithms — I suspect a bug in my implementation and will try to fix it later.

---

### Logistic Regression

Finally, I implemented logistic regression. I created a toy example to predict gender from height. For practice I split the data in half, labeled female as `0` and male as `1`, and sorted by height. This artificially separates the classes and is not realistic — many exceptions exist in real data.

Here’s the plot of the data:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2012.png)

I also plotted what would happen if I used linear regression for this problem.

The sigmoid function I used:

```py
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Then I applied gradient descent to find the decision boundary:

```py
def logistic_regression_gradient_descent(*x:tuple,y:np.ndarray,w_i:float=0,b_i:float=0,alpha:float=0.001,iters:int=1000,cost:bool=False):

    """

    *x (tuple) : features values and their exponent | (x:ndarray,e:float)

    y (ndarray) : target values

    w_i (float) : initial regression coefficient

    b_i (float) : initial interception value

    alpha (float) : value of learning rate

    iters (int) : number of gradient descent iterations

    cost (bool) : return cost function and number of iteration for learning curve

    """

    w=np.full(len(x),w_i,dtype="float64").reshape((1,-1))

    b=b_i

    a=alpha

    xa=[]

    e=[]

    for i in range(len(x)):

        xa.append(x[i][0])

    for i in range(len(x)):

        e.append(x[i][1])

    x=np.array(xa)

    e=np.array(e)

    m=len(x[0])

    x=np.power(x.T,e)

    cost_history=[]    #saving data for learning curve

    x_iters=np.arange(0,iters)

    for _ in range(iters):

        z=np.dot(w,x.T) + b

        f_wb=sigmoid(z)

        j_wb=(-1/(m)) * np.sum(y*np.log(f_wb)+(1-y)*np.log(1-f_wb))

        dj_dw = (1/m) * np.dot((f_wb-y),x)

        dj_db = (1/m) * np.sum((f_wb-y))

        cost_history.append(j_wb)

        w-=a*dj_dw

        b-=a*dj_db

    match cost:

        case False:

           return w,b

        case True:

            return w,b,cost_history,x_iters

        case _ :

            raise ValueError("Invalid input!")
```

The cost curve:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2013.png)

It doesn’t flatten completely even after many iterations (I ran gradient descent 10,000 times with a learning rate of `0.1`).

Here is the decision boundary and the sigmoid function:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2014.png)

Finally, I compared my result with `scikit-learn`:

![graph](https://github.com/Pourixa/life-style-analyze/blob/40c1a50cff11674ffa2de595fba79d9b8b337bcf/img/Figure%2015.png)

There isn’t a huge difference; `sklearn` produced a slightly better decision boundary, but the results are comparable.
