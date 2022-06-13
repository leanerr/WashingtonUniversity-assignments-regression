# Regression Week 4: Ridge Regression (interpretation)

In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
* Use a pre-built implementation of regression (Turi Create) to run polynomial regression
* Use matplotlib to visualize polynomial regressions
* Use a pre-built implementation of regression (Turi Create) to run polynomial regression, this time with L2 penalty
* Use matplotlib to visualize polynomial regressions under L2 regularization
* Choose best L2 penalty using cross-validation.
* Assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# Fire up Turi Create


```python
import turicreate
```

# Polynomial regression, revisited

We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:


```python
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = turicreate.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            tmp = feature.apply(lambda x: x**power)
            poly_sframe[name] = tmp
    return poly_sframe
```

Let's use matplotlib to visualize what a polynomial regression looks like on the house data.


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
sales = turicreate.SFrame('m_1ce96d9d245ca490.frame_idx')
```

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living','price'])
```

Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:


```python
l2_small_penalty = 1e-5
l2_penalty=1e-5
```


```python
poly15_data = polynomial_sframe(sales['sqft_living'],15)
poly15_data
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_1</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_2</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_3</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_4</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_5</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_6</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_7</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">290.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">84100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24389000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7072810000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2051114900000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">594823321000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7249876309e+17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">370.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">136900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">50653000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18741610000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6934395700000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2565726409000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.4931877133e+17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">380.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">144400.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">54872000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20851360000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7923516800000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3010936384000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.14415582592e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">384.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">147456.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">56623104.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21743271936.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8349416423424.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3206175906594816.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2311715481324093e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">390.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">152100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">59319000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">23134410000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9022419900000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3518743761000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.37231006679e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">390.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">152100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">59319000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">23134410000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9022419900000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3518743761000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.37231006679e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">410.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">168100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">68921000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">28257610000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11585620100000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4750104241000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.94754273881e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">420.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">176400.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">74088000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">31116960000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13069123200000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5489031744000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.30539333248e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">420.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">176400.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">74088000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">31116960000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13069123200000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5489031744000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.30539333248e+18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">430.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">184900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">79507000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34188010000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14700844300000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6321363049000000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.71818611107e+18</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_8</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_9</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_10</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_11</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.00246412961e+19</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.4507145975869e+22</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.20707233300201e+24</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2200509765705829e+27</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.512479453921e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.29961739795077e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.808584372417849e+25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7791762177946042e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.347792138496e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.65216101262848e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.278211847988224e+25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.3857205022355252e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.727698744828452e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.8154363180141255e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.971275461174242e+25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.676969777090909e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.352009260481e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.08728361158759e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.140406085191601e+25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.1747583732247245e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.352009260481e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.08728361158759e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.140406085191601e+25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.1747583732247245e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.984925229121e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.27381934393961e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.3422659310152401e+26</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.503290317162485e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.682651996416e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.06671383849472e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7080198121677826e+26</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.173683211104686e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.682651996416e+20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.06671383849472e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7080198121677826e+26</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.173683211104686e+28</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.1688200277601e+21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.02592611936843e+23</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.161148231328425e+26</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.292937394712227e+28</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_12</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_13</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_14</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">power_15</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.53814783205469e+29</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0260628712958602e+32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.9755823267579947e+34</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.629188747598184e+36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.582952005840035e+30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.435692242160813e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.012061295995008e+35</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.334462679518153e+38</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.065737908494996e+30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.444980405228098e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.3090925539866774e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.974551705149374e+38</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.027956394402909e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.9473525545071707e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.5157833809307535e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.8206081827740936e+38</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2381557655576424e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.8288074856748057e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.8832349194131742e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.34461618571138e+38</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2381557655576424e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.8288074856748057e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.8832349194131742e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.34461618571138e+38</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.2563490300366187e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.251031023150136e+33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.792922719491556e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.555098314991538e+39</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.012946948663968e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2654377184388666e+34</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.31483841744324e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.2322321353261606e+39</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.012946948663968e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2654377184388666e+34</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.31483841744324e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.2322321353261606e+39</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.995963079726258e+31</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7182641242822908e+34</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.38853573441385e+36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.1770703657979555e+39</td>
    </tr>
</table>
[21613 rows x 15 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)

With the L2 penalty specified above, fit the model and print out the learned weights.

Hint: make sure to add 'price' column to the new SFrame before calling `turicreate.linear_regression.create()`. Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set=None` in this call.


```python
poly15_data = polynomial_sframe(sales['sqft_living'],15)
my_features = poly15_data.column_names()
poly15_data['price'] = sales['price']
model15 = turicreate.linear_regression.create(poly15_data,target='price',features=my_features,l2_penalty=l2_small_penalty,validation_set=None)


```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.553648     | 2662555.740373     | 245656.462168                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
model15.coefficients

```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">167924.8583256238</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">103.09094647349599</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.13460456308152913</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.00012907137783039016</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.1892904286759434e-08</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-7.771696034002178e-12</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7114543513050507e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.5117738872015835e-20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-4.788388245043285e-25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.3334328221412876e-28</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
</table>
[16 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# Observe overfitting

Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 


```python
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)
```

Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.

Hint: When calling `turicreate.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
poly01_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly01_data.column_names() # get the name of the features
poly01_data['price'] = set_1['price'] # add price to the data since it's the target
model01 = turicreate.linear_regression.create(poly01_data, target = 'price', features = my_features, l2_penalty=l2_small_penalty,validation_set = None)
model01.coefficients.print_rows(num_rows = 16)
plt.plot(poly01_data['power_1'],poly01_data['price'],'.',
        poly01_data['power_1'], model01.predict(poly01_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.478515     | 2191984.902429     | 248699.117251                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |     9306.46911851177    |   883694.9239156723    |
    |   power_1   |  None |    585.8657979889792    |   3849.587810042651    |
    |   power_2   |  None |   -0.39730586781956484  |   6.927486995536432    |
    |   power_3   |  None |  0.00014147088639384313 |  0.006797732890308357  |
    |   power_4   |  None | -1.5294595567376532e-08 | 4.045891040095697e-06  |
    |   power_5   |  None | -3.7975661137124193e-13 | 1.5369945376872841e-09 |
    |   power_6   |  None |  5.974815387264118e-17  | 3.8400475548421245e-13 |
    |   power_7   |  None |  1.0688854378568836e-20 |  6.61302154279885e-17  |
    |   power_8   |  None |  1.593440657641439e-25  | 8.875839342463916e-21  |
    |   power_9   |  None |  -6.928349573805408e-29 | 9.897239847583721e-25  |
    |   power_10  |  None |  -6.838135946598277e-33 | 6.990388673262675e-29  |
    |   power_11  |  None |  -1.626861217068946e-37 |          nan           |
    |   power_12  |  None |  2.851187892085434e-41  |          nan           |
    |   power_13  |  None |  3.799981518539984e-45  | 1.7195929202640956e-41 |
    |   power_14  |  None |  1.5265267099563408e-49 | 9.701551470557726e-46  |
    |   power_15  |  None | -2.3380734460550423e-53 |  2.14462774816717e-50  |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11e02fead0>,
     <matplotlib.lines.Line2D at 0x7f11e02fe550>]




![png](output_24_17.png)



```python
poly02_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly02_data.column_names() # get the name of the features
poly02_data['price'] = set_2['price'] # add price to the data since it's the target
model02 = turicreate.linear_regression.create(poly02_data, target = 'price', features = my_features,l2_penalty=l2_small_penalty, validation_set = None)
model02.coefficients.print_rows(num_rows = 16)
plt.plot(poly02_data['power_1'],poly02_data['price'],'.',
        poly02_data['power_1'], model02.predict(poly02_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.676680     | 1975178.189311     | 234533.610640                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |   -25115.923486618092   |   2035199.6082566872   |
    |   power_1   |  None |    783.4938642047707    |   12158.136719034628   |
    |   power_2   |  None |   -0.7677593839265497   |   30.524187705069675   |
    |   power_3   |  None |  0.0004387664198818728  |  0.042697915424997575  |
    |   power_4   |  None | -1.1516918417633088e-07 | 3.720303507718868e-05  |
    |   power_5   |  None |  6.842817271468358e-12  | 2.1344178702759516e-08 |
    |   power_6   |  None |  2.5119512059090315e-15 | 8.290716541977766e-12  |
    |   power_7   |  None | -2.0644048036423923e-19 | 2.2044284855048032e-15 |
    |   power_8   |  None | -4.5967318511045613e-23 | 4.0254833703473075e-19 |
    |   power_9   |  None | -2.7127303907683216e-29 | 5.1026810499214924e-23 |
    |   power_10  |  None |  6.218184574011556e-31  | 4.4420508864405564e-27 |
    |   power_11  |  None |   6.51741397777227e-35  | 1.9021797372671635e-31 |
    |   power_12  |  None |  -9.41315104664047e-40  | 4.056313939281119e-35  |
    |   power_13  |  None | -1.0242139004677438e-42 | 6.021969350406664e-39  |
    |   power_14  |  None | -1.0039109057995391e-46 | 3.7873865111225014e-43 |
    |   power_15  |  None |  1.3011336947932297e-50 | 9.208134022456939e-48  |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11e044b8d0>,
     <matplotlib.lines.Line2D at 0x7f11e0406fd0>]




![png](output_25_17.png)



```python
poly03_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly03_data.column_names() # get the name of the features
poly03_data['price'] = set_3['price'] # add price to the data since it's the target
model03 = turicreate.linear_regression.create(poly03_data, target = 'price', features = my_features, l2_penalty=l2_small_penalty,validation_set = None)
model03.coefficients.print_rows(num_rows = 16)
plt.plot(poly03_data['power_1'],poly03_data['price'],'.',
        poly03_data['power_1'], model03.predict(poly03_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.665918     | 2283722.683345     | 251097.728062                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |    462426.5817589968    |   4469956.252955852    |
    |   power_1   |  None |    -759.2518952484717   |   25130.544871572558   |
    |   power_2   |  None |    1.0286701120640382   |   58.85671529799961    |
    |   power_3   |  None |  -0.0005282645662534578 |  0.07626141665345167   |
    |   power_4   |  None |  1.154229200694207e-07  | 6.112218452942081e-05  |
    |   power_5   |  None | -2.2609608859339102e-12 | 3.197688941851107e-08  |
    |   power_6   |  None | -2.0821429500690547e-15 | 1.1156078042283364e-11 |
    |   power_7   |  None |  4.087707610650796e-20  | 2.570225971807701e-15  |
    |   power_8   |  None |   2.57079130209995e-23  | 3.645773574370856e-19  |
    |   power_9   |  None |  1.2431125587439647e-27 |  2.18823494064304e-23  |
    |   power_10  |  None | -1.7202588793955896e-31 |  1.30473523928642e-27  |
    |   power_11  |  None |  -2.967609923932442e-35 | 5.228392166297212e-31  |
    |   power_12  |  None |  -1.065749434886338e-39 | 5.233427574807933e-35  |
    |   power_13  |  None |  2.426357060036537e-43  | 2.6932340923844063e-39 |
    |   power_14  |  None |  3.5559866817473437e-47 |  6.91338418412126e-44  |
    |   power_15  |  None | -2.8577743307249887e-51 | 6.706108216742883e-49  |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d8785950>,
     <matplotlib.lines.Line2D at 0x7f11d8785b90>]




![png](output_26_17.png)



```python
poly04_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly04_data.column_names() # get the name of the features
poly04_data['price'] = set_4['price'] # add price to the data since it's the target
model04 = turicreate.linear_regression.create(poly04_data, target = 'price', features = my_features, l2_penalty=l2_small_penalty,validation_set = None)
model04.coefficients.print_rows(num_rows = 16)
plt.plot(poly04_data['power_1'],poly04_data['price'],'.',
        poly04_data['power_1'], model04.predict(poly04_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.674211     | 2378292.371620     | 244341.293204                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |    -170240.0431876816   |   1657485.6625040893   |
    |   power_1   |  None |    1247.5903823305678   |   10919.946344644475   |
    |   power_2   |  None |    -1.224609172719573   |   29.832391699268413   |
    |   power_3   |  None |   0.000555254662311615  |  0.04489923707128707   |
    |   power_4   |  None |  -6.382625193214442e-08 | 4.168836610488775e-05  |
    |   power_5   |  None |  -2.202159547124807e-11 | 2.5226919963842032e-08 |
    |   power_6   |  None |  4.8183463260806445e-15 | 1.018371948330214e-11  |
    |   power_7   |  None |  4.214616831978828e-19  | 2.740916130713002e-15  |
    |   power_8   |  None |  -7.998807484235487e-23 | 4.855419340174364e-19  |
    |   power_9   |  None | -1.3236591177677348e-26 | 6.313960173444996e-23  |
    |   power_10  |  None |  1.6019784814104984e-31 | 9.802559217426981e-27  |
    |   power_11  |  None |  2.399043308636641e-34  | 1.4467520607930132e-30 |
    |   power_12  |  None |  2.3335451300571133e-38 | 1.5620973589816553e-34 |
    |   power_13  |  None | -1.7987405831667493e-42 | 1.3386130452266427e-38 |
    |   power_14  |  None |  -6.02862686120259e-46  | 7.621669073485304e-43  |
    |   power_15  |  None |  4.394726748702676e-50  | 1.9135972913408425e-47 |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d876ccd0>,
     <matplotlib.lines.Line2D at 0x7f11d876cf10>]




![png](output_27_17.png)


The four curves should differ from one another a lot, as should the coefficients you learned.

***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# Ridge regression comes to 


```python
l2_penalty=1e5
```

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
poly01_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly01_data.column_names() # get the name of the features
poly01_data['price'] = set_1['price'] # add price to the data since it's the target
model01 = turicreate.linear_regression.create(poly01_data, target = 'price', features = my_features, l2_penalty=l2_penalty,validation_set = None)
model01.coefficients.print_rows(num_rows = 16)
plt.plot(poly01_data['power_1'],poly01_data['price'],'.',
        poly01_data['power_1'], model01.predict(poly01_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.410638     | 5978778.434729     | 374261.720860                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   530317.0245158835    |   1329852.6613023512   |
    |   power_1   |  None |    2.58738875672869    |   5793.158312393828    |
    |   power_2   |  None | 0.0012741440059211384  |   10.425019730033746   |
    |   power_3   |  None | 1.7493422693158888e-07 |  0.010229755688697093  |
    |   power_4   |  None | 1.0602211909664251e-11 | 6.0885706383485295e-06 |
    |   power_5   |  None | 5.422476044821804e-16  | 2.312988590331272e-09  |
    |   power_6   |  None | 2.895638283427736e-20  | 5.778801396421396e-13  |
    |   power_7   |  None | 1.6500066635095529e-24 |  9.95178772666577e-17  |
    |   power_8   |  None | 9.860815284092936e-29  | 1.3357051456814885e-20 |
    |   power_9   |  None |  6.06589348254357e-33  | 1.4894134157222879e-24 |
    |   power_10  |  None | 3.789178688696589e-37  | 1.0519679053359913e-28 |
    |   power_11  |  None | 2.3822312131219896e-41 |          nan           |
    |   power_12  |  None | 1.498479692145694e-45  |          nan           |
    |   power_13  |  None | 9.391611902848268e-50  | 2.5877767988493167e-41 |
    |   power_14  |  None | 5.845231619806178e-54  | 1.4599647109791573e-45 |
    |   power_15  |  None | 3.601202072029721e-58  |  3.22740217377908e-50  |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d86dfd50>,
     <matplotlib.lines.Line2D at 0x7f11d86dff90>]




![png](output_32_17.png)



```python
poly02_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly02_data.column_names() # get the name of the features
poly02_data['price'] = set_2['price'] # add price to the data since it's the target
model02 = turicreate.linear_regression.create(poly02_data, target = 'price', features = my_features,l2_penalty=l2_penalty, validation_set = None)
model02.coefficients.print_rows(num_rows = 16)
plt.plot(poly02_data['power_1'],poly02_data['price'],'.',
        poly02_data['power_1'], model02.predict(poly02_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.668673     | 2984894.541944     | 323238.809634                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   519216.89738342643   |   2804951.9083565865   |
    |   power_1   |  None |   2.044704741819378    |   16756.58183785149    |
    |   power_2   |  None |  0.001131436268395812  |   42.06903254453224    |
    |   power_3   |  None | 2.9307427754897134e-07 |  0.058847102204772926  |
    |   power_4   |  None | 4.435405984532598e-11  | 5.1273950630230305e-05 |
    |   power_5   |  None | 4.808491122043447e-15  | 2.9416964577686945e-08 |
    |   power_6   |  None | 4.530917078263863e-19  | 1.1426427703562646e-11 |
    |   power_7   |  None | 4.1604291057458364e-23 | 3.038186457076231e-15  |
    |   power_8   |  None | 3.900946351283381e-27  | 5.547999918978602e-19  |
    |   power_9   |  None | 3.777318760202607e-31  | 7.032614830823562e-23  |
    |   power_10  |  None | 3.7665032684171824e-35 | 6.1221214176683534e-27 |
    |   power_11  |  None | 3.842280947539598e-39  |  2.6216213203063e-31   |
    |   power_12  |  None | 3.985208284143722e-43  | 5.590491212125366e-35  |
    |   power_13  |  None | 4.1827276239367314e-47 | 8.299595947719759e-39  |
    |   power_14  |  None | 4.427383328777781e-51  | 5.219850170449319e-43  |
    |   power_15  |  None | 4.715182454121553e-55  | 1.2690830419733565e-47 |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d865b710>,
     <matplotlib.lines.Line2D at 0x7f11d865b950>]




![png](output_33_17.png)



```python
poly03_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly03_data.column_names() # get the name of the features
poly03_data['price'] = set_3['price'] # add price to the data since it's the target
model03 = turicreate.linear_regression.create(poly03_data, target = 'price', features = my_features, l2_penalty=l2_penalty,validation_set = None)
model03.coefficients.print_rows(num_rows = 16)
plt.plot(poly03_data['power_1'],poly03_data['price'],'.',
        poly03_data['power_1'], model03.predict(poly03_data),'-')

```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.476968     | 3695342.767093     | 350033.521294                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   522911.5180475718    |   6231177.555153195    |
    |   power_1   |  None |   2.268904218765781    |    35032.3086605071    |
    |   power_2   |  None | 0.0012590504184157225  |   82.04703191276603    |
    |   power_3   |  None | 2.7755291815451754e-07 |  0.10630941353417926   |
    |   power_4   |  None | 3.2093309779039004e-11 | 8.520512568994496e-05  |
    |   power_5   |  None | 2.8757357236448285e-15 | 4.457620261864559e-08  |
    |   power_6   |  None | 2.5007611267119203e-19 | 1.5551718890905234e-11 |
    |   power_7   |  None | 2.2468526590627845e-23 |  3.58292866437101e-15  |
    |   power_8   |  None | 2.0934998313470212e-27 | 5.082256107711997e-19  |
    |   power_9   |  None | 2.0043538329631967e-31 | 3.0504281643741544e-23 |
    |   power_10  |  None | 1.954108002485115e-35  | 1.818818010373774e-27  |
    |   power_11  |  None | 1.9273411945583563e-39 | 7.288447150825337e-31  |
    |   power_12  |  None | 1.914836990129071e-43  | 7.295466576232977e-35  |
    |   power_13  |  None | 1.9110227704649904e-47 | 3.7544035953688895e-39 |
    |   power_14  |  None | 1.912462423017048e-51  | 9.637348090322117e-44  |
    |   power_15  |  None | 1.9169955803503674e-55 | 9.348402677311253e-49  |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d85c4250>,
     <matplotlib.lines.Line2D at 0x7f11d85c4490>]




![png](output_34_17.png)



```python
poly04_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly04_data.column_names() # get the name of the features
poly04_data['price'] = set_4['price'] # add price to the data since it's the target
model04 = turicreate.linear_regression.create(poly04_data, target = 'price', features = my_features,l2_penalty=l2_penalty, validation_set = None)
model04.coefficients.print_rows(num_rows = 16)
plt.plot(poly04_data['power_1'],poly04_data['price'],'.',
        poly04_data['power_1'], model04.predict(poly04_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.470022     | 3601895.280124     | 323111.582889                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   513667.0870874073    |   2191822.794279959    |
    |   power_1   |  None |   1.9104093824432022   |   14440.298249304962   |
    |   power_2   |  None |  0.001100580291747722  |   39.449702409829094   |
    |   power_3   |  None | 3.1275398787880605e-07 |  0.059373769248748445  |
    |   power_4   |  None | 5.5006788682463904e-11 | 5.5127783697951324e-05 |
    |   power_5   |  None | 7.204675578247076e-15  | 3.3359527299132035e-08 |
    |   power_6   |  None | 8.249772493837889e-19  | 1.3466727947638874e-11 |
    |   power_7   |  None | 9.065032234977419e-23  | 3.624527553034904e-15  |
    |   power_8   |  None | 9.956831604526309e-27  | 6.420700357373784e-19  |
    |   power_9   |  None | 1.1083812798160369e-30 |  8.34944285999122e-23  |
    |   power_10  |  None | 1.2531522414327038e-34 | 1.296268994723963e-26  |
    |   power_11  |  None | 1.4360078140197673e-38 | 1.9131532876893333e-30 |
    |   power_12  |  None | 1.662699678001347e-42  | 2.0656833876486512e-34 |
    |   power_13  |  None | 1.939817245296962e-46  | 1.7701526182830714e-38 |
    |   power_14  |  None | 2.2754148577027196e-50 | 1.0078728512490218e-42 |
    |   power_15  |  None | 2.6794878489713864e-54 | 2.5304992116171003e-47 |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f11d85b1490>,
     <matplotlib.lines.Line2D at 0x7f11d85b16d0>]




![png](output_35_17.png)


These curves should vary a lot less, now that you applied a high degree of regularization.

***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# Selecting an L2 penalty via cross-validation

Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
...<br>
Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. The package turicreate_cross_validation (see below) has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answer.)

  
_Note:_ For applying cross-validation, we will import a package called `turicreate_cross_validation`. To install it, please run this command on your terminal:

`pip install -e git+https://github.com/Kagandi/turicreate-cross-validation.git#egg=turicreate_cross_validation`

You can find the documentation on this package here: https://github.com/Kagandi/turicreate-cross-validation


```python
import turicreate_cross_validation.cross_validation as tcv

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = tcv.shuffle_sframe(train_valid, random_seed=1)
```

Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.


```python
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in range(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print(i, (start, end))
```

    0 (0.0, 1938.6)
    1 (1939.6, 3878.2)
    2 (3879.2, 5817.8)
    3 (5818.8, 7757.4)
    4 (7758.4, 9697.0)
    5 (9698.0, 11636.6)
    6 (11637.6, 13576.2)
    7 (13577.2, 15515.8)
    8 (15516.8, 17455.4)
    9 (17456.4, 19395.0)


Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.


```python
train_valid_shuffled[0:10] # rows 0 to 9
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">date</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">price</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bedrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bathrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">floors</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">waterfront</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8645511350</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-12-01 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">300000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1810.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21138.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7237501370</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-07-17 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1079000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12727.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7278700100</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-01-21 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">625000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2740.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9599.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1421079007</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-03-24 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">408506.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2480.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">209199.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4338800370</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-11-17 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">220000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6020.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7511200020</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-08-29 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">509900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1690.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">53578.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3300701615</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-09-30 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">655000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7011200260</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-12-19 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">485000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1400.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3600.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3570000130</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-06-11 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">580379.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">27820.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2796100640</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-04-24 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">264900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2040.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">view</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">condition</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">grade</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_above</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_basement</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_built</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_renovated</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">zipcode</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">lat</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">570.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1977.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98058</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.46736904</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98059</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.53108576</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1820.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">920.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1961.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98177</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.77279701</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1870.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">610.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98010</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.30847072</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1944.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98166</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.47933643</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1690.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1984.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98053</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.6545751</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2002.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98117</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.69151411</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">300.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98119</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.63846783</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1976.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98075</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.59357299</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1250.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">790.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1979.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98031</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.40555074</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">long</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living15</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot15</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.17768631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1850.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12200.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.13389261</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4750.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13602.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.38485302</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2660.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8280.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-121.88816296</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2040.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">219229.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.34575463</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1300.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8640.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.04899568</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2290.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">52707.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.38139901</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1640.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4000.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.36993806</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2048.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.05362447</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2330.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20000.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.17648783</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7378.0</td>
    </tr>
</table>
[10 rows x 21 columns]<br/>
</div>



Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.


```python
validation4 = train_valid_shuffled[5818:7758]

```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $559,642.


```python
print(int(round(validation4['price'].mean(), 0)))
```

    559490


After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.


```python
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
first_two.append(last_two)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">date</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">price</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bedrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bathrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">floors</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">waterfront</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8645511350</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-12-01 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">300000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1810.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21138.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7237501370</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-07-17 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1079000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12727.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4077800582</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-09-12 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">522000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1150.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7080.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7853370620</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-02-06 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">605000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3040.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">view</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">condition</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">grade</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_above</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_basement</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_built</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_renovated</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">zipcode</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">lat</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">570.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1977.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98058</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.46736904</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98059</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.53108576</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1150.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1952.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98125</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.71063854</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2280.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">760.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98065</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.51887717</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">long</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living15</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot15</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.17768631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1850.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12200.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.13389261</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4750.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13602.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.28837299</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1490.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7921.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-121.87558112</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3070.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5558.0</td>
    </tr>
</table>
[4 rows x 21 columns]<br/>
</div>



Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.


```python
train4 = train_valid_shuffled[0:5818].append(train_valid_shuffled[7758:n])

```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $536,865.


```python
int(round(train4['price'].mean(), 0))
```




    536866



Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.

* For each i in [0, 1, ..., k-1]:
  * Compute starting and ending indices of segment i and call 'start' and 'end'
  * Form validation set by taking a slice (start:end+1) from the data.
  * Form training set by appending slice (end+1:n) to the end of slice (0:start).
  * Train a linear model using training set just formed, with a given l2_penalty
  * Compute validation error using validation set just formed


```python
import numpy as np
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    empty_vector = np.empty(k)
    n = len(data)
    for i in range(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        #print i, (start, end)
        validation_set = data[start:end+1]
        train_set = data[0:start].append(data[end+1:n])
        model = turicreate.linear_regression.create(train_set,target=output_name, features=features_list,l2_penalty=l2_penalty,validation_set=None)
        predict = model.predict(validation_set)
        errors = validation_set[output_name] - predict
        square_errors = errors ** 2
        RSS = square_errors.sum()
        empty_vector[i] = RSS
    return empty_vector.mean()
    print('mean: '+ str(empty_vector.mean()))
```

Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
* We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
* For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
    * Run 10-fold cross-validation with `l2_penalty`
* Report which L2 penalty produced the lowest average validation error.

Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!


```python
poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
my_features = poly_data.column_names() # get the name of the features
poly_data['price'] = train_valid_shuffled['price'] # add price to the data since it's the target
a = np.logspace(1, 7, num=13)
nn = len(a)
error_vector = np.empty(13)
for i in range(nn):
    #print 'l2_penalty: ' + str(l2_penalty)
    
    error_vector[i] = k_fold_cross_validation(10, a[i], poly_data, 'price', my_features)
    
    #print 'error_vector: ' + str(error_vector)
print(error_vector)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.689295     | 2294888.562886     | 249144.499425                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.568277     | 2286325.709104     | 248774.399860                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.695720     | 2208982.587195     | 245552.806680                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.669489     | 2309002.827632     | 244477.872977                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.574574     | 2321480.736144     | 246905.451990                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.582640     | 2323899.211423     | 248413.276806                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.579381     | 2459696.484312     | 248829.743248                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.673580     | 2309077.033937     | 247741.645735                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.572294     | 2312295.722362     | 248506.647627                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.684564     | 2393327.579088     | 245320.536762                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.772496     | 2321569.319654     | 249284.017817                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.672539     | 2330626.436280     | 248877.816217                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.675805     | 2251162.990766     | 245849.241167                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.667613     | 2263997.230058     | 244620.787794                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.796566     | 2290188.080708     | 247032.884248                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.700714     | 2289947.954328     | 248575.909450                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.766555     | 2376010.440669     | 249027.803525                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.579565     | 2304390.920411     | 247901.523528                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.681310     | 2301677.259466     | 248651.236808                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.477767     | 2349770.403948     | 245445.367387                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.509143     | 2379742.974523     | 249444.967375                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.567027     | 2388721.707170     | 249041.329266                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.692555     | 2342134.209196     | 246216.350290                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.474733     | 2210567.741305     | 244771.891822                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.689218     | 2344991.174744     | 247192.279984                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.604560     | 2345699.304503     | 248757.084811                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.574808     | 2299805.272620     | 249427.616155                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.569245     | 2359406.526325     | 248083.372575                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.601471     | 2356225.056688     | 248815.952738                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.780359     | 2292999.501998     | 245612.751166                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.676498     | 2447015.020270     | 249750.229813                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.780595     | 2453328.030048     | 249342.408182                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.571770     | 2410356.936518     | 246603.648252                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.775991     | 2223878.166903     | 245062.976802                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.766594     | 2409639.643296     | 247506.590027                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.694886     | 2410999.432515     | 249090.114714                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.775102     | 2403520.208238     | 249966.803546                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.595800     | 2423351.452495     | 248413.891990                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.771221     | 2419339.931375     | 249122.212221                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.684762     | 2350168.913286     | 245948.889329                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.684434     | 2484322.085029     | 250669.158521                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.584840     | 2488532.444233     | 250223.679393                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.599456     | 2446377.535404     | 247491.937273                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.671870     | 2253329.593333     | 245956.084807                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.775555     | 2446384.745000     | 248448.865245                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.687742     | 2448925.789860     | 250063.174242                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.680237     | 2488892.926483     | 251062.408344                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.605289     | 2459436.871449     | 249379.368945                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.679675     | 2453758.139689     | 250031.990297                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.678557     | 2398915.586957     | 246930.925832                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.611545     | 2372173.819430     | 254159.338256                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.666395     | 2380022.178272     | 253684.176281                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.493022     | 2365887.756510     | 250926.035226                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.680474     | 2321744.378136     | 249335.485191                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.666593     | 2338668.102477     | 251962.454286                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.879769     | 2343447.046000     | 253627.725900                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.579635     | 2490307.633523     | 254702.949432                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.585071     | 2352163.093710     | 252961.059308                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.558745     | 2344145.446343     | 253473.697066                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.704872     | 2328940.236213     | 250422.259778                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.494398     | 2652942.693723     | 266922.830566                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.679922     | 2642964.750193     | 266483.529108                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.680736     | 2520083.949746     | 262980.791453                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.878949     | 2667823.241346     | 261883.662524                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.668122     | 2673362.563281     | 264668.199394                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.692283     | 2665689.376297     | 266396.326807                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.665177     | 2462633.764978     | 265833.916913                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.583092     | 2659135.148955     | 265833.266876                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.678668     | 2669017.653909     | 266070.319463                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.609583     | 2701478.011602     | 262802.615244                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.483470     | 3420363.400299     | 297705.301193                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.573248     | 3414191.902158     | 297593.099075                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.685796     | 3063315.609645     | 292723.436708                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.788229     | 3436647.994586     | 292341.878250                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.581841     | 3426657.996692     | 295451.042363                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.774547     | 3418047.501306     | 297380.349434                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.587230     | 3086357.224657     | 291202.686303                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.671942     | 3413749.103727     | 297039.061604                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.596258     | 3423268.964967     | 296807.851883                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.583969     | 3452982.712531     | 292577.175660                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.683979     | 4753927.084729     | 333347.681991                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.664167     | 4759706.136913     | 333640.800647                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.664001     | 4734832.226546     | 328531.718730                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.595717     | 4785235.275091     | 327351.346906                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.591559     | 4748244.020607     | 331508.448147                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.777214     | 4731503.522185     | 333901.354382                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.672027     | 3859267.406333     | 324776.685796                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.665161     | 4726425.182825     | 333776.078422                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.667136     | 4741597.831604     | 332959.823788                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.585255     | 4718267.515858     | 326994.076011                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.668833     | 5752250.866971     | 355401.718722                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.707124     | 5757070.724554     | 355842.056712                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.575673     | 5761209.698545     | 350776.248626                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.871719     | 5769030.545041     | 348851.381891                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.692556     | 5748618.802458     | 354013.021098                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.675094     | 5740655.207437     | 356795.467205                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.587145     | 4956593.541618     | 350017.788696                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.767542     | 5737963.189210     | 356786.481185                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.772931     | 5746086.436983     | 355571.778485                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.881656     | 5644996.029370     | 348195.194827                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.588876     | 6491389.506948     | 365601.449343                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.681762     | 6501940.326636     | 366020.746240                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.606535     | 6241306.047274     | 360648.324430                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.674134     | 6507475.769400     | 358759.248061                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.566037     | 6486833.867420     | 364460.910646                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.682107     | 6480051.893274     | 367436.806928                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.672990     | 5890327.196297     | 364129.590760                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.489683     | 6477802.816843     | 367476.364721                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.702081     | 6484774.003856     | 366077.471757                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.513108     | 6521292.267593     | 357936.789122                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.680736     | 6925573.008212     | 369633.621424                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.662784     | 6929292.040167     | 370016.635097                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.775373     | 6428125.911528     | 364385.860534                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.566066     | 6933262.408242     | 362672.154746                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.686724     | 6924293.449479     | 368595.175548                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.661445     | 6921731.640741     | 371646.665038                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.671437     | 6395274.614548     | 370346.090426                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.692088     | 6920283.672659     | 371704.228047                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.574084     | 6924155.154823     | 370235.171124                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.695841     | 6938231.221083     | 361769.513984                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.786925     | 7083162.371879     | 371042.857543                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.596746     | 7084297.315758     | 371408.793624                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.671263     | 6492478.455370     | 365664.490439                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.667324     | 7087786.195489     | 364039.452495                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.576414     | 7083081.603983     | 370040.308246                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.572405     | 7082047.225866     | 373117.840877                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.576099     | 6907268.099656     | 372610.460111                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17457</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.871929     | 7080885.513255     | 373181.521695                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.772381     | 7083639.746520     | 371688.502168                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17456</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.488341     | 7089496.696001     | 363106.279237                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    [6.24702191e+14 3.59522656e+14 1.82500743e+14 1.24345199e+14
     1.20963608e+14 1.23921949e+14 1.37124115e+14 1.71719393e+14
     2.29172268e+14 2.52982619e+14 2.58749756e+14 2.62867019e+14
     2.64926582e+14]


***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  


```python
# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
plt.plot(a,error_vector,'k-')
plt.xlabel('$\ell_2$ penalty')
plt.ylabel('cross validation error')
plt.xscale('log')
plt.yscale('log')
print(a)
print(error_vector)
```

    [1.00000000e+01 3.16227766e+01 1.00000000e+02 3.16227766e+02
     1.00000000e+03 3.16227766e+03 1.00000000e+04 3.16227766e+04
     1.00000000e+05 3.16227766e+05 1.00000000e+06 3.16227766e+06
     1.00000000e+07]
    [6.24702191e+14 3.59522656e+14 1.82500743e+14 1.24345199e+14
     1.20963608e+14 1.23921949e+14 1.37124115e+14 1.71719393e+14
     2.29172268e+14 2.52982619e+14 2.58749756e+14 2.62867019e+14
     2.64926582e+14]



![png](output_61_1.png)


Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`. This way, your final model will be trained on the entire dataset.


```python
l2_penalty = 1e3
data = polynomial_sframe(train_valid['sqft_living'], 15)
my_features = data.column_names() # get the name of the features
data['price'] = train_valid['price'] # add price to the data since it's the target
final_model = turicreate.linear_regression.create(data, target = 'price', features = my_features,l2_penalty=l2_penalty, validation_set = None)

predict = final_model.predict(test)
errors = test['price'] - predict
square_errors = errors ** 2
RSS = square_errors.sum()
print(RSS)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 19396</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.369408     | 2461778.986191     | 248914.007014                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    252897427447157.5


***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***


```python

```
