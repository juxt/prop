# prop

prop is a Clojure library for writing recursive neural networks based on Backpropagation and the Logist Function. It utitlises core.matrix to allow you to write a multi-layered nueral network with any number of nuerons in the hidden layers.

There is also a feature of adding biases to the neurons in the hidden layers and the outputs. These biases are added by the user and range from 0 to 1, with 0 being no bias.

My Blog on [Machine Learning](https://juxt.pro/blog/posts/machine-learning.html) and [Deep Learning](https://juxt.pro/blog/posts/deep-learning.html) which go into more detail about Backpropagation and the Logist Function. 

[Here](https://juxt.pro/blog/posts/neural-maths.html) is my blog on the Maths and ideas behind this Recursive Nueral Network.

You can find a detailed explaination of how I wrote this Neural Network [here]().

## Installation:

Add the following dependency to your `project.clj` or ` build.boot` file:

```
[prop "1.0.0"]
```

## Using the Neural Network

Require in your namespace

```clojure
(:require [prop.core :as prop])
```

Alternatively, require in the repl
```clojure
(require '[prop.core :as prop])
```

### Constructing a Neural Network

```clojure
(def nn (prop/nn [0 0 1] [0.1] {0.1 0.11} {0.3 0.33 :num 1} {0.6 0.66 0.8 0.88 :num 2} {0.8 0.88 0.9 0.99 :num 2}))
```

We have created a neural network that has an input of 3 neurons, `0`, `0`, `1` and output of 1 neuron `0.1`. To add another input or output, add the value into the respective vector. 

`{0.1 0.11}` is the bias-weight and bias on the output neuron, the `key` being the bias-weight and the `val` being the bias value. Note that the number of key-value pairs in this vector must be equal to the number of values of the output vector.

The final 3 arguments represent the 3 hidden layers. For another hidden layer simply add another map. `{0.3 0.33 :num 1}` tells us that this hidden layer has 1 nueron, and the bias-weight (`key`) is `0.3` and the bias value (`val`) is `0.33`. To add another neuron to this hidden layer, add another key-value pair and update the `:num`

### Training the Neural Network

When training the Neural Network, you have the choice of the having the error and iteration printed at each iteration or just the final result

With Error and Iterations:

```clojure
(prop/trained-nn nn 0.5 10 true)
```
```
Total error is:  0.1684661161803551 Interation is:  0
Total error is:  0.15752415668372022 Interation is:  1
Total error is:  0.1467385655076329 Interation is:  2
Total error is:  0.13621841666686685 Interation is:  3
Total error is:  0.12606294004762203 Interation is:  4
Total error is:  0.11635667169660045 Interation is:  5
Total error is:  0.10716610500800687 Interation is:  6
Total error is:  0.09853811184099649 Interation is:  7
Total error is:  0.09050009311610498 Interation is:  8
Total error is:  0.0830615631948953 Interation is:  9
{:output [0.1],
 :calc-output [0.4723014944380909],
 :output-errors [0.06930420138041792],
 :total-error 0.06930420138041792,
 :weights
 [[[0.31937510630652] [0.5116851634216558] [0.8724308142797698]]
  [[0.4074469683918901 0.9030234851054735]]
  [[0.201389551317126 0.1031079822069274]
   [0.09874808792495016 0.7984895019078827]]
  [[0.49644754567036664] [0.2293279675945864]]]}
```

Without Error and Iterations:
```clojure
(prop/trained-nn nn 0.5 10 false)
```
```
{:output [0.1],
 :calc-output [0.4723014944380909],
 :output-errors [0.06930420138041792],
 :total-error 0.06930420138041792,
 :weights
 [[[0.31937510630652] [0.5116851634216558] [0.8724308142797698]]
  [[0.4074469683918901 0.9030234851054735]]
  [[0.201389551317126 0.1031079822069274]
   [0.09874808792495016 0.7984895019078827]]
  [[0.49644754567036664] [0.2293279675945864]]]}
```

`nn` is the neural network we created. `0.5` is the learning rate (0.5 is often the default but the learning rate can be any number between 0 and 1). `10` is the number of iterations. `true/false` is whether we want the printlns or not.

## Acknowledgments

Thank you to the following people for inspiration, contributions, feedback and suggestions.
* Rickesh Bedia
* Dominic Monroe
* Patrick Karlin
* Matt Butler

## Copyright & License

The MIT License (MIT)

Copyright © 2017-2018 JUXT LTD.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
