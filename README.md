# machine-learning-2-week-10-dl2-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 10-DL2 Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-10-dl2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98846&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 10-DL2 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
&nbsp;

Exercise Sheet 10

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 1: Mixture Density Networks (20 + 10 P)

In this exercise, we prove some of the results from the paper Mixture Density Networks by Bishop (1994). The mixture density network is given by

</div>
</div>
<div class="layoutArea">
<div class="column">
with the mixture elements

</div>
<div class="column">
m

p(t|x) = 􏰄 αi(x)φi(t|x)

i=1

1 exp 􏰁 − ∥t − μi(x)∥2 􏰂. (2π)c/2σi(x)c 2σi(x)2

</div>
</div>
<div class="layoutArea">
<div class="column">
φi(t|x) =

The contribution to the error function of one data point q is given by

</div>
</div>
<div class="layoutArea">
<div class="column">
We also define the posterior distribution

</div>
<div class="column">
i=1

αi φi πi(x,t) = 􏰃mj=1 αjφj

</div>
</div>
<div class="layoutArea">
<div class="column">
m

Eq =−log􏰗􏰄αi(xq)φi(tq|xq)􏰘

</div>
</div>
<div class="layoutArea">
<div class="column">
which is obtained using the Bayes theorem.

(a) Compute the gradient of the error Eq w.r.t. the mixture parameters, i.e. show that

∂Eq πi (i) ∂α =−α

ii

∂Eq 􏰁μik −tk􏰂 (ii)∂μ =πi σ2

ik i

(b) We now assume that the neural network produces the mixture coefficients as: exp(ziα )

where zα denotes the outputs of the neural network (after the last linear layer) associated to these mixture coefficients. Compute using the chain rule for derivatives (i.e. by reusing some of the results in the first part of this exercise) the derivative ∂Eq/∂ziα.

Exercise 2: Conditional RBM (20 + 10 P)

The conditional restricted Boltzmann machine is a system of binary variable comprising inputs x ∈ {0, 1}d, outputs y ∈ {0,1}c, and hidden units h ∈ {0,1}K. It associates to each configuration of these binary variables the energy:

E(x,y,h) = −x⊤Wh − y⊤Uh and the probability associated to each configuration is then given as:

p(x, y, h) = Z1 exp(−E(x, y, h)) where Z is a normalization constant that makes probabilities sum to one.

</div>
</div>
<div class="layoutArea">
<div class="column">
αi = 􏰃Mj=1 exp(zjα)

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
(a) Let sigm(t) = exp(t)/(1 + exp(t)) be the sigmoid function. Show that (i) p(hk = 1 | x, y) = sigm􏰍x⊤W:,k + y⊤U:,k􏰎

</div>
</div>
<div class="layoutArea">
<div class="column">
(ii) p(yj = 1|h,x) = sigm􏰍U⊤ h􏰎 j,:

(b) Show that

where

</div>
</div>
<div class="layoutArea">
<div class="column">
p(x, y) = Z1 exp(−F (x, y))

K F(x,y)=−􏰄log􏰍1+exp􏰍x⊤W:,k +y⊤U:,k􏰎􏰎

</div>
</div>
<div class="layoutArea">
<div class="column">
k=1

is the free energy and where Z is again a normalization constant.

Exercise 3: Programming (40 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 10 (programming) [SoSe 2021] Machine Learning 2

</div>
</div>
<div class="layoutArea">
<div class="column">
MNIST Inpainting with Energy-Based Learning

In this exercise, we consider the task of inpainting incomplete handwritten digits, and for this, we would like to make use of neural networks and the Energy-Based Learning framework.

In [1]: import torch

import torch.nn as nn

import utils import numpy %matplotlib inline

As a first step, we load the MNIST dataset

In [2]: Xr,Xt = utils.getdata()

We consider the following perturbation process that draws some region near the center of the image randomly and set the pixels in this area to some gray value.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [3]: def

</div>
<div class="column">
removepatch(X):

mask = torch.zeros(len(X),28,28) for i in range(len(X)):

<pre>    j = numpy.random.randint(-4,5)
    k = numpy.random.randint(-4,5)
    mask[i,11+j:17+j,11+k:17+k] = 1
</pre>
mask = mask.view(len(X),784) return (X*(1-mask)).data,mask

</div>
</div>
<div class="layoutArea">
<div class="column">
The outcome of the perturbation process can be visualized below:

In [4]: %matplotlib inline

xmask = removepatch(Xt[:10])[0]

<pre>              utils.vis10(xmask)
</pre>
PCA Reconstruction (20 P)

A simple technique for impainting an image is principal component analysis. It consists of taking the incomplete image and projecting it on the d principal components of the training data.

Task:

Implement a function that takes a collection of test examples z and projects them on the d principal components of the training data x .

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
In [5]: def pca(z,x,d):

</div>
</div>
<div class="layoutArea">
<div class="column">
# ——————————- # TODO: replace by your code

# ——————————- import solution

<pre>                  y = solution.pca(z,x,d)
</pre>
<pre>                  # -------------------------------
</pre>
return y

The PCA-based inpainting technique is tested below on 10 test points for which a patch is missing. We observe that the patch-like perturbation is less severe when d is low, but the reconstructed part of the digit appears blurry. Conversely, if setting d high, more details become available, but the missing pattern appears more prominent.

In [6]: Xn,m = removepatch(Xt[:10])

<pre>              utils.vis10(pca(Xn,Xr,10)*m+Xn*(1-m))
              utils.vis10(pca(Xn,Xr,60)*m+Xn*(1-m))
              utils.vis10(pca(Xn,Xr,360)*m+Xn*(1-m))
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Energy-Based Learning (20 P)

We now consider the energy-based learning framework where we learn an energy function to discriminate between correct and incorrect reconstructions.

In [7]: torch.manual_seed(0) enet = nn.Sequential(

<pre>                  nn.Linear(784,256),nn.Hardtanh(),
                  nn.Linear(256,256),nn.Hardtanh(),
                  nn.Linear(256,1),
</pre>
)

To be able to generate good contrastive examples (i.e. incorrect reconstructions that are still plausible enough to confuse the energy- based model and for which meaningful gradient signal can be extracted), we consider a generator network that takes as input the incomplete images.

In [8]: gnet = nn.Sequential( nn.Linear(784,256),nn.Hardtanh(),

<pre>                  nn.Linear(256,256),nn.Hardtanh(),
</pre>
<pre>                  nn.Linear(256,784),nn.Hardtanh()
              )
</pre>
The whole architecture is depicted in the diagram below:

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
The two networks are then jointly optimized. The structure of the optimization problem is already provided to you, however, the code that computes the forward pass from the input data up to the error function are missing.

Task:

Write the code that computes the error function. Here, we use a single optimizer and must therefore implement the gradient flip trick described in the slides. A similar trick can be used to only let the gradient flow into the generator only via the missing image patch and not through all pixels.

</div>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
In [9]: import torch.optim as optim N = 10000

mb = 100

optimizer = optim.SGD(list(enet.parameters())+list(gnet.parameters()), lr=0.05) for epoch in numpy.arange(100):

for i in range(N//mb):

<pre>                      optimizer.zero_grad()
</pre>
<pre>                      # Take a minibatch and train it
</pre>
<pre>                      x   = Xr[mb*i:mb*(i+1)].data*1.0
                      z,m = removepatch(x)
</pre>
<pre>                      # Build the forward pass from the input until the loss function
</pre>
# ——————————- # TODO: replace by your code

# ——————————- import solution

<pre>                      err = solution.err(x,m,z,gnet,enet)
</pre>
<pre>                      # -------------------------------
</pre>
<pre>                      # Compute the gradient and perform one step of gradient descent
</pre>
<pre>                      err.backward()
                      optimizer.step()
</pre>
if epoch%10==0: print(epoch,err)

<pre>              /home/gregoire/.local/lib/python3.8/site-packages/torch/autograd/__init__.py:130: Use
              rWarning: CUDA initialization: The NVIDIA driver on your system is too old (found ver
              sion 10010). Please update your GPU driver by downloading and installing a new versio
              n from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: http
              s://pytorch.org to install a PyTorch version that has been compiled with your version
              of the CUDA driver. (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:10
              0.)
</pre>
<pre>                Variable._execution_engine.run_backward(
</pre>
<pre>              0 tensor(0.6744, grad_fn=&lt;MeanBackward0&gt;)
              10 tensor(0.1035, grad_fn=&lt;MeanBackward0&gt;)
              20 tensor(0.2146, grad_fn=&lt;MeanBackward0&gt;)
              30 tensor(0.3614, grad_fn=&lt;MeanBackward0&gt;)
              40 tensor(0.3134, grad_fn=&lt;MeanBackward0&gt;)
              50 tensor(0.3515, grad_fn=&lt;MeanBackward0&gt;)
              60 tensor(0.4389, grad_fn=&lt;MeanBackward0&gt;)
              70 tensor(0.3787, grad_fn=&lt;MeanBackward0&gt;)
              80 tensor(0.4541, grad_fn=&lt;MeanBackward0&gt;)
              90 tensor(0.4365, grad_fn=&lt;MeanBackward0&gt;)
</pre>
After optimizing for a sufficient number of epochs, the solution has ideally come close to some nash equilibrium where both the generator and energy-based model perform well. In particular, the generator should generate examples that look similar to the true examples. The code below plots the incomplete digits and the reconstruction obtained by the generator network.

In [10]: x = Xt[:10]

z,m = removepatch(x)

<pre>              utils.vis10(z)
              utils.vis10(gnet(z)*m+z*(1-m))
</pre>
</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="section">
<div class="layoutArea">
<div class="column">
As we can see, although some artefacts still persist, the reconstructions are quite plausible and look better than those one gets with the simple PCA-based approach. Note however that the procedure is also more complex and computationally more demanding than a simple PCA-based reconstruction.

</div>
</div>
</div>
</div>
<div class="page" title="Page 8"></div>
<div class="page" title="Page 9"></div>
