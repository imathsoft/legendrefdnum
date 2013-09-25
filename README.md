legendrefdnum
=============

`legendrefdnum` is a Python 2.x module for solving Sturm-Liouville problems of the type

![-\frac{d}{d x}\left[(1-x^{2})\frac{d u(x)}{d x}\right]+q(x)u(x)=\lambda u(x),\quad x\in(-1, 1),](http://www.sciweavers.org/tex2img.php?eq=-%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft%5B%281-x%5E%7B2%7D%29%5Cfrac%7Bd%20u%28x%29%7D%7Bd%20x%7D%5Cright%5D%2Bq%28x%29u%28x%29%3D%5Clambda%20u%28x%29%2C%5Cquad%20x%5Cin%28-1%2C%201%29%2C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![\lim\limits_{x\rightarrow \pm 1}(1-x^{2})\frac{d u(x)}{d x}=0.](http://www.sciweavers.org/tex2img.php?eq=%5Clim%5Climits_%7Bx%5Crightarrow%20%5Cpm%201%7D%281-x%5E%7B2%7D%29%5Cfrac%7Bd%20u%28x%29%7D%7Bd%20x%7D%3D0.&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)


It has been proved to be exponentially convergent for

![\|q\|_{1, \rho}=\int\limits_{-1}^{1}\frac{|q(x)|}{\sqrt{1-x^{2}}}d x<\infty.](http://www.sciweavers.org/tex2img.php?eq=%20%20%20%20%5C%7Cq%5C%7C_%7B1%2C%20%5Crho%7D%3D%5Cint%5Climits_%7B-1%7D%5E%7B1%7D%5Cfrac%7B%7Cq%28x%29%7C%7D%7B%5Csqrt%7B1-x%5E%7B2%7D%7D%7Dd%20x%3C%5Cinfty.&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

The module was developed based on the algorithm published in the article *Volodymyr L. Makarov, Denys V. Dragunov, Danyil V. Bohdan. Exponentially convergent numerical-analytical method for solving eigenvalue problems for singular differential operators.* A preprint of this article will soon be availible from <http://arxiv.org/>.

`legendrefdnum` comes extensively commented and with example code. You don't have to understand the FD-method to use `legendrefdnum`.


Running
-------
The following modules are mandatory dependencies: `numpy`, `scipy`, `matplotlib`. Having `mpmath` installed is optional but highly recommended for calculations with dense meshes (see `legendrepqwrappers.py` for an explanation).

To run `legendrefdnum` on Ubuntu 12.04 or later do the following:

1. Clone the GitHub repository.

2. Execute the command `sudo apt-get install python-numpy python-scipy python-mpmath python-matplotlib` in the terminal.

3. Run `example1.py` from the cloned repository to make sure everything works.

License
-------
GNU LGPLv2.1+
