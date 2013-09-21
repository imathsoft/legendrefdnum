legendrefdnum
=============

`legendrefdnum` is a Python 2.x module for solving Sturm-Liouville problems of the type

![-\frac{d}{d x}\left[(1-x^{2})\frac{d u(x)}{d x}\right]+q(x)u(x)=\lambda u(x),\quad x\in(-1, 1),](http://www.imath.kiev.ua/~dbohdan/legendrefdnum/eq1.png)

![\lim\limits_{x\rightarrow \pm 1}(1-x^{2})\frac{d u(x)}{d x}=0.](http://www.imath.kiev.ua/~dbohdan/legendrefdnum/eq2.png)


It has been proved to be exponentially convergent for

![\|q\|_{1, \rho}=\int\limits_{-1}^{1}\frac{|q(x)|}{\sqrt{1-x^{2}}}d x<\infty.](http://www.imath.kiev.ua/~dbohdan/legendrefdnum/eq3.png)

The module was developed based on the algorithm published in the article *Volodymyr L. Makarov, Denys V. Dragunov, Danyil V. Bohdan. Exponentially convergent numerical-analytical method for solving eigenvalue problems for singular differential operators.* A preprint of this article will soon be availible from <http://arxiv.org/>.

`legendrefdnum` comes extensively commented and with example code. You don't have to understand the FD-method to use `legendrefdnum`.


Running
-------
The following modules are mandatory dependencies: `numpy`, `scipy`, `matplotlib`. Having `mpmath` installed is optional but highly recommended for calculations with dense meshes (see `legendrepqwrappers.py` for an explanation).

To run `legendrefdnum` on Ubuntu 12.04 or later do the following:

1. Clone the GitHub repository.

2. Execute the command `sudo apt-get install python-numpy python-scipy python-mpmath python-matplotlib` in the terminal.

3. Run `example1.py` to make sure everything works.

License
-------
GNU LGPLv2.1+
