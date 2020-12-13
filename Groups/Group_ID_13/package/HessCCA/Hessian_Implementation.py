{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sympy import *\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "c,d,a,b,m,x,y,z,pi,r,t,p,h,e,t = symbols(\"c d a b m x y z pi r t p h e t\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 h p \\pi r t + 2 p \\pi r^{2} t$"
      ],
      "text/plain": [
       "2*h*p*pi*r*t + 2*p*pi*r**2*t"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m = 2 *pi*r**2*t*p+2*pi*r*h*t*p\n",
    "m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 h p \\pi t + 4 p \\pi r t$"
      ],
      "text/plain": [
       "2*h*p*pi*t + 4*p*pi*r*t"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff_mr = diff(m,r)\n",
    "diff_mr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 p \\pi r t$"
      ],
      "text/plain": [
       "2*p*pi*r*t"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff_mh = diff(m,h)\n",
    "diff_mh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 h p \\pi r + 2 p \\pi r^{2}$"
      ],
      "text/plain": [
       "2*h*p*pi*r + 2*p*pi*r**2"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff_mt = diff(m,t)\n",
    "diff_mt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 h \\pi r t + 2 \\pi r^{2} t$"
      ],
      "text/plain": [
       "2*h*pi*r*t + 2*pi*r**2*t"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff_mp = diff(m,p)\n",
    "diff_mp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{y z^{2}} \\sin{\\left(x \\right)}$"
      ],
      "text/plain": [
       "e**(y*z**2)*sin(x)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxyz = sin(x)*e**(y*(z**2))\n",
    "fxyz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{y z^{2}} \\cos{\\left(x \\right)}$"
      ],
      "text/plain": [
       "e**(y*z**2)*cos(x)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fx= diff(fxyz,x)\n",
    "fx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{y z^{2}} z^{2} \\log{\\left(e \\right)} \\sin{\\left(x \\right)}$"
      ],
      "text/plain": [
       "e**(y*z**2)*z**2*log(e)*sin(x)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fy= diff(fxyz,y)\n",
    "fy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 e^{y z^{2}} y z \\log{\\left(e \\right)} \\sin{\\left(x \\right)}$"
      ],
      "text/plain": [
       "2*e**(y*z**2)*y*z*log(e)*sin(x)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fz= diff(fxyz,z)\n",
    "fz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 e^{y z^{2}} z \\left(y z^{2} \\log{\\left(e \\right)} + 1\\right) \\log{\\left(e \\right)} \\cos{\\left(x \\right)}$"
      ],
      "text/plain": [
       "2*e**(y*z**2)*z*(y*z**2*log(e) + 1)*log(e)*cos(x)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_all = diff(fxyz, x, y, z)\n",
    "f_all"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\pi x^{3} + x y^{2} + y^{4} \\left(2 h p \\pi r t + 2 p \\pi r^{2} t\\right)$"
      ],
      "text/plain": [
       "pi*x**3 + x*y**2 + y**4*(2*h*p*pi*r*t + 2*p*pi*r**2*t)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxy = pi*x**3+x*y**2+m*y**4\n",
    "fxy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 3 \\pi x^{2} + y^{2}$"
      ],
      "text/plain": [
       "3*pi*x**2 + y**2"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fx = diff(fxy, x)\n",
    "fx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x y + 4 y^{3} \\left(2 h p \\pi r t + 2 p \\pi r^{2} t\\right)$"
      ],
      "text/plain": [
       "2*x*y + 4*y**3*(2*h*p*pi*r*t + 2*p*pi*r**2*t)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fy = diff(fxy, y)\n",
    "fy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2} y + x z^{2} + y^{2} z$"
      ],
      "text/plain": [
       "x**2*y + x*z**2 + y**2*z"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxyz = x**2*y+y**2*z+z**2*x\n",
    "fxyz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x y + z^{2}$"
      ],
      "text/plain": [
       "2*x*y + z**2"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fx = diff(fxyz, x)\n",
    "fx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2} + 2 y z$"
      ],
      "text/plain": [
       "x**2 + 2*y*z"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fy = diff(fxyz, y)\n",
    "fy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x z + y^{2}$"
      ],
      "text/plain": [
       "2*x*z + y**2"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fz = diff(fxyz, z)\n",
    "fz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{2 x} z^{2} \\sin{\\left(y \\right)} + e^{x} e^{y} \\cos{\\left(z \\right)}$"
      ],
      "text/plain": [
       "e**(2*x)*z**2*sin(y) + e**x*e**y*cos(z)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fzxy = e**(2*x)*sin(y)*(z**2)+cos(z)*(e**x) * (e**y)\n",
    "fzxy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 e^{2 x} z^{2} \\log{\\left(e \\right)} \\sin{\\left(y \\right)} + e^{x} e^{y} \\log{\\left(e \\right)} \\cos{\\left(z \\right)}$"
      ],
      "text/plain": [
       "2*e**(2*x)*z**2*log(e)*sin(y) + e**x*e**y*log(e)*cos(z)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fx = diff(fzxy, x)\n",
    "fx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{2 x} z^{2} \\cos{\\left(y \\right)} + e^{x} e^{y} \\log{\\left(e \\right)} \\cos{\\left(z \\right)}$"
      ],
      "text/plain": [
       "e**(2*x)*z**2*cos(y) + e**x*e**y*log(e)*cos(z)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fy = diff(fzxy, y)\n",
    "fy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 e^{2 x} z \\sin{\\left(y \\right)} - e^{x} e^{y} \\sin{\\left(z \\right)}$"
      ],
      "text/plain": [
       "2*e**(2*x)*z*sin(y) - e**x*e**y*sin(z)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fz = diff(fzxy, z)\n",
    "fz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 e^{2 x} z \\sin{\\left(y \\right)} - e^{x} e^{y} \\sin{\\left(z \\right)}$"
      ],
      "text/plain": [
       "2*e**(2*x)*z*sin(y) - e**x*e**y*sin(z)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fyx = diff(fzxy,z)\n",
    "fyx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\frac{\\sqrt{x}}{y}$"
      ],
      "text/plain": [
       "sqrt(x)/y"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fyx = sqrt(x)/y\n",
    "fyx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\frac{\\sqrt{t}}{\\sin{\\left(t \\right)}}$"
      ],
      "text/plain": [
       "sqrt(t)/sin(t)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "insert = fyx.subs(x, t).subs(y, sin(t))\n",
    "insert"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle - \\frac{\\sqrt{t} \\cos{\\left(t \\right)}}{\\sin^{2}{\\left(t \\right)}} + \\frac{1}{2 \\sqrt{t} \\sin{\\left(t \\right)}}$"
      ],
      "text/plain": [
       "-sqrt(t)*cos(t)/sin(t)**2 + 1/(2*sqrt(t)*sin(t))"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "insert_diff = diff(insert, t)\n",
    "insert_diff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{2 z} \\sin{\\left(y \\right)} \\cos{\\left(x \\right)}$"
      ],
      "text/plain": [
       "e**(2*z)*sin(y)*cos(x)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxyz = cos(x)*sin(y)*e**(2*z)\n",
    "fxyz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle e^{2 t^{2}} \\sin{\\left(t - 1 \\right)} \\cos{\\left(t + 1 \\right)}$"
      ],
      "text/plain": [
       "e**(2*t**2)*sin(t - 1)*cos(t + 1)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_subs = fxyz.subs(x, t+1).subs(y, t-1).subs(z, (t**2))\n",
    "f_subs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 4 e^{2 t^{2}} t \\log{\\left(e \\right)} \\sin{\\left(t - 1 \\right)} \\cos{\\left(t + 1 \\right)} - e^{2 t^{2}} \\sin{\\left(t - 1 \\right)} \\sin{\\left(t + 1 \\right)} + e^{2 t^{2}} \\cos{\\left(t - 1 \\right)} \\cos{\\left(t + 1 \\right)}$"
      ],
      "text/plain": [
       "4*e**(2*t**2)*t*log(e)*sin(t - 1)*cos(t + 1) - e**(2*t**2)*sin(t - 1)*sin(t + 1) + e**(2*t**2)*cos(t - 1)*cos(t + 1)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diff(f_subs, t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2} y z$"
      ],
      "text/plain": [
       "x**2*y*z"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#theHessian\n",
    "fxyz = x**2*y*z\n",
    "fxyz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x y z$"
      ],
      "text/plain": [
       "2*x*y*z"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_x = diff(fxyz, x)\n",
    "f_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2} z$"
      ],
      "text/plain": [
       "x**2*z"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_y = diff(fxyz, y)\n",
    "f_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2} y$"
      ],
      "text/plain": [
       "x**2*y"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_z = diff(fxyz, z)\n",
    "f_z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\left[\\begin{matrix}2 x y z & x^{2} z & x^{2} y\\end{matrix}\\right]$"
      ],
      "text/plain": [
       "Matrix([[2*x*y*z, x**2*z, x**2*y]])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jacob = Matrix([[f_x,f_y,f_z]])\n",
    "jacob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 y z$"
      ],
      "text/plain": [
       "2*y*z"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jx_x = diff(f_x, x)\n",
    "jx_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x z$"
      ],
      "text/plain": [
       "2*x*z"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jx_y = diff(f_x, y)\n",
    "jx_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x y$"
      ],
      "text/plain": [
       "2*x*y"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jx_z = diff(f_x, z)\n",
    "jx_z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x z$"
      ],
      "text/plain": [
       "2*x*z"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jy_x = diff(f_y, x)\n",
    "jy_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 0$"
      ],
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jy_y = diff(f_y, y)\n",
    "jy_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2}$"
      ],
      "text/plain": [
       "x**2"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jy_z = diff(f_y, z)\n",
    "jy_z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 2 x y$"
      ],
      "text/plain": [
       "2*x*y"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jz_x = diff(f_z, x)\n",
    "jz_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{2}$"
      ],
      "text/plain": [
       "x**2"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jz_y = diff(f_z, y)\n",
    "jz_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 0$"
      ],
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jz_z = diff(f_z, z)\n",
    "jz_z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\left[\\begin{matrix}2 y z & 2 x z & 2 x y\\\\2 x z & 0 & x^{2}\\\\2 x y & x^{2} & 0\\end{matrix}\\right]$"
      ],
      "text/plain": [
       "Matrix([\n",
       "[2*y*z, 2*x*z, 2*x*y],\n",
       "[2*x*z,     0,  x**2],\n",
       "[2*x*y,  x**2,     0]])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hess = Matrix([[jx_x, jx_y, jx_z],\n",
    "                [jy_x, jy_y, jy_z],\n",
    "               [jz_x, jz_y, jz_z]])\n",
    "hess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{3} y + x + 2 y$"
      ],
      "text/plain": [
       "x**3*y + x + 2*y"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#calculating hessian\n",
    "fxy = x**3*y+x+2*y\n",
    "fxy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 3 x^{2} y + 1$"
      ],
      "text/plain": [
       "3*x**2*y + 1"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxy_x = diff(fxy, x)\n",
    "fxy_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle x^{3} + 2$"
      ],
      "text/plain": [
       "x**3 + 2"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fxy_y = diff(fxy, y)\n",
    "fxy_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\left[\\begin{matrix}3 x^{2} y + 1 & x^{3} + 2\\end{matrix}\\right]$"
      ],
      "text/plain": [
       "Matrix([[3*x**2*y + 1, x**3 + 2]])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jacob = Matrix([[fxy_x, fxy_y]])\n",
    "jacob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 6 x y$"
      ],
      "text/plain": [
       "6*x*y"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jx_x = diff(fxy_x, x)\n",
    "jx_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 3 x^{2}$"
      ],
      "text/plain": [
       "3*x**2"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jx_y = diff(fxy_x, y)\n",
    "jx_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 3 x^{2}$"
      ],
      "text/plain": [
       "3*x**2"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jy_x = diff(fxy_y, x)\n",
    "jy_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle 0$"
      ],
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jy_y = diff(fxy_y, y)\n",
    "jy_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$\\displaystyle \\left[\\begin{matrix}6 x y & 3 x^{2}\\\\3 x^{2} & 0\\end{matrix}\\right]$"
      ],
      "text/plain": [
       "Matrix([\n",
       "[ 6*x*y, 3*x**2],\n",
       "[3*x**2,      0]])"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hess = Matrix([[jx_x, jx_y],\n",
    "              [jy_x, jy_y]])\n",
    "hess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
